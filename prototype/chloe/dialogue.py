"""
The conversational loop.

Implements the original design's core cycle as far as a single-session prototype
reasonably can:

    1. Converse.                  -> ChloeEngine.turn()
    2. Acquire candidate knowledge. -> _handle_statement / _handle_negation
    3. Consolidate during "sleep".   -> consolidation.sleep() (separate module)
    4. Generate hypotheses.          -> consolidation.sleep()
    5. Verify through future conversation. -> _ask_pending_question / _handle_yn_question
    6. Revise beliefs and source reliability. -> trust.py + this module

Every exchange is logged as an Interaction and every stored belief carries
a Provenance record back to the interaction and person that produced it,
per CHLOE.md's "epistemic transparency" principle -- you can always answer
"who said this, and when."
"""

from enum import Enum
from typing import Optional

from . import trust as trust_mod
from .models import Atom, AtomStatus, Interaction, Person, Provenance
from .llm_nlu import parse  # LLM-first routing parser; same seam as nlu.parse
from .nlu import Utterance, UtteranceType
from .storage import KnowledgeStore

_YES = {"yes", "yeah", "yep", "sure", "ok", "okay", "y"}
_NO = {"no", "nope", "nah", "n"}


class AuthState(str, Enum):
    """Where an in-progress identity check stands. NONE means normal
    conversation; the other three are multi-turn detours that greet()/turn()
    walk through before self.person is bound to anything."""

    NONE = "none"
    AWAITING_SECRET_CHOICE = "awaiting_secret_choice"   # new person: "want to set a secret word?"
    AWAITING_SECRET_VALUE = "awaiting_secret_value"     # new person, said yes: waiting for the word itself
    AWAITING_SECRET_VERIFY = "awaiting_secret_verify"   # returning name with a secret on file: waiting for it


class ChloeEngine:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.person: Optional[Person] = None
        self.auth_state: AuthState = AuthState.NONE
        self._pending_name: Optional[str] = None
        self._secret_attempts = 0
        self.MAX_SECRET_ATTEMPTS = 3

    # ------------------------------------------------------------- greeting
    def greet(self, name: str) -> str:
        """Look the name up rather than always creating/binding immediately:
        a name alone isn't proof of identity. Three cases:

        1. Never seen this name before -> greet normally, then offer to set
           up a secret word for next time (self-service, opt-in).
        2. Known name, no secret on file (pre-feature person, or they
           declined) -> unchanged legacy behaviour, bind immediately.
        3. Known name *with* a secret on file -> don't bind self.person yet;
           hold the name pending and ask for the secret. turn() intercepts
           input until it's confirmed (or gives up after MAX_SECRET_ATTEMPTS
           and falls back to a distinct, unverified identity).
        """
        existing = self.store.find_person_by_name(name)

        if existing is None:
            self.person = self.store.get_or_create_person(name)
            self.auth_state = AuthState.AWAITING_SECRET_CHOICE
            self._log("chloe", "Hi! I am Chloe.")
            reply = (
                f"Hi! I am Chloe. Hello, {name}. If you think you'll talk to me again, "
                f"I can set up a secret word so I know it's really you next time -- want to do that? (yes/no)"
            )
            self._log("chloe", reply)
            return reply

        if existing.has_secret:
            self._pending_name = name
            self._secret_attempts = 0
            self.auth_state = AuthState.AWAITING_SECRET_VERIFY
            self._log("chloe", "Hi! I am Chloe.")
            reply = f"Hi! I am Chloe. Welcome back, {name} -- what's your secret word?"
            self._log("chloe", reply)
            return reply

        self.person = existing
        self._log("chloe", "Hi! I am Chloe.")
        seen_before = any(p.person_id == self.person.id for a in self.store.all_atoms() for p in a.provenance)
        if seen_before:
            return f"Hi! I am Chloe. Nice to talk to you again, {name}."
        return f"Hi! I am Chloe. Hello, {name}."

    # -------------------------------------------------------- secret-word flow
    def _handle_secret_choice(self, text: str) -> str:
        choice = text.strip().lower()
        if choice in _YES:
            self.auth_state = AuthState.AWAITING_SECRET_VALUE
            reply = "Okay -- type your secret word. I'll only ever store it hashed, and won't repeat it back."
        elif choice in _NO:
            self.auth_state = AuthState.NONE
            reply = "No problem -- what would you like to talk about?"
        else:
            # Neither yes nor no. The offer is opt-in and must not hijack the
            # conversation: greet() already bound self.person before making
            # the offer, so nothing is left unresolved here except the offer
            # itself. Drop it and treat this input as an ordinary turn.
            #
            # Note the asymmetry with AWAITING_SECRET_VERIFY, which must NOT
            # fall through: there the claimed identity is still unproven, and
            # falling through would hand a returning person's accumulated
            # trust and belief history to anyone who simply ignored the
            # question.
            self.auth_state = AuthState.NONE
            return self.turn(text)
        self._log("human", text)
        self._log("chloe", reply)
        return reply

    def _handle_secret_value(self, text: str) -> str:
        secret = text.strip()
        if not secret:
            reply = "That came through empty -- what would you like your secret word to be?"
            self._log("human", text)
            self._log("chloe", reply)
            return reply
        self.store.set_person_secret(self.person.id, secret)
        self.auth_state = AuthState.NONE
        reply = f"Got it -- I'll ask for that if someone claiming to be {self.person.name} shows up again."
        self._log("human", "***")  # don't log the plaintext secret, even locally
        self._log("chloe", reply)
        return reply

    def _handle_secret_verify(self, text: str) -> str:
        candidate = text.strip()
        existing = self.store.find_person_by_name(self._pending_name)
        self._log("human", "***")  # don't log secret-word guesses either
        if existing and self.store.verify_person_secret(existing, candidate):
            self.person = existing
            self.auth_state = AuthState.NONE
            self._pending_name = None
            self._secret_attempts = 0
            reply = f"That matches -- good to have you back, {existing.name}."
            self._log("chloe", reply)
            return reply

        self._secret_attempts += 1
        if self._secret_attempts >= self.MAX_SECRET_ATTEMPTS:
            # Don't lock the conversation, but don't silently hand over the
            # real person's trust/belief history either -- fall back to a
            # distinct, clearly-marked identity of their own.
            fallback_name = f"{self._pending_name} (unverified)"
            self.person = self.store.get_or_create_person(fallback_name)
            self.auth_state = AuthState.NONE
            self._pending_name = None
            self._secret_attempts = 0
            reply = (
                f"That still doesn't match what I have on file for {existing.name if existing else 'that name'}. "
                f"I'll treat you as a new, unverified person for now rather than guess."
            )
            self._log("chloe", reply)
            return reply

        reply = "That doesn't match what I have on file -- try again?"
        self._log("chloe", reply)
        return reply

    # ------------------------------------------------------------------ turn
    def turn(self, text: str) -> str:
        """Process one line of human input, return Chloe's reply."""
        if self.auth_state == AuthState.AWAITING_SECRET_CHOICE:
            return self._handle_secret_choice(text)
        if self.auth_state == AuthState.AWAITING_SECRET_VALUE:
            return self._handle_secret_value(text)
        if self.auth_state == AuthState.AWAITING_SECRET_VERIFY:
            return self._handle_secret_verify(text)

        assert self.person is not None, "call greet() first"
        self._log("human", text)
        utt = parse(text)

        if utt.type == UtteranceType.COMMAND:
            reply = self._handle_command(utt)
        elif utt.type == UtteranceType.NEGATION:
            reply = self._handle_negation(utt)
        elif utt.type == UtteranceType.STATEMENT:
            reply = self._handle_statement(utt)
        elif utt.type == UtteranceType.YN_QUESTION:
            reply = self._handle_yn_question(utt)
        elif utt.type == UtteranceType.WH_QUESTION:
            reply = self._handle_wh_question(utt)
        else:
            reply = self._unknown_reply(utt)

        self._log("chloe", reply)
        return reply

    def _unknown_reply(self, utt: Utterance) -> str:
        """The parser refused rather than guessed (llm_nlu.py's contract);
        say so in a way that matches why."""
        reason = utt.extra.get("reason")
        if reason == "multiple_statements":
            return "That sounds like more than one thing at once -- tell me one at a time?"
        if reason == "low_confidence":
            return "I'm not sure I understood that correctly, so I won't store it. Could you say it more plainly?"
        return "I don't understand that yet. Can you rephrase it as '<X> is <Y>'?"

    def pending_question(self) -> Optional[str]:
        """Active conversational verification: if Chloe has an open question
        (from a contradiction or a sleep-generated hypothesis), surface it."""
        q = self.store.next_question()
        if not q:
            return None
        self.store.mark_question_asked(q["id"])
        self._log("chloe", q["question"])
        return q["question"]

    # -------------------------------------------------------------- routing
    def _handle_command(self, utt: Utterance) -> str:
        cmd = utt.extra["command"]
        if cmd == "trust_report":
            people = self.store.all_people()
            if not people:
                return "I don't have any trust profiles yet."
            lines = [f"{p.name}: {p.trust_in():.2f}" for p in people]
            return "Here's what I've got: " + "; ".join(lines)
        if cmd == "knowledge_report":
            atoms = self.store.all_atoms()
            if not atoms:
                return "I don't know anything yet."
            lines = [f"{a.statement()} [{a.status.value}, conf={a.confidence:.2f}]" for a in atoms]
            return "Here's what I know: " + " | ".join(lines)
        if cmd == "open_questions":
            qs = self.store.pending_questions()
            if not qs:
                return "No open questions right now."
            return "Open questions: " + " | ".join(q["question"] for q in qs)
        if cmd == "sleep":
            from . import consolidation
            report = consolidation.sleep(self.store)
            return report.summary()
        if cmd == "exit":
            return "Okay. Bye."
        return "Okay."

    def _handle_statement(self, utt: Utterance) -> str:
        atom = self._find_atom_for_statement(utt.subject, utt.relation, utt.obj, utt.scope)
        if atom is None:
            atom = Atom(id=None, subject=utt.subject, relation=utt.relation, object=utt.obj, scope=utt.scope,
                        status=AtomStatus.CANDIDATE, confidence=self.person.trust_in())
            atom = self.store.save_atom(atom)
            self._attach_provenance(atom, polarity=1, utt=utt)
            return f"Okay, I'll remember that {atom.statement()}."

        if atom.object.strip().lower() == utt.obj.strip().lower():
            # corroboration
            self._attach_provenance(atom, polarity=1, utt=utt)
            trust_mod.update_trust_on_corroboration(self.person, atom.domain)
            self.store.save_person(self.person)
            self._recompute(atom)
            return f"Good, that matches what I already believed: {atom.statement()}."

        # conflicting object, same subject/relation/scope -> contradiction
        self._attach_provenance(atom, polarity=-1, utt=utt)
        trust_mod.update_trust_on_contradiction(self.person, atom.domain)
        self.store.save_person(self.person)
        self._recompute(atom)
        self.store.queue_question(
            f"I've heard conflicting things about whether {atom.subject} {atom.relation} "
            f"{atom.object} or {utt.obj}{(' (' + atom.scope + ')') if atom.scope else ''} -- which is it?",
            reason="contradiction", related_atom_id=atom.id,
        )
        return (f"Hmm, that contradicts what I was told before ({atom.subject} {atom.relation} {atom.object}). "
                f"I'll flag it and ask around.")

    def _handle_negation(self, utt: Utterance) -> str:
        atom = self._find_atom_for_statement(utt.subject, utt.relation, utt.obj, utt.scope)
        if atom is not None and atom.object.strip().lower() == utt.obj.strip().lower():
            # this person denies an existing positive belief
            self._attach_provenance(atom, polarity=-1, utt=utt)
            trust_mod.update_trust_on_contradiction(self.person, atom.domain)
            self.store.save_person(self.person)
            self._recompute(atom)
            self.store.queue_question(
                f"Is it true that {atom.subject} {atom.relation} {atom.object}? I've had that denied.",
                reason="denial", related_atom_id=atom.id,
            )
            return f"Okay -- that conflicts with what I believed. I'll double check: {atom.statement()}."

        # store the negative fact itself, human-readable, as its own atom
        neg_object = f"not {utt.obj}"
        atom = Atom(id=None, subject=utt.subject, relation=utt.relation, object=neg_object, scope=utt.scope,
                    status=AtomStatus.CANDIDATE, confidence=self.person.trust_in())
        atom = self.store.save_atom(atom)
        self._attach_provenance(atom, polarity=1, utt=utt)
        return f"Okay, I'll remember that {atom.statement()}."

    def _handle_yn_question(self, utt: Utterance) -> str:
        atom = self._find_atom_for_statement(utt.subject, utt.relation, utt.obj, utt.scope)
        if atom is None:
            return f"I don't know. What is {utt.subject}?"
        matches = atom.object.strip().lower() == utt.obj.strip().lower()
        if matches:
            return f"Yes, as far as I know ({atom.status.value}, confidence {atom.confidence:.2f})."
        return f"I don't think so -- I believe {atom.statement()} instead (confidence {atom.confidence:.2f})."

    def _handle_wh_question(self, utt: Utterance) -> str:
        candidates = [a for a in self.store.all_atoms() if a.subject.strip().lower() == utt.subject.strip().lower()]
        if not candidates:
            self.store.queue_question(f"What is {utt.subject}?", reason="unknown_term")
            return f"I don't know yet -- what is {utt.subject}?"
        best = max(candidates, key=lambda a: a.confidence)
        return f"{best.subject} {best.relation} {best.object} (confidence {best.confidence:.2f})."

    # ------------------------------------------------------------- internals
    def _find_matching_scope_atom(self, subject: str, relation: str, scope: str) -> Optional[Atom]:
        key = f"{subject.strip().lower()}|{relation.strip().lower()}"
        candidates = self.store.find_atoms_by_key(key)
        for a in candidates:
            if a.scope.strip().lower() == (scope or "").strip().lower():
                return a
        return None

    def _find_atom_for_statement(self, subject: str, relation: str, obj: str, scope: str) -> Optional[Atom]:
        """Find the atom a new 'subject relation object' statement should be
        compared against.

        Same subject+relation+scope can legitimately have more than one
        atom (e.g. 'Felix is a cat' and 'Felix is an animal' are not
        contradictory -- 'is' is not a functional/single-valued relation).
        So: if an atom with the *exact same object* already exists, treat
        this as corroborating THAT atom (this is also how a hypothesis
        generated during sleep gets verified/promoted when someone later
        states it plainly). Otherwise fall back to the atom the store would
        most naturally treat as "the current belief" (highest confidence)
        for contradiction reporting.
        """
        key = f"{subject.strip().lower()}|{relation.strip().lower()}"
        scope_norm = (scope or "").strip().lower()
        candidates = [a for a in self.store.find_atoms_by_key(key) if a.scope.strip().lower() == scope_norm]
        if not candidates:
            return None
        exact = [a for a in candidates if a.object.strip().lower() == obj.strip().lower()]
        if exact:
            return exact[0]
        return max(candidates, key=lambda a: a.confidence)

    def _attach_provenance(self, atom: Atom, polarity: int, utt: Optional[Utterance] = None) -> None:
        interaction = self._log("human", atom.statement(), commit_only=True)
        prov = Provenance(
            person_id=self.person.id,
            interaction_id=interaction.id if interaction else 0,
            polarity=polarity,
            # How sure the parser was of its reading (None for the pattern
            # parser). Recorded, inspectable, and kept out of the
            # belief-confidence maths -- see models.Provenance.
            parse_confidence=utt.extra.get("parse_confidence") if utt else None,
        )
        self.store.add_provenance(atom.id, prov)
        atom.provenance.append(prov)

    def _recompute(self, atom: Atom) -> None:
        people_by_id = {p.id: p for p in self.store.all_people()}
        atom.confidence = trust_mod.recompute_confidence(atom, people_by_id)
        atom.status = trust_mod.status_from_confidence(atom.confidence, len(atom.provenance))
        self.store.save_atom(atom)

    def _log(self, role: str, text: str, commit_only: bool = False) -> Interaction:
        interaction = Interaction(id=None, person_id=self.person.id if self.person else 0, role=role, text=text)
        return self.store.log_interaction(interaction)
