"""
The conversational loop.

The core cycle, as far as a single-session prototype reasonably goes:

    1. Converse.                           -> turn()
    2. Acquire candidate knowledge.        -> _handle_statement / _handle_negation
    3. Consolidate during "sleep".         -> consolidation.sleep()
    4. Generate hypotheses.                -> consolidation.sleep()
    5. Verify through later conversation.  -> pending_question / _handle_yn_question
    6. Revise beliefs and source trust.    -> trust.py and this module

Every exchange is logged as an Interaction, and every stored belief carries
a Provenance record back to the interaction and person that produced it, so
"who said this, and when" always has an answer.
"""

from enum import Enum
from typing import Optional

from . import grammar, trust as trust_mod
from .models import Atom, AtomStatus, Interaction, Person, Provenance, Stance
from .llm_nlu import parse  # LLM-first routing parser, same interface as nlu.parse
from .nlu import Utterance, UtteranceType
from .storage import KnowledgeStore

_YES = {"yes", "yeah", "yep", "sure", "ok", "okay", "y"}
_NO = {"no", "nope", "nah", "n"}

# Interaction roles. ROLE_EVIDENCE marks the row a Provenance record points
# at: the canonical form of a belief at the moment it was admitted, not a
# turn anybody spoke. Transcripts show HUMAN and CHLOE only.
ROLE_HUMAN = "human"
ROLE_CHLOE = "chloe"
ROLE_EVIDENCE = "evidence"
CONVERSATION_ROLES = (ROLE_HUMAN, ROLE_CHLOE)


class AuthState(str, Enum):
    """Where an in-progress identity check stands. NONE is normal
    conversation; the others are multi-turn detours that greet()/turn()
    walk through before self.person is bound."""

    NONE = "none"
    AWAITING_SECRET_CHOICE = "awaiting_secret_choice"   # new person: "want to set a secret word?"
    AWAITING_SECRET_VALUE = "awaiting_secret_value"     # new person, said yes: waiting for the word itself
    AWAITING_SECRET_VERIFY = "awaiting_secret_verify"   # returning name with a secret on file: waiting for it


class QuestionState(str, Enum):
    """Where the question detour stands. Separate from AuthState: identity
    must be settled before CHLOE asks anybody anything, so the two never
    overlap."""

    NONE = "none"
    AWAITING_CONSENT = "awaiting_consent"   # "can I ask you some questions?"
    AWAITING_ANSWER = "awaiting_answer"     # a question is on the table


# Why CHLOE is unsure, in its own words. Taken from the `reason` recorded
# when the question was queued, so the explanation is the real one rather
# than a stock line.
QUESTION_PREAMBLE = {
    "hypothesis": "This one is my own inference, not something anybody told me",
    "llm_hypothesis": "This one I put together myself, out of more than one thing I was told",
    "unresolved_contradiction": "I still can't settle this one",
    "contradiction": "My sources disagree here",
    "denial": "I believed this, and then someone denied it",
    "weak_candidate_recheck": "This has only ever come from one person",
    "unknown_term": "This one is a blank for me",
}
DEFAULT_PREAMBLE = "I'm not sure I have this right"

# Which questions a plain yes or no can actually settle. The others ask
# "which is it?" and want a statement, so a yes/no vote on them would be
# recording an answer to a question that was never put.
YES_NO_REASONS = {"hypothesis", "llm_hypothesis", "weak_candidate_recheck", "denial"}

# Ending the detour. "no" is deliberately absent: it is the commonest
# *answer*, and treating it as a refusal would silently discard evidence.
_STOP = {"stop", "not now", "later", "no thanks", "enough", "that's enough"}


class ChloeEngine:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.person: Optional[Person] = None
        self.auth_state: AuthState = AuthState.NONE
        self._pending_name: Optional[str] = None
        self._secret_attempts = 0
        self.MAX_SECRET_ATTEMPTS = 3
        self.question_state: QuestionState = QuestionState.NONE
        self._current_question: Optional[dict] = None
        self._questions_declined = False   # honoured for the rest of the session
        self._resume_questions = False     # carry on after a statement-shaped answer
        self._pending_answer: Optional[dict] = None   # question this turn may yet answer in its own words
        self.last_stance: Optional[Stance] = None     # stance of the last reply, for the output-side guard

    # ------------------------------------------------------------- greeting
    def greet(self, name: str) -> str:
        """Look the name up rather than binding to it: a name alone is not
        proof of identity. Three cases:

        1. New name -> greet, then offer to set up a secret word for next
           time (opt-in).
        2. Known name, no secret on file -> bind immediately.
        3. Known name with a secret on file -> hold the name pending and
           ask for the secret; turn() intercepts input until it is
           confirmed, or falls back to a distinct unverified identity after
           MAX_SECRET_ATTEMPTS.
        """
        existing = self.store.find_person_by_name(name)

        if existing is None:
            self.person = self.store.get_or_create_person(name)
            self.auth_state = AuthState.AWAITING_SECRET_CHOICE
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
            reply = f"Hi! I am Chloe. Welcome back, {name} -- what's your secret word?"
            self._log("chloe", reply)
            return reply

        self.person = existing
        seen_before = any(p.person_id == self.person.id for a in self.store.all_atoms() for p in a.provenance)
        reply = (f"Hi! I am Chloe. Nice to talk to you again, {name}." if seen_before
                 else f"Hi! I am Chloe. Hello, {name}.")
        self._log("chloe", reply)
        return reply

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
            # The offer is opt-in and must not hijack the conversation:
            # greet() already bound self.person, so nothing is unresolved
            # but the offer itself. Drop it and take this as a normal turn.
            # AWAITING_SECRET_VERIFY must NOT fall through this way -- there
            # the claimed identity is still unproven.
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
            # Neither lock the conversation nor hand over the real person's
            # trust and belief history: give them a distinct, clearly
            # marked identity instead.
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
        self.last_stance = None   # this turn's stance, set by whatever answers
        if self.auth_state == AuthState.AWAITING_SECRET_CHOICE:
            return self._handle_secret_choice(text)
        if self.auth_state == AuthState.AWAITING_SECRET_VALUE:
            return self._handle_secret_value(text)
        if self.auth_state == AuthState.AWAITING_SECRET_VERIFY:
            return self._handle_secret_verify(text)

        if self.question_state == QuestionState.AWAITING_CONSENT:
            handled = self._handle_question_consent(text)
            if handled is not None:
                return handled
        elif self.question_state == QuestionState.AWAITING_ANSWER:
            handled = self._handle_question_answer(text)
            if handled is not None:
                return handled

        assert self.person is not None, "call greet() first"
        self._log("human", text)
        utt = grammar.resolve_referents(parse(text), self.person.name)

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
        elif utt.type == UtteranceType.SMALL_TALK:
            reply = self._small_talk_reply(utt)
        else:
            reply = self._unknown_reply(utt)

        if self._pending_answer is not None:
            # A question was on the table and the reply was neither yes nor
            # no. Now that it has been parsed, it can be judged: if it
            # speaks to the belief the question hangs on, it was an answer
            # in the person's own words. The ordinary path above has already
            # recorded what they said -- all that is left is to close the
            # question and carry on down the queue. Anything else is a
            # change of subject, and she stops asking.
            q, self._pending_answer = self._pending_answer, None
            if self._statement_answers(utt, q):
                self.store.mark_question_answered(q["id"])
                self._resume_questions = True
            else:
                self._questions_declined = True

        if self._resume_questions:
            self._resume_questions = False
            nxt = self._ask_next_question()
            if nxt:
                reply = f"{reply}\n{nxt}"

        self._log("chloe", reply)
        return reply

    def _small_talk_reply(self, utt: Utterance) -> str:
        """A pleasantry is not evidence, and not a failure to understand
        either. She answers it and writes nothing: the language layer may
        converse freely precisely because nothing it says reaches the store.

        The 2000 version picked from a short list for its own stock replies
        (cMisc.hh); with a language model configured, this text is what gets
        phrased naturally on the way out.
        """
        openers = ("hello", "hi", "hey", "greetings", "good morning",
                   "good afternoon", "good evening", "good day")
        thanks = ("thank", "cheers")
        farewell = ("bye", "goodbye", "good night", "see you")
        low = utt.raw.strip().lower()

        if low.startswith(farewell):
            return "Goodbye. I'll keep what you told me."
        if low.startswith(thanks):
            return "You're welcome."
        if low.startswith(openers):
            name = self.person.name if self.person else "there"
            return f"Hello, {name}. Tell me something, or ask me what I know."
        if "how are" in low:
            known = len(self.store.all_atoms())
            if known:
                plural = "" if known == 1 else "s"
                return (f"I'm well, thank you -- {known} thing{plural} on file at the moment, "
                        f"some better attested than others. How are you?")
            return "I'm well, thank you, though I don't know much yet. How are you?"
        return "Noted. Tell me something, or ask me what I know."

    def _unknown_reply(self, utt: Utterance) -> str:
        """The parser refused rather than guessed (llm_nlu.py's contract).
        Say so in a way that matches why."""
        reason = utt.extra.get("reason")
        if reason == "multiple_statements":
            return "That sounds like more than one thing at once -- tell me one at a time?"
        if reason == "low_confidence":
            return "I'm not sure I understood that correctly, so I won't store it. Could you say it more plainly?"
        return "I don't understand that yet. Can you rephrase it as '<X> is <Y>'?"

    # -------------------------------------------------------- sleep / wake
    def sleep(self) -> str:
        """Consolidate, then wake with whatever that turned up.

        The 2000 version treated "Sleep" as the end of the session: it
        persisted state, said goodbye and exited, and chloe.sh brought it
        back. Here the pass is the same idea without the process restart,
        and waking is where the questions consolidation produced get put to
        somebody -- which is the point of having slept.
        """
        from . import consolidation
        report = consolidation.sleep(self.store)
        self._questions_declined = False   # a fresh pass earns a fresh ask
        self.question_state = QuestionState.NONE
        self._current_question = None
        summary = report.summary()
        self._log("chloe", summary)
        offer = self.offer_questions()
        return f"{summary}\n{offer}" if offer else summary

    # ----------------------------------------------------- question detour
    def offer_questions(self) -> Optional[str]:
        """Ask permission to put CHLOE's open questions to this person.

        Returns the offer, or None if there is nothing to ask or the person
        has already said stop this session. Consent is asked for once and
        then honoured: the point of the detour is that it is a guest in the
        conversation, not that it gets to interrogate.
        """
        if self._questions_declined or self.question_state != QuestionState.NONE:
            return None
        if self.store.next_question() is None:
            return None
        self.question_state = QuestionState.AWAITING_CONSENT
        reply = ("While I was asleep I turned over a few things I'm not certain about. "
                 "Can I ask you some questions? If yes, you then say stop whenever you want me to stop.")
        self._log("chloe", reply)
        return reply

    def _handle_question_consent(self, text: str) -> Optional[str]:
        """Yes starts the questions; stop ends them for the session. Anything
        else is read as 'not now' -- the detour gets out of the way and the
        input is processed as an ordinary turn, so an unanswered offer never
        swallows what the person actually wanted to say."""
        choice = text.strip().lower().rstrip(".!?")
        if choice in _YES:
            self.question_state = QuestionState.NONE
            self._log("human", text)
            return self._ask_next_question() or "Actually, nothing outstanding after all."
        self.question_state = QuestionState.NONE
        if choice in _STOP or choice in _NO:
            self._questions_declined = True
            self._log("human", text)
            reply = "Of course -- I'll keep them to myself."
            self._log("chloe", reply)
            return reply
        # neither: treat it as a change of subject and don't ask again
        self._questions_declined = True
        return None

    def _ask_next_question(self) -> Optional[str]:
        """Put the next open question, prefaced with why it is unsure."""
        q = self.store.next_question()
        if q is None:
            self.question_state = QuestionState.NONE
            self._current_question = None
            return None
        self.store.mark_question_asked(q["id"])
        self._current_question = q
        self.question_state = QuestionState.AWAITING_ANSWER
        preamble = QUESTION_PREAMBLE.get(q["reason"], DEFAULT_PREAMBLE)
        reply = f"{preamble}. {q['question']}"
        self._log("chloe", reply)
        return reply

    def _handle_question_answer(self, text: str) -> Optional[str]:
        """A yes or no settles the belief the question hangs on. Anything
        else ends the detour and is processed as an ordinary turn -- the
        question stays open rather than being recorded as answered by
        something that wasn't an answer."""
        answer = text.strip().lower().rstrip(".!?")
        q = self._current_question or {}

        if answer in _STOP:
            self._end_questions()
            self._questions_declined = True
            self._log("human", text)
            reply = "Okay, I'll stop there. Thanks for the ones you did answer."
            self._log("chloe", reply)
            return reply

        if q.get("reason") in YES_NO_REASONS and (answer in _YES or answer in _NO):
            self._log("human", text)
            resolved = self._settle_question(q, confirmed=answer in _YES)
            self.store.mark_question_answered(q["id"])
            self._end_questions()
            return self._then_next(resolved)

        if q.get("reason") not in YES_NO_REASONS:
            # An open question ("which is it?") is answered by a statement.
            # Close it and let the ordinary path store what they said, which
            # is what settles the belief -- no special-case vote needed --
            # then pick the detour back up after that turn is processed.
            if q.get("id"):
                self.store.mark_question_answered(q["id"])
            self._end_questions()
            self._resume_questions = True
            return None

        # A yes/no question answered with neither. It may still be an
        # answer -- "Claire is an artist, indeed" settles the question as
        # surely as "yes" -- but that cannot be told from the raw text, and
        # parsing it here would parse it twice. Step aside, let the ordinary
        # path handle the line, and judge it in turn() once it is parsed.
        self._end_questions()
        self._pending_answer = q or None
        return None

    def _statement_answers(self, utt: Utterance, q: dict) -> bool:
        """Whether this utterance answers the question that was on the table.

        The test is the subject and relation of the atom the question hangs
        on. Agreeing and disagreeing are both answers -- which it was has
        already been recorded by the ordinary path -- so only the topic is
        checked. A question with no atom behind it (an unknown term) has
        nothing to match against and is left open.
        """
        from .consolidation import _norm_relation
        if utt is None or utt.type not in (UtteranceType.STATEMENT, UtteranceType.NEGATION):
            return False
        atom_id = q.get("related_atom_id")
        if not atom_id:
            return False
        atoms = [a for a in self.store.all_atoms() if a.id == atom_id]
        if not atoms:
            return False
        atom = atoms[0]
        return (grammar.same_referent(utt.subject or "", atom.subject)
                and _norm_relation(utt.relation or "") == _norm_relation(atom.relation))

    def _end_questions(self) -> None:
        self.question_state = QuestionState.NONE
        self._current_question = None

    def _then_next(self, said: str) -> str:
        """Chain the acknowledgement onto the next question, if there is one."""
        nxt = self._ask_next_question()
        return f"{said}\n{nxt}" if nxt else f"{said} That's everything I had -- thank you."

    def _settle_question(self, q: dict, confirmed: bool) -> str:
        """Route a yes/no onto the belief the question was queued about.

        A question with no related atom (an unknown term, say) has nothing to
        vote on, so the answer is acknowledged and nothing is written -- an
        unattached vote would be provenance pointing at nothing.
        """
        atom_id = q.get("related_atom_id")
        if not atom_id:
            return "Noted, thank you." if confirmed else "Understood."
        atoms = [a for a in self.store.all_atoms() if a.id == atom_id]
        if not atoms:
            return "Thanks -- though I seem to have lost the belief that went with that."
        atom = atoms[0]
        self._attach_provenance(atom, polarity=1 if confirmed else -1)
        if confirmed:
            trust_mod.update_trust_on_corroboration(self.person, atom.domain)
        else:
            trust_mod.update_trust_on_contradiction(self.person, atom.domain)
        self.store.save_person(self.person)
        self._recompute(atom)
        lead = "that supports" if confirmed else "I've noted you disputing"
        return (f"Thanks -- {lead} {self._say_atom(atom)} "
                f"(now {atom.status.value}, confidence {atom.confidence:.2f}).")

    def pending_question(self) -> Optional[str]:
        """Surface an open question, if there is one -- from a
        contradiction, or from a hypothesis generated during sleep."""
        q = self.store.next_question()
        if not q:
            return None
        self.store.mark_question_asked(q["id"])
        self._log("chloe", q["question"])
        return q["question"]

    # ------------------------------------------------------------- phrasing
    def _say(self, subject: str, relation: str, obj: str, scope: str = "") -> str:
        """A stored triple phrased for whoever is speaking: their own name
        comes back as "you", Chloe's as "I", with the copula agreed."""
        return grammar.clause(subject, relation, obj, scope,
                              self.person.name if self.person else "")

    def _say_atom(self, atom: Atom) -> str:
        return self._say(atom.subject, atom.relation, atom.object, atom.scope)

    def _ask_wh(self, subject: str, wh: str = "what") -> str:
        return grammar.wh_clause(subject, self.person.name if self.person else "", wh)

    def _is_speaker(self, subject: str) -> bool:
        return bool(self.person) and grammar.same_referent(subject, self.person.name)

    def _vacuous_reply(self, utt: Utterance) -> str:
        """Subject and object turned out to denote the same thing, so the
        statement carries no information and is not stored. Pronoun
        resolution makes this common: "I am Dan", said by Dan, resolves to
        "Dan is Dan"."""
        if self._is_speaker(utt.subject) and grammar.is_pronoun(utt.raw.strip().split()[0]):
            return f"Yes -- I have you as {self.person.name}. Tell me something about yourself?"
        return f"Everything is itself, so that tells me nothing about {utt.subject}."

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
        if cmd == "stop":
            self._questions_declined = True
            self.question_state = QuestionState.NONE
            self._current_question = None
            return "Okay, I'll stop asking."
        if cmd == "sleep":
            return self.sleep()
        if cmd == "exit":
            return "Okay. Bye."
        return "Okay."

    def _handle_statement(self, utt: Utterance) -> str:
        if grammar.same_referent(utt.subject, utt.obj):
            return self._vacuous_reply(utt)
        atom = self._find_atom_for_statement(utt.subject, utt.relation, utt.obj, utt.scope)
        if atom is None:
            atom = Atom(id=None, subject=utt.subject, relation=utt.relation, object=utt.obj, scope=utt.scope,
                        status=AtomStatus.CANDIDATE, confidence=0.5)
            atom = self.store.save_atom(atom)
            self._attach_provenance(atom, polarity=1, utt=utt)
            self._recompute(atom)
            return f"Okay, I'll remember that {self._say_atom(atom)}."

        if atom.object.strip().lower() == utt.obj.strip().lower():
            # corroboration
            self._attach_provenance(atom, polarity=1, utt=utt)
            trust_mod.update_trust_on_corroboration(self.person, atom.domain)
            self.store.save_person(self.person)
            self._recompute(atom)
            return f"Good, that matches what I already believed: {self._say_atom(atom)}."

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
        return (f"Hmm, that contradicts what I was told before "
                f"({self._say(atom.subject, atom.relation, atom.object)}). "
                f"I'll flag it and ask around.")

    def _handle_negation(self, utt: Utterance) -> str:
        if grammar.same_referent(utt.subject, utt.obj):
            return f"Nothing can fail to be itself, so I won't record that about {utt.subject}."
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
            return f"Okay -- that conflicts with what I believed. I'll double check: {self._say_atom(atom)}."

        # store the negative fact itself, human-readable, as its own atom
        neg_object = f"not {utt.obj}"
        atom = Atom(id=None, subject=utt.subject, relation=utt.relation, object=neg_object, scope=utt.scope,
                    status=AtomStatus.CANDIDATE, confidence=0.5)
        atom = self.store.save_atom(atom)
        self._attach_provenance(atom, polarity=1, utt=utt)
        self._recompute(atom)
        return f"Okay, I'll remember that {self._say_atom(atom)}."

    def _handle_yn_question(self, utt: Utterance) -> str:
        """Answer from the store alone, and record which way the answer went.

        The stance is what the output-side guard needs: the reply's wording
        is the language layer's business, but whether it comes back as a yes,
        a no, or neither is the core's, and a naturalisation that turns one
        into another has changed the outcome rather than phrased it.
        """
        if grammar.same_referent(utt.subject, utt.obj):
            self.last_stance = Stance.AFFIRM
            return "Yes -- necessarily."
        atom = self._find_atom_for_statement(utt.subject, utt.relation, utt.obj, utt.scope)
        if atom is None:
            self.last_stance = Stance.UNKNOWN
            wh = utt.extra.get("wh", "what")
            return f"I don't know. {grammar.capitalise(self._ask_wh(utt.subject, wh))}?"
        matches = atom.object.strip().lower() == utt.obj.strip().lower()
        if matches:
            self.last_stance = Stance.AFFIRM
            return f"Yes, as far as I know ({atom.status.value}, confidence {atom.confidence:.2f})."
        # Not a denial of the question so much as a different value on file:
        # nothing here knows that a dog is also an animal, and the reply says
        # only what the store holds.
        self.last_stance = Stance.DENY
        return (f"I don't think so -- I believe {self._say_atom(atom)} instead "
                f"(confidence {atom.confidence:.2f}).")

    def _handle_wh_question(self, utt: Utterance) -> str:
        wh = utt.extra.get("wh", "what")
        candidates = [a for a in self.store.all_atoms() if a.subject.strip().lower() == utt.subject.strip().lower()]
        if not candidates:
            # The queued question keeps the resolved name: it may be asked
            # of someone else later, for whom "you" would mean someone else.
            self.store.queue_question(f"What is {utt.subject}?", reason="unknown_term")
            # A name is not a belief and is held on the Person, not as an
            # atom -- but answering a flat "I don't know" immediately after
            # greeting someone by that name reads as a contradiction.
            if self._is_speaker(utt.subject):
                known = f"Only that you're {self.person.name}"
            elif grammar.same_referent(utt.subject, grammar.CHLOE_NAME):
                known = f"Only that I'm {grammar.CHLOE_NAME}"
            else:
                known = "I don't know yet"
            return f"{known} -- {self._ask_wh(utt.subject, wh)}?"
        best = max(candidates, key=lambda a: a.confidence)
        clause = self._say(best.subject, best.relation, best.object)
        return f"{grammar.capitalise(clause)} (confidence {best.confidence:.2f})."

    # ------------------------------------------------------------- internals
    def _find_matching_scope_atom(self, subject: str, relation: str, scope: str) -> Optional[Atom]:
        key = f"{subject.strip().lower()}|{relation.strip().lower()}"
        candidates = self.store.find_atoms_by_key(key)
        for a in candidates:
            if a.scope.strip().lower() == (scope or "").strip().lower():
                return a
        return None

    def _find_atom_for_statement(self, subject: str, relation: str, obj: str, scope: str) -> Optional[Atom]:
        """Find the atom a new 'subject relation object' statement should
        be compared against.

        An atom with the exact same object is corroborated -- this is also
        how a sleep-generated hypothesis gets promoted when someone later
        states it plainly. Failing that, the highest-confidence atom on the
        same subject, relation and scope is taken as the current belief and
        reported as contradicted.

        That fallback is the prototype's known limit on multi-valued
        relations. One subject+relation+scope can legitimately carry several
        true objects -- 'Felix is a cat' and 'Felix is an animal' do not
        conflict -- but nothing here distinguishes a rival value from a
        compatible one, so the second is reported as a contradiction. A
        denial ('Felix is not a cat') is the unambiguous case: it attaches
        opposing provenance to the atom itself and needs no such judgement.
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
        interaction = self._log(ROLE_EVIDENCE, atom.statement(), commit_only=True)
        prov = Provenance(
            person_id=self.person.id,
            interaction_id=interaction.id if interaction else 0,
            polarity=polarity,
            # Recorded and inspectable, but kept out of the
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
