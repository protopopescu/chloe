"""
The conversational loop.

The core cycle, as far as a single-session prototype reasonably goes:

    1. Converse.                           -> turn()
    2. Acquire candidate knowledge.        -> _record()
    3. Consolidate during "sleep".         -> consolidation.sleep()
    4. Generate hypotheses.                -> consolidation.sleep()
    5. Verify through later conversation.  -> the question detour, questions.py
    6. Revise beliefs and source trust.    -> trust.py and _record()

Every exchange is logged as an Interaction, and every stored belief carries
a Provenance record back to the interaction and person that produced it, so
"who said this, and when" always has an answer.
"""

from enum import Enum
from typing import Callable, Optional

from . import entailment, grammar, questions, trust as trust_mod
from .models import Atom, AtomStatus, Interaction, Person, Provenance, Stance
from .llm_nlu import parse  # LLM-first routing parser, same interface as nlu.parse
from .nlu import Utterance, UtteranceType
from .storage import KnowledgeStore

# Words that only mean something as a reply to what is on the table. They
# are interpreted against it and never handed to the parser, which given a
# bare "no" and nothing to attach it to has nothing true to return.
_REPLY_WORDS = {
    "yes": {"yes", "yeah", "yep", "sure", "ok", "okay", "y"},
    "no": {"no", "nope", "nah", "n"},
    "stop": {"stop", "not now", "later", "no thanks", "enough", "that's enough"},
}


# How many beliefs a wh-answer names before the remainder is counted.
WH_ANSWER_LIMIT = 3

# How many entries the two reports list.
KNOWLEDGE_REPORT_LIMIT = 5
TRUST_REPORT_LIMIT = 5


def _join(items: list) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _reply_word(text: str) -> Optional[str]:
    """'yes', 'no' or 'stop' if that is all the line says, else None."""
    low = text.strip().lower().rstrip(".!?").strip()
    for word, forms in _REPLY_WORDS.items():
        if low in forms:
            return word
    return None


# Interaction roles. ROLE_EVIDENCE marks the row a Provenance record points
# at: the canonical form of a belief at the moment it was admitted, not a
# turn anybody spoke. Transcripts show SOURCE and CHLOE only.
ROLE_SOURCE = "source"
ROLE_CHLOE = "chloe"
ROLE_EVIDENCE = "evidence"
CONVERSATION_ROLES = (ROLE_SOURCE, ROLE_CHLOE)


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


class ChloeEngine:
    """One conversation with one person.

    turn() is where a line of input becomes a reply, and the only place a
    turn is logged: one row for what the person said and one for what CHLOE
    answered, whatever path produced it. Handlers return text and never log.
    The other public entry points -- greet(), wake(), pending_question() --
    speak unprompted, and log what they say themselves.

    `nap`, if given, replaces the synchronous sleep() for the "sleep"
    command: a deployment that sleeps on its own schedule (the web server's
    dream window) passes a callable that starts it and returns what to say.
    """

    def __init__(self, store: KnowledgeStore, nap: Optional[Callable[[], str]] = None):
        self.store = store
        self.nap = nap
        self.person: Optional[Person] = None
        self.auth_state: AuthState = AuthState.NONE
        self._pending_name: Optional[str] = None
        self._secret_attempts = 0
        self.MAX_SECRET_ATTEMPTS = 3
        self.question_state: QuestionState = QuestionState.NONE
        self._current_question: Optional[dict] = None
        self._questions_declined = False   # honoured for the rest of the session
        self.last_stance: Optional[Stance] = None     # stance of the last reply, for the output-side guard
        # The question this turn put to the person, if any. It closes the
        # reply, and the output interface passes it through unphrased: it is
        # what the person's next answer will be recorded against.
        self.last_question: Optional[str] = None

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
            self._log(ROLE_CHLOE, reply)
            return reply

        if existing.has_secret:
            self._pending_name = name
            self._secret_attempts = 0
            self.auth_state = AuthState.AWAITING_SECRET_VERIFY
            reply = f"Hi! I am Chloe. Welcome back, {name} -- what's your secret word?"
            self._log(ROLE_CHLOE, reply)
            return reply

        self.person = existing
        seen_before = any(p.person_id == self.person.id for a in self.store.all_atoms() for p in a.provenance)
        reply = (f"Hi! I am Chloe. Nice to talk to you again, {name}." if seen_before
                 else f"Hi! I am Chloe. Hello, {name}.")
        self._log(ROLE_CHLOE, reply)
        return reply

    # -------------------------------------------------------- secret-word flow
    def _handle_secret_choice(self, text: str) -> str:
        choice = _reply_word(text)
        if choice == "yes":
            self.auth_state = AuthState.AWAITING_SECRET_VALUE
            reply = "Okay -- type your secret word. I'll only ever store it hashed, and won't repeat it back."
        elif choice == "no":
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
        self._log(ROLE_SOURCE, text)
        self._log(ROLE_CHLOE, reply)
        return reply

    def _handle_secret_value(self, text: str) -> str:
        secret = text.strip()
        if not secret:
            reply = "That came through empty -- what would you like your secret word to be?"
            self._log(ROLE_SOURCE, text)
            self._log(ROLE_CHLOE, reply)
            return reply
        self.store.set_person_secret(self.person.id, secret)
        self.auth_state = AuthState.NONE
        reply = f"Got it -- I'll ask for that if someone claiming to be {self.person.name} shows up again."
        self._log(ROLE_SOURCE, "***")  # don't log the plaintext secret, even locally
        self._log(ROLE_CHLOE, reply)
        return reply

    def _handle_secret_verify(self, text: str) -> str:
        candidate = text.strip()
        existing = self.store.find_person_by_name(self._pending_name)
        self._log(ROLE_SOURCE, "***")  # don't log secret-word guesses either
        if existing and self.store.verify_person_secret(existing, candidate):
            self.person = existing
            self.auth_state = AuthState.NONE
            self._pending_name = None
            self._secret_attempts = 0
            reply = f"That matches -- good to have you back, {existing.name}."
            self._log(ROLE_CHLOE, reply)
            return reply

        self._secret_attempts += 1
        if self._secret_attempts >= self.MAX_SECRET_ATTEMPTS:
            # Neither lock the conversation nor hand over the real person's
            # trust and belief history: give them a distinct, clearly
            # marked identity instead.
            fallback_name = f"{self._pending_name}{Person.UNVERIFIED_SUFFIX}"
            self.person = self.store.get_or_create_person(fallback_name)
            self.auth_state = AuthState.NONE
            self._pending_name = None
            self._secret_attempts = 0
            reply = (
                f"That still doesn't match what I have on file for {existing.name if existing else 'that name'}. "
                f"I'll treat you as a new, unverified person for now rather than guess."
            )
            self._log(ROLE_CHLOE, reply)
            return reply

        reply = "That doesn't match what I have on file -- try again?"
        self._log(ROLE_CHLOE, reply)
        return reply

    # ------------------------------------------------------------------ turn
    def turn(self, text: str) -> str:
        """Process one line of human input, return Chloe's reply."""
        self.last_stance = None   # this turn's stance, set by whatever answers
        self.last_question = None
        if self.auth_state == AuthState.AWAITING_SECRET_CHOICE:
            return self._handle_secret_choice(text)
        if self.auth_state == AuthState.AWAITING_SECRET_VALUE:
            return self._handle_secret_value(text)
        if self.auth_state == AuthState.AWAITING_SECRET_VERIFY:
            return self._handle_secret_verify(text)

        assert self.person is not None, "call greet() first"
        # Other sessions, and sleep, change people's trust in the store; the
        # turn works from the store's copy, not one read at greeting.
        self.person = self.store.person_by_id(self.person.id) or self.person
        self._log(ROLE_SOURCE, text)
        reply = self._respond(text)
        self._log(ROLE_CHLOE, reply)
        return reply

    def _respond(self, text: str) -> str:
        """What is on the table decides how a reply word is read; anything
        else is parsed once and handled on its merits, and then judged
        against the question it may have been answering."""
        word = _reply_word(text)
        state, q = self.question_state, self._current_question
        self._end_questions()

        if state == QuestionState.AWAITING_CONSENT:
            if word == "yes":
                return self._ask_next() or "Actually, nothing outstanding after all."
            if word in ("no", "stop"):
                self._questions_declined = True
                return "Of course -- I'll keep them to myself."
            # Anything else: the offer steps aside, and the line is taken on
            # its merits rather than swallowed by an offer it ignored.
            self._questions_declined = True

        elif state == QuestionState.AWAITING_ANSWER:
            if word == "stop":
                self._questions_declined = True
                return "Okay, I'll stop there. Thanks for the ones you did answer."
            if word in ("yes", "no"):
                if questions.kind_of(q) == questions.Kind.CONFIRM:
                    return self._answer_confirm(q, confirmed=(word == "yes"))
                if questions.kind_of(q) == questions.Kind.IMPLY:
                    return self._answer_implication(q, holds=(word == "yes"))
                # "Which is it?" and "what is it?" are not settled by a yes
                # or a no. The question stays on the table.
                return f"I need more than a yes or no for that one. {self._put(q)}"

        elif word in ("yes", "no"):
            # A yes or no with nothing on the table answers nothing, and is
            # an acknowledgement rather than testimony.
            return self._small_talk_reply(Utterance(raw=text, type=UtteranceType.SMALL_TALK))

        utt = grammar.resolve_referents(parse(text), self.person.name)
        reply = self._dispatch(utt)

        if state == QuestionState.AWAITING_ANSWER and self.question_state == QuestionState.NONE:
            # The line was not a reply word, and did not itself put anything
            # new on the table ("ask more", "sleep"). If it speaks to the question --
            # a statement or denial about the same thing -- it was an answer
            # in the person's own words, already recorded above; the question
            # closes and the next one follows. Anything else is a change of
            # subject, and she stops asking.
            if questions.answered_by(self.store, q, utt):
                self.store.mark_question_answered(q["id"])
                reply = self._and_next(reply)
            else:
                self._questions_declined = True
        return reply

    def _dispatch(self, utt: Utterance) -> str:
        if utt.type == UtteranceType.COMMAND:
            return self._handle_command(utt)
        if utt.type == UtteranceType.NEGATION:
            return self._record(utt.subject, utt.relation, utt.obj, utt.scope, -1, utt)
        if utt.type == UtteranceType.STATEMENT:
            return self._record(utt.subject, utt.relation, utt.obj, utt.scope, +1, utt)
        if utt.type == UtteranceType.YN_QUESTION:
            return self._handle_yn_question(utt)
        if utt.type == UtteranceType.WH_QUESTION:
            return self._handle_wh_question(utt)
        if utt.type == UtteranceType.SMALL_TALK:
            return self._small_talk_reply(utt)
        return self._unknown_reply(utt)

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
        if reason == "too_long":
            return ("That's longer than I can read, so I'll ignore it. Can we start "
                    "with something simpler, like '<X> is <Y>'?")
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
        summary = consolidation.sleep(self.store).summary()
        offer = self._offer()
        return f"{summary}\n{offer}" if offer else summary

    def wake(self) -> Optional[str]:
        """Waking from a sleep this person asked for, when the sleeping was
        done elsewhere (the web server's dream window). Returns the offer of
        questions, logged, or None."""
        offer = self._offer()
        if offer:
            self._log(ROLE_CHLOE, offer)
        return offer

    # ----------------------------------------------------- question detour
    def _offer(self) -> Optional[str]:
        """Ask permission to put CHLOE's open questions to this person, if
        there is anything worth asking them. A fresh sleep earns a fresh
        ask, so an earlier 'stop' is not held against this offer; after it,
        consent is honoured for the rest of the session. The detour is a
        guest in the conversation, not an interrogation."""
        self._questions_declined = False
        self._end_questions()
        if questions.next_for(self.store, self.person) is None:
            return None
        self.question_state = QuestionState.AWAITING_CONSENT
        return ("While I was asleep I turned over a few things I'm not certain about. "
                "Can I ask you some questions? If yes, you then say stop whenever you want me to stop.")

    def offer_questions(self) -> Optional[str]:
        """The offer outside a sleep: None if this person has said stop."""
        if self._questions_declined or self.question_state != QuestionState.NONE:
            return None
        return self.wake()

    def _ask_next(self) -> Optional[str]:
        """Put the next question worth asking this person, if any."""
        q = questions.next_for(self.store, self.person)
        if q is None:
            return None
        self.store.mark_question_asked(q["id"])
        return self._put(q)

    def _put(self, q: dict) -> str:
        """Put a question on the table; returns it as said to this person.
        Whatever reply carries it must end with it (see last_question)."""
        self._current_question = q
        self.question_state = QuestionState.AWAITING_ANSWER
        self.last_question = self._render(q)
        return self.last_question

    def _render(self, q: dict) -> str:
        return questions.render(self.store, q, self.person)

    def _end_questions(self) -> None:
        self.question_state = QuestionState.NONE
        self._current_question = None

    def _answer_confirm(self, q: dict, confirmed: bool) -> str:
        """A yes or a no to a question about one belief is a claim about
        that belief like any other, and is recorded by the same path."""
        atom = self.store.atom_by_id(q.get("related_atom_id"))
        self.store.mark_question_answered(q["id"])
        if atom is None:
            said = "Thanks -- though I seem to have lost the belief that went with that."
        else:
            said = self._record(atom.subject, atom.relation, atom.object, atom.scope,
                                +1 if confirmed else -1)
        return self._and_next(said)

    def _answer_implication(self, q: dict, holds: bool) -> str:
        """Whether one belief follows from another is a question about what
        the words mean, not evidence about the world: it links the two, or
        keeps them apart, and records who said so."""
        specific = self.store.atom_by_id(q.get("related_atom_id"))
        general = self.store.atom_by_id(q.get("other_atom_id"))
        self.store.mark_question_answered(q["id"])
        if specific is None or general is None:
            said = "Thanks -- though I seem to have lost one of the two beliefs that went with that."
        elif holds:
            entailment.link(self.store, specific, general, confirmed_by=self.person.id)
            said = (f"Thanks -- then I'll take it that {self._say_atom(general)} "
                    f"whenever {self._say_atom(specific)}.")
        else:
            entailment.reject(self.store, specific, general, self.person.id)
            said = "Thanks -- I'll keep the two apart."
        return self._and_next(said)

    def _and_next(self, said: str) -> str:
        """An answer acknowledged, and the next question, if there is one."""
        nxt = self._ask_next()
        return f"{said}\n{nxt}" if nxt else f"{said} That's everything I had -- thank you."

    def pending_question(self) -> Optional[str]:
        """Lead with an open question, without the consent step: for the
        start of a CLI session. The question is put on the table, so the
        reply is taken as its answer."""
        q = questions.next_for(self.store, self.person)
        if q is None:
            return None
        self.store.mark_question_asked(q["id"])
        text = self._put(q)
        self._log(ROLE_CHLOE, text)
        return text

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
            people.sort(key=lambda p: (-p.trust_in(), p.name.lower()))
            shown = people[:TRUST_REPORT_LIMIT]
            lines = [f"{p.name}: {p.trust_in():.2f}" for p in shown]
            opener = ("Here's who I trust most: " if len(people) > len(shown)
                      else "Here's who I trust: ")
            return opener + "; ".join(lines)
        if cmd == "knowledge_report":
            atoms = self.store.all_atoms()
            if not atoms:
                return "I don't know anything yet."
            atoms.sort(key=lambda a: (a.confidence, a.id or 0), reverse=True)
            shown = atoms[:KNOWLEDGE_REPORT_LIMIT]
            lines = [f"{a.statement()} [{a.status.value}, conf={a.confidence:.2f}]" for a in shown]
            opener = ("Here are a few things I know: " if len(atoms) > len(shown)
                      else "Here's what I know: ")
            return opener + " | ".join(lines)
        if cmd == "open_questions":
            qs = self.store.pending_questions()
            if not qs:
                return "No open questions right now."
            return "Open questions: " + " | ".join(q["question"] for q in qs)
        if cmd == "stop":
            self._questions_declined = True
            self._end_questions()
            return "Okay, I'll stop asking."
        if cmd == "ask":
            # Asked for outright, so no consent step, and an earlier "stop"
            # gives way to it.
            self._questions_declined = False
            return self._ask_next() or "I've nothing to ask you just now."
        if cmd == "sleep":
            return self.nap() if self.nap else self.sleep()
        if cmd == "exit":
            return "Okay. Bye."
        return "Okay."

    # --------------------------------------------------------------- claims
    def _record(self, subject: str, relation: str, obj: str, scope: str, polarity: int,
                utt: Optional[Utterance] = None) -> str:
        """Record one claim -- a statement, a denial, or a yes or a no to a
        question -- and say how it bears on what was believed before it.

        "X is not Y" and a denial of "X is Y" are one claim, so the claim is
        put in positive form first: an object "not Y" becomes Y with the
        polarity reversed. It then lands on the belief it is about: "X is Y"
        if that is on file; failing that "X is not Y", with the polarity
        reversed again; failing both, it is a new belief.
        """
        if grammar.same_referent(subject, obj):
            if polarity > 0 and utt is not None:
                return self._vacuous_reply(utt)
            return f"Nothing can fail to be itself, so I won't record that about {subject}."
        if self.person.is_unverified and grammar.refers_to(subject, self.person.name):
            claimed = self.person.name[:-len(Person.UNVERIFIED_SUFFIX)]
            return f"I can't be sure you're {claimed}, so I won't note anything about you yet."

        low = grammar.norm_phrase(obj)
        if low.startswith("not "):
            obj, polarity = obj.strip()[4:].strip(), -polarity
        atom = self._atom_for(subject, relation, obj, scope)
        if atom is not None:
            return self._add_evidence(atom, polarity, utt)
        negative = self._atom_for(subject, relation, f"not {obj}", scope)
        if negative is not None:
            return self._add_evidence(negative, -polarity, utt)
        return self._new_belief(subject, relation, obj if polarity > 0 else f"not {obj}", scope, utt)

    def _add_evidence(self, atom: Atom, polarity: int, utt: Optional[Utterance]) -> str:
        """Evidence for or against a belief already on file.

        Whether it agrees is judged against the belief's standing before it
        arrived: a contradicted belief stands as false, so denying it agrees
        with what CHLOE already holds, and the source's trust moves up.
        """
        before = atom.status
        agrees = (polarity > 0) != (before == AtomStatus.CONTRADICTED)
        prov = self._attach_provenance(atom, polarity, utt)
        if agrees:
            trust_mod.update_trust_on_corroboration(self.person, atom.domain)
        else:
            trust_mod.update_trust_on_contradiction(self.person, atom.domain)
        self.store.save_person(self.person)
        self._recompute(atom)
        entailment.carry(self.store, atom, prov)
        clause = self._say_atom(atom)

        if before == AtomStatus.HYPOTHESIS:
            if agrees:
                return f"Good -- I'd worked that out myself, and now I have it from you: {clause}."
            return f"Thanks -- I had only guessed that {clause}, so I'll think again."
        if before == AtomStatus.CONTRADICTED:
            if agrees:
                return f"That matches what others have told me: I don't believe that {clause}."
            return f"I'll note that, but others have denied it, so I can't say yet that {clause}."
        if agrees:
            return f"Good, that matches what I already believed: {clause}."
        questions.queue(self.store, "denial", atom)
        return f"Okay -- that conflicts with what I believed. I'll double check: {clause}."

    def _new_belief(self, subject: str, relation: str, obj: str, scope: str,
                    utt: Optional[Utterance]) -> str:
        """A claim nothing on file speaks to directly.

        If a different value stands on the same subject, relation and scope,
        the claim is also evidence against it -- this prototype cannot tell a
        rival value from a compatible one -- and the question of which holds
        is queued. The new value is kept as a belief of its own either way,
        so what was said survives that judgement.
        """
        atom = Atom(id=None, subject=subject, relation=relation, object=obj, scope=scope,
                    status=AtomStatus.CANDIDATE, confidence=0.5)
        standing = questions.rivals(self.store, atom)
        incumbent = max(standing, key=lambda a: a.confidence) if standing and not atom.negative else None
        atom = self.store.save_atom(atom)
        if incumbent is not None:
            # Carried from the new value, so that if sleep later finds the
            # two compatible, this dispute can be voided (entailment.py).
            self._attach_provenance(incumbent, polarity=-1, utt=utt, via=atom)
            trust_mod.update_trust_on_contradiction(self.person, incumbent.domain)
            self.store.save_person(self.person)
            self._recompute(incumbent)
        self._attach_provenance(atom, polarity=1, utt=utt)
        self._recompute(atom)
        if incumbent is None:
            return f"Okay, I'll remember that {self._say_atom(atom)}."
        questions.queue(self.store, "contradiction", incumbent)
        return (f"Hmm, that contradicts what I was told before "
                f"({self._say(incumbent.subject, incumbent.relation, incumbent.object)}). "
                f"I'll keep both and ask around.")

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
        atom = self._find_atom_for_statement(utt.subject, utt.relation, utt.obj, utt.scope,
                                             exclude_contradicted=True)
        if atom is None:
            self.last_stance = Stance.UNKNOWN
            if self._find_atom_for_statement(utt.subject, utt.relation, utt.obj, utt.scope) is not None:
                return ("I've been told conflicting things about that, so I'd rather not say "
                        "either way until it's settled.")
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
        # A belief under unresolved dispute answers nothing. It keeps its row
        # and its provenance, and a value on the same subject that still
        # stands answers in its place.
        candidates = [a for a in self.store.find_atoms_about(utt.subject)
                      if a.status != AtomStatus.CONTRADICTED]
        if not candidates:
            questions.queue(self.store, "unknown_term", term=utt.subject)
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
        candidates.sort(key=lambda a: (a.confidence, a.id or 0), reverse=True)
        best, rest = candidates[0], candidates[1:]
        reply = f"{grammar.capitalise(self._say_atom(best))} (confidence {best.confidence:.2f})."
        if rest:
            reply += " " + self._also(utt.subject, rest)
        return reply

    def _also(self, subject: str, atoms: list) -> str:
        """The rest of what is held about the subject, named as far as
        WH_ANSWER_LIMIT allows and counted beyond it."""
        speaker = self.person.name if self.person else ""
        subj = grammar.swap_pronoun(subject, speaker)
        said = [grammar.predicate(a.relation, a.object, a.scope, speaker, subj)
                for a in atoms[:WH_ANSWER_LIMIT - 1]]
        reply = f"I also know that {subj} {_join(said)}"
        held = len(atoms) - len(said)
        if held:
            reply += f", and {held} more thing{'s' if held > 1 else ''} about {subj}"
        return reply + "."

    # ------------------------------------------------------------- internals
    def _atom_for(self, subject: str, relation: str, obj: str, scope: str) -> Optional[Atom]:
        """The belief this phrasing states, if it is on file (grammar.identity)."""
        wanted = grammar.identity(subject, relation, obj, scope)
        for atom in self.store.find_atoms(subject, relation):
            if atom.identity() == wanted:
                return atom
        return None

    def _find_atom_for_statement(self, subject: str, relation: str, obj: str, scope: str,
                                 exclude_contradicted: bool = False) -> Optional[Atom]:
        """The belief a yes/no question should be answered from: the one it
        names if on file, else the strongest value on the same subject,
        relation and scope.

        `exclude_contradicted` serves the answering paths. A belief under
        unresolved dispute is not used to answer, while it keeps its row and
        its provenance.
        """
        scope_norm = grammar.norm_phrase(scope)
        candidates = [a for a in self.store.find_atoms(subject, relation)
                      if grammar.norm_phrase(a.scope) == scope_norm]
        if exclude_contradicted:
            candidates = [a for a in candidates if a.status != AtomStatus.CONTRADICTED]
        if not candidates:
            return None
        wanted = grammar.identity(subject, relation, obj, scope)
        exact = [a for a in candidates if a.identity() == wanted]
        return exact[0] if exact else max(candidates, key=lambda a: a.confidence)

    def _attach_provenance(self, atom: Atom, polarity: int, utt: Optional[Utterance] = None,
                           via: Optional[Atom] = None) -> Provenance:
        interaction = self._log(ROLE_EVIDENCE, atom.statement(), commit_only=True)
        prov = Provenance(
            person_id=self.person.id,
            interaction_id=interaction.id if interaction else 0,
            polarity=polarity,
            # Recorded and inspectable, but kept out of the
            # belief-confidence maths -- see models.Provenance.
            parse_confidence=utt.extra.get("parse_confidence") if utt else None,
            via_atom_id=via.id if via else None,
        )
        self.store.add_provenance(atom.id, prov)
        atom.provenance.append(prov)
        return prov

    def _recompute(self, atom: Atom) -> None:
        entailment.recompute(self.store, atom)

    def _log(self, role: str, text: str, commit_only: bool = False) -> Interaction:
        interaction = Interaction(id=None, person_id=self.person.id if self.person else 0, role=role, text=text)
        return self.store.log_interaction(interaction)
