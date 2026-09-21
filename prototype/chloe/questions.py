"""
Open questions: what CHLOE asks, of whom, and when a question has stopped
being worth asking.

A question is stored as a reason and the thing it is about -- an atom, a
pair of atoms, or an unknown term -- never as a finished sentence. It is checked and worded
at the moment it is asked, because by then the belief may have been
settled, merged or retired, and because the wording depends on who is
being asked: "you once told me" is true of one person and false of the
next, and "Bob" is "you" to Bob.

Each reason carries three things: the kind of answer it takes, why CHLOE
is unsure (the preamble), and the condition under which that doubt still
stands. A question whose condition has lapsed is closed rather than asked.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from . import grammar
from .models import Atom, AtomStatus, Person
from .nlu import UtteranceType


class Kind(str, Enum):
    CONFIRM = "confirm"   # about one belief: a yes or a no settles it
    CHOOSE = "choose"     # between rival values: answered by the value
    DEFINE = "define"     # about an unknown term: answered by saying what it is
    IMPLY = "imply"       # whether one belief follows from another: a yes or a no


def rivals(store, atom: Atom) -> List[Atom]:
    """The standing values on this atom's subject, relation and scope,
    itself included. A value is standing if somebody stated it and it has
    not been contradicted. A negative fact ("X is not Y") rules out one
    value and competes with none, and a value that implies this one, or
    follows from it, is compatible with it (entailment.py)."""
    scope = grammar.norm_phrase(atom.scope)
    return [a for a in store.find_atoms(atom.subject, atom.relation)
            if grammar.norm_phrase(a.scope) == scope and not a.negative
            and a.status in (AtomStatus.CANDIDATE, AtomStatus.CONFIRMED)
            and (atom.id is None or a.id == atom.id or not store.linked(a.id, atom.id))]


def names_person(atom: Optional[Atom], person: Person) -> bool:
    """Whether an atom is about this person under either of their names.
    Which self somebody is is the one thing their own word cannot settle,
    so a proposal that names them is not put to them."""
    if atom is None or person is None:
        return True
    return any(person.claims(side) for side in (atom.subject, atom.object))


@dataclass(frozen=True)
class Reason:
    kind: Kind
    preamble: str
    still_open: Callable[..., bool]   # (store, atom, term, other)
    ask: str = ""              # CONFIRM only: the question, around {clause}
    ask_source: str = ""       # CONFIRM only: the question put to the belief's own source
    sources_may_answer: bool = False


def _is(status: AtomStatus):
    return lambda store, atom, term, other: atom is not None and atom.status == status


REASONS = {
    "hypothesis": Reason(
        Kind.CONFIRM, "This one is my own inference, not something anybody told me",
        _is(AtomStatus.HYPOTHESIS), ask="Am I right that {clause}?"),
    "llm_hypothesis": Reason(
        Kind.CONFIRM, "This one I put together myself, out of more than one thing I was told",
        _is(AtomStatus.HYPOTHESIS), ask="Am I right that {clause}?"),
    "weak_candidate_recheck": Reason(
        Kind.CONFIRM, "This has only ever come from one person",
        lambda store, atom, term, other: atom is not None and atom.status == AtomStatus.CANDIDATE
        and len({p.person_id for p in atom.evidence}) <= 1,
        ask="I've been told that {clause}. Is that right?",
        ask_source="You once told me that {clause}. Is that still true?",
        sources_may_answer=True),
    "denial": Reason(
        Kind.CONFIRM, "I believed this, and then someone denied it",
        _is(AtomStatus.CANDIDATE), ask="Is it true that {clause}?"),
    "unresolved_contradiction": Reason(
        Kind.CONFIRM, "I still can't settle this one",
        _is(AtomStatus.CONTRADICTED), ask="Is it true that {clause}?"),
    "contradiction": Reason(
        Kind.CHOOSE, "My sources disagree here",
        lambda store, atom, term, other: atom is not None and len(rivals(store, atom)) >= 2),
    "unknown_term": Reason(
        Kind.DEFINE, "This one is a blank for me",
        lambda store, atom, term, other: bool(term) and not store.find_atoms_about(term)),
    # The LLM proposed that two subjects name one thing. Nobody has said so,
    # and the claimant cannot say it of themselves (for_person), so the
    # proposal waits for someone else.
    "coreference": Reason(
        Kind.CONFIRM, "These might be two names for one thing, though nobody has told me so",
        _is(AtomStatus.HYPOTHESIS),
        ask="Am I right that {clause} -- two names for the same thing?"),
    # The LLM proposed that one stated value implies another, but the words
    # alone do not carry it ("risky" -> "dangerous"). Its reading of the
    # words is a conjecture like any other, and a person settles it.
    "implication": Reason(
        Kind.IMPLY, "I think one of these follows from the other, but that's only my reading of the words",
        lambda store, atom, term, other: atom is not None and other is not None
        and not store.judged(atom.id, other.id)),
}
DEFAULT_PREAMBLE = "I'm not sure I have this right"


def reason_of(q: dict) -> Optional[Reason]:
    return REASONS.get(q.get("reason"))


def kind_of(q: dict) -> Optional[Kind]:
    r = reason_of(q)
    return r.kind if r else None


# ------------------------------------------------------------- queueing

def queue(store, reason: str, atom: Optional[Atom] = None, term: Optional[str] = None,
          other: Optional[Atom] = None) -> bool:
    """Open a question, or bring the reason of an existing one up to date.
    Returns True if a new question was opened. For an implication, `atom`
    is the belief that would imply `other`."""
    label = describe_parts(store, reason, atom, term, speaker="", other=other)
    return store.queue_question(label or reason, reason,
                                related_atom_id=atom.id if atom else None, term=term,
                                other_atom_id=other.id if other else None)


# ---------------------------------------------------------- still worth it?

def still_open(store, q: dict) -> bool:
    r = reason_of(q)
    return r is not None and r.still_open(store, store.atom_by_id(q.get("related_atom_id")), q.get("term"),
                                          store.atom_by_id(q.get("other_atom_id")))


def for_person(store, q: dict, person: Person) -> bool:
    """Whether this person has anything to add. Someone who has already
    given evidence on a belief has answered it in effect, unless the
    question exists precisely to go back to its source -- and then only
    once: a source who has confirmed it again is not asked a third time."""
    r = reason_of(q)
    if q.get("reason") == "coreference":
        return not names_person(store.atom_by_id(q.get("related_atom_id")), person)
    if r is None or r.kind in (Kind.DEFINE, Kind.IMPLY):
        return True     # about words, not about a belief anyone gave evidence on
    atom = store.atom_by_id(q.get("related_atom_id"))
    if atom is None:
        return False
    involved = [atom] + (rivals(store, atom) if r.kind == Kind.CHOOSE else [])
    mine = [p for a in involved for p in a.evidence if p.person_id == person.id]
    return not mine or (r.sources_may_answer and len(mine) == 1)


def waits_on(store, q: dict, pending: List[dict]) -> bool:
    """Whether another open question has to be settled first. Which of two
    values holds is not worth asking while it is still open whether one
    implies the other: a yes there makes them compatible, and the choice
    lapses."""
    if kind_of(q) != Kind.CHOOSE:
        return False
    atom = store.atom_by_id(q.get("related_atom_id"))
    if atom is None:
        return False
    values = {a.id for a in rivals(store, atom)}
    return any(kind_of(p) == Kind.IMPLY and p.get("related_atom_id") in values
               and p.get("other_atom_id") in values for p in pending)


def next_for(store, person: Person) -> Optional[dict]:
    """The next question worth putting to this person. Questions whose
    doubt has lapsed are closed on the way past; questions this person
    cannot help with, or that wait on another, stay open for later."""
    pending = store.pending_questions()
    for q in pending:
        if not still_open(store, q):
            store.mark_question_answered(q["id"])
            continue
        if for_person(store, q, person) and not waits_on(store, q, pending):
            return q
    return None


# ----------------------------------------------------------------- wording

def _or_list(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def describe_parts(store, reason: str, atom: Optional[Atom], term: Optional[str],
                   speaker: str = "", person: Optional[Person] = None,
                   other: Optional[Atom] = None) -> str:
    r = REASONS.get(reason)
    if r is None:
        return ""
    if r.kind == Kind.DEFINE:
        return grammar.capitalise(grammar.wh_clause(term or "", speaker)) + "?"
    if atom is None:
        return ""
    clause = grammar.clause(atom.subject, atom.relation, atom.object, atom.scope, speaker)
    if r.kind == Kind.IMPLY:
        if other is None:
            return ""
        then = grammar.clause(other.subject, other.relation, other.object, other.scope, speaker)
        return f"If {clause}, does that mean {then}?"
    if r.kind == Kind.CHOOSE:
        values = [atom.object] + [a.object for a in rivals(store, atom) if a.id != atom.id]
        head = grammar.clause(atom.subject, atom.relation, _or_list(values), atom.scope, speaker)
        return f"I've heard conflicting things about whether {head} -- which is it?"
    is_source = person is not None and any(p.person_id == person.id and p.polarity > 0
                                           for p in atom.evidence)
    template = r.ask_source if (is_source and r.ask_source) else r.ask
    return template.format(clause=clause)


def render(store, q: dict, person: Optional[Person] = None) -> str:
    """The question as said to this person, prefaced with why CHLOE is
    unsure. With no person, a neutral rendering for inspection."""
    speaker = person.name if person else ""
    r = reason_of(q)
    body = describe_parts(store, q.get("reason"), store.atom_by_id(q.get("related_atom_id")),
                          q.get("term"), speaker, person,
                          other=store.atom_by_id(q.get("other_atom_id"))) or q.get("question", "")
    preamble = r.preamble if r else DEFAULT_PREAMBLE
    return f"{preamble}. {body}" if person else body


def answered_by(store, q: dict, utt) -> bool:
    """Whether a parsed statement or denial speaks to this question. Agreeing
    and disagreeing are both answers -- which it was has already been
    recorded by the ordinary path -- so only what it is about is checked."""
    if utt is None or not utt.subject or utt.type not in (UtteranceType.STATEMENT, UtteranceType.NEGATION):
        return False
    r = reason_of(q)
    if r is None or r.kind == Kind.IMPLY:
        return False    # a question about meaning is answered yes or no
    if r.kind == Kind.DEFINE:
        return grammar.norm_phrase(utt.subject) == grammar.norm_phrase(q.get("term") or "")
    atom = store.atom_by_id(q.get("related_atom_id"))
    return atom is not None and \
        grammar.identity(utt.subject, utt.relation or "", "")[:2] == atom.identity()[:2]
