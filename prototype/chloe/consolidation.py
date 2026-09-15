"""
Offline consolidation -- "sleep" / "dreaming".

During inactivity CHLOE cross-matches existing ideas, derives possible new
ones, detects inconsistencies, compresses redundant knowledge, and prepares
the questions whose answers would tell it most. Conversation gathers
evidence; sleep produces understanding.

This pass runs over everything currently in the KnowledgeStore. It is
triggered by the "sleep" command in dialogue.py, and on a schedule in the
web deployment -- a distinct offline phase, as in the original design,
without an actual process restart.

New ideas come from two passes that do different jobs.
_generate_hypotheses derives what *follows*, using only relation properties
explicitly declared via store.declare_relation(); transitivity and symmetry
are never inferred from the wording of a relation. _generate_hypotheses_llm
proposes what *might* hold, by asking the language model to combine what
CHLOE has been told -- "Felix is a cat" and "Felix is black" giving "Felix
is a black cat", which no relation algebra reaches. The second pass
conjectures, it does not derive, and the distinction is kept: its output is
a HYPOTHESIS like any other, put to a person before it can become a belief.

Privacy: the LLM pass sends the stored triples to the configured server.
With VLLM_BASE_URL unset it does not run at all and the rest of sleep is
unchanged.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import llm_client, trust as trust_mod
from .models import Atom, AtomStatus
# Reused rather than re-implemented: the reply contract for a JSON-returning
# model is the same one the input parser enforces -- validate, never repair.
from .llm_nlu import BadParse, _clean_field, _extract_json

# Treated as the same relation for duplicate detection only -- surface
# synonyms, not a claim about their semantics.
RELATION_SYNONYMS = {"is": "is", "are": "is", "was": "is", "were": "is", "means": "is"}

MAX_VERIFICATION_QUESTIONS_PER_SLEEP = 3

# A hypothesis is never more confident than the weakest atom it rests on,
# and is damped below it.
HYPOTHESIS_DAMPING = 0.8

# What one sleep may add to the queue. The limit is the patience of whoever
# wakes up to the questions, not the model's throughput.
MAX_LLM_HYPOTHESES_PER_SLEEP = int(os.getenv("CHLOE_MAX_LLM_HYPOTHESES", "5"))

# How many atoms are shown to the model, newest first, so a large store
# still makes a bounded request.
MAX_LLM_CONTEXT_ATOMS = int(os.getenv("CHLOE_HYPOTHESIS_CONTEXT_ATOMS", "60"))

# The model's own confidence is a filter, not evidence: below this the
# candidate is dropped, above it the number plays no part in the belief
# maths. Same separation as Provenance.parse_confidence.
MIN_LLM_HYPOTHESIS_CONFIDENCE = float(os.getenv("CHLOE_HYPOTHESIS_MIN_CONFIDENCE", "0.6"))


def _grounded_only() -> bool:
    """Whether a proposal may use words that appear in no atom it cites.

    On (the default), the model can only recombine what it was shown, so its
    own knowledge cannot enter the store as a belief-in-waiting. Off is a
    different architecture -- the model conjecturing from what it knows, with
    the person who answers becoming the source -- and the switch exists so
    that can be probed on a live endpoint without a code change.
    """
    return os.getenv("CHLOE_HYPOTHESIS_GROUNDED", "1").strip().lower() not in ("0", "false", "no")


@dataclass
class SleepReport:
    contradictions_flagged: int = 0
    merged: int = 0
    hypotheses_generated: int = 0
    verification_questions_queued: int = 0
    details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Sleep complete. Contradictions flagged: {self.contradictions_flagged}, "
            f"merged duplicates: {self.merged}, hypotheses generated: {self.hypotheses_generated}, "
            f"new verification questions: {self.verification_questions_queued}."
        ]
        lines.extend(self.details)
        return "\n".join(lines)


def sleep(store) -> SleepReport:
    report = SleepReport()
    people_by_id = {p.id: p for p in store.all_people()}

    _merge_duplicates(store, report)
    _flag_contradictions(store, report)
    _generate_hypotheses(store, report, people_by_id)
    _generate_hypotheses_llm(store, report)
    _queue_verification_for_weak_atoms(store, report)

    return report


def _norm_relation(rel: str) -> str:
    return RELATION_SYNONYMS.get(rel.strip().lower(), rel.strip().lower())


def _signature(subject: str, relation: str, obj: str) -> tuple:
    return (subject.strip().lower(), _norm_relation(relation), obj.strip().lower())


def _merge_duplicates(store, report: SleepReport) -> None:
    atoms = store.all_atoms()
    seen: Dict[tuple, Atom] = {}
    for atom in atoms:
        sig = (atom.subject.strip().lower(), _norm_relation(atom.relation), atom.object.strip().lower(), atom.scope.strip().lower())
        if sig not in seen:
            seen[sig] = atom
            continue
        keeper = seen[sig]
        # Combine provenance onto the keeper, then drop the duplicate.
        for prov in atom.provenance:
            store.add_provenance(keeper.id, prov)
            keeper.provenance.append(prov)
        store.delete_atom(atom.id)
        report.merged += 1
        report.details.append(f"  merged duplicate '{atom.statement()}' into atom #{keeper.id}")


def _flag_contradictions(store, report: SleepReport) -> None:
    for atom in store.all_atoms():
        if atom.status == AtomStatus.CONTRADICTED:
            report.contradictions_flagged += 1
            store.queue_question(
                f"I still have conflicting evidence about '{atom.subject} {atom.relation} ...' "
                f"(currently: {atom.statement()}, confidence {atom.confidence:.2f}). Can you help me settle it?",
                reason="unresolved_contradiction", related_atom_id=atom.id,
            )


def _generate_hypotheses(store, report: SleepReport, people_by_id: dict) -> None:
    atoms = [a for a in store.all_atoms() if a.status != AtomStatus.CONTRADICTED]
    by_relation: Dict[str, List[Atom]] = {}
    for a in atoms:
        by_relation.setdefault(_norm_relation(a.relation), []).append(a)

    for rel_name, group in by_relation.items():
        props = store.get_relation(rel_name)
        if not props.transitive:
            continue  # undeclared relations are not assumed transitive
        by_subject = {a.subject.strip().lower(): a for a in group}
        for a in group:
            bridge = by_subject.get(a.object.strip().lower())
            if bridge is None or bridge is a:
                continue
            # a: X rel Y, bridge: Y rel Z  =>  hypothesis X rel Z
            existing = None
            for cand in group:
                if cand.subject.strip().lower() == a.subject.strip().lower() and \
                   cand.object.strip().lower() == bridge.object.strip().lower():
                    existing = cand
                    break
            if existing is not None:
                continue
            hyp = Atom(
                id=None, subject=a.subject, relation=a.relation, object=bridge.object, scope=a.scope,
                status=AtomStatus.HYPOTHESIS, confidence=min(a.confidence, bridge.confidence) * HYPOTHESIS_DAMPING,
            )
            store.save_atom(hyp)
            report.hypotheses_generated += 1
            report.details.append(f"  hypothesis: {hyp.statement()} (derived via declared-transitive '{rel_name}')")
            store.queue_question(
                f"I worked out from what you've told me that {hyp.subject} {hyp.relation} {hyp.object} -- is that right?",
                reason="hypothesis", related_atom_id=hyp.id,
            )
            report.verification_questions_queued += 1


# --------------------------------------------------------- LLM hypotheses

_HYPOTHESIS_SYSTEM_PROMPT = """\
You are the consolidation step of CHLOE, a knowledge system that stores what
people have told it as subject-relation-object triples with an optional
scope (the condition under which a statement holds). CHLOE is asleep. You
are shown the facts it has been told, and your only job is to propose
statements it has NOT been told but which those facts, put together,
suggest.

Nothing you propose is believed. Each proposal is put to a person as a
question when CHLOE wakes, and only their answer settles it. A good
proposal is therefore one worth asking about, not one you are certain of.

Return ONLY a JSON object (no prose, no code fences) of exactly this shape:

  {"hypotheses": [
     {"subject": string, "relation": string, "object": string,
      "scope": string or null, "sources": [fact numbers],
      "confidence": number between 0 and 1}
  ]}

Rules:
- Combine the facts you are given. Never use knowledge of your own: every
  word you write must already appear in the facts you cite.
- "sources" lists the numbers of the facts a proposal rests on -- at least
  one, usually two. A proposal citing nothing is not a proposal.
- Do not restate a fact you were given, and do not merely reword one.
- Keep the wording of the facts: articles, plurals and names as written.
- "confidence" is how strongly those facts alone support the proposal.
- ONE value per proposal. "subject" and "object" each name a single thing.
  Never join two with "and", "or" or a comma: "blue in daytime, and grey
  when overcast" is two facts, not one, and they are already on file.
- A condition goes in "scope", never inside "object", and only if a fact you
  cite carries it. If the facts you would combine hold under DIFFERENT
  conditions, propose nothing: there is nothing to add.
- Propose nothing rather than something weak. {"hypotheses": []} is a good
  answer and often the correct one.

Example facts:
  1. Felix | is | a cat
  2. Felix | is | black
  3. the sky | is | blue
Answer:
{"hypotheses": [{"subject": "Felix", "relation": "is", "object": "a black cat", "scope": null, "sources": [1, 2], "confidence": 0.85}]}

Example facts:
  1. the sky | is | blue | scope: it is daytime
  2. Dan | is | a physicist
Answer:
{"hypotheses": []}

Example facts:
  1. the sky | is | blue | scope: it is daytime
  2. the sky | is | grey | scope: it is overcast
Answer:
{"hypotheses": []}

Example facts:
  1. Claire | is | an artist | scope: she is at work
  2. Claire | is | tired | scope: she is at work
Answer:
{"hypotheses": [{"subject": "Claire", "relation": "is", "object": "a tired artist", "scope": "she is at work", "sources": [1, 2], "confidence": 0.7}]}
"""

_WORD_RE = re.compile(r"[a-z0-9']+")

# An atom is one triple with one scope, so a proposal naming two things at
# once is not a hypothesis this representation can hold -- it is two, and
# both are usually on file already.
_COMPOUND_RE = re.compile(r"[,;]|\s(?:and|or|but)\s", re.IGNORECASE)

# Conditions live in the scope field. One inside the object would be stored
# as part of the value and matched against rival values as if it were one.
_CONDITION_RE = re.compile(r"\s(?:when|if|while|during|unless|whenever)\s", re.IGNORECASE)

# Words a proposal may use without them appearing in its sources: they carry
# no content of their own, and demanding them back would refuse "a black
# cat" for the sake of its article.
_FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "and", "or", "is", "are", "was", "were", "be", "been", "am", "not",
    "that", "this", "these", "those", "its", "his", "her", "their", "some", "it",
}


def _fold(word: str) -> str:
    """Crude singular fold, enough to let 'cats' and 'a cat' meet. Nothing
    here is linguistics; it is the smallest normalisation that stops the
    grounding check refusing a proposal over a plural."""
    w = word.strip().lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def _vocabulary(atoms: List[Atom]) -> set:
    vocab = set()
    for atom in atoms:
        for field_text in (atom.subject, atom.relation, atom.object, atom.scope):
            for word in _WORD_RE.findall((field_text or "").lower()):
                vocab.add(_fold(word))
    return vocab


def _ungrounded_words(text: str, vocab: set) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower())
            if w not in _FUNCTION_WORDS and _fold(w) not in vocab]


def _atom_listing(atoms: List[Atom]) -> str:
    lines = []
    for atom in atoms:
        line = f"{atom.id}. {atom.subject} | {atom.relation} | {atom.object}"
        if atom.scope:
            line += f" | scope: {atom.scope}"
        lines.append(line)
    return "\n".join(lines)


def _validated_hypothesis(cand, by_id: Dict[int, Atom],
                          existing: set) -> Tuple[Optional[Atom], Optional[str]]:
    """One candidate, checked against the contract. Returns the atom to save,
    or None and the reason it was refused. Nothing is patched up: a proposal
    that fails a check is dropped, not corrected into something acceptable."""
    if not isinstance(cand, dict):
        return None, "reply contained something that is not a proposal"

    subject = _clean_field(cand.get("subject"))
    relation = _clean_field(cand.get("relation")).lower()
    obj = _clean_field(cand.get("object"))
    scope = _clean_field(cand.get("scope"))
    if not (subject and relation and obj):
        return None, "a proposal was missing its subject, relation or object"
    label = f"{subject} {relation} {obj}"

    sources = cand.get("sources")
    if not isinstance(sources, list) or not sources:
        return None, f"'{label}' cites no facts"
    try:
        cited = [by_id[int(s)] for s in sources]
    except (TypeError, ValueError, KeyError):
        return None, f"'{label}' cites a fact that was not shown to it"

    confidence = cand.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) \
            or not (0.0 <= float(confidence) <= 1.0):
        return None, f"'{label}' has no usable confidence"
    if float(confidence) < MIN_LLM_HYPOTHESIS_CONFIDENCE:
        return None, f"'{label}' came with confidence {float(confidence):.2f}, below the floor"

    if subject.strip().lower() == obj.strip().lower():
        return None, f"'{label}' says nothing"
    for field_name, text in (("subject", subject), ("object", obj)):
        if _COMPOUND_RE.search(text):
            return None, f"'{label}' names more than one thing in its {field_name}"
    if _CONDITION_RE.search(obj):
        return None, f"'{label}' puts a condition in its object"

    # A proposal inherits the condition of the facts it rests on, and cannot
    # span two: statements holding under different conditions have nothing to
    # combine, which is why they are separate atoms in the first place.
    scopes = {a.scope.strip() for a in cited if a.scope.strip()}
    if len(scopes) > 1:
        return None, (f"'{label}' combines facts holding under different conditions "
                      f"({'; '.join(sorted(scopes))})")
    if scopes and scope.strip().lower() != next(iter(scopes)).lower():
        return None, f"'{label}' drops or alters the condition '{next(iter(scopes))}'"
    if scope and not scopes:
        return None, f"'{label}' adds a condition nobody stated"
    if _signature(subject, relation, obj) in existing:
        return None, f"'{label}' is already on file"

    if _grounded_only():
        stray = _ungrounded_words(" ".join([subject, relation, obj, scope]), _vocabulary(cited))
        if stray:
            return None, (f"'{label}' uses words from nowhere on file: "
                          f"{', '.join(sorted(set(stray)))}")

    # The weakest fact it rests on sets the ceiling; the model's own
    # confidence got the proposal this far and plays no further part.
    floor = min(a.confidence for a in cited)
    return Atom(
        id=None, subject=subject, relation=relation, object=obj, scope=scope,
        domain=cited[0].domain, status=AtomStatus.HYPOTHESIS,
        confidence=round(floor * HYPOTHESIS_DAMPING, 4),
    ), None


def _generate_hypotheses_llm(store, report: SleepReport) -> None:
    """Ask the language model to combine what CHLOE has been told.

    The pass above derives what follows from declared relation properties;
    this one proposes what those properties cannot reach -- two things said
    about the same subject, a scope that carries across, a name that turns
    out to be a description. Both end in the same place, a HYPOTHESIS atom
    and a queued question, because an idea nobody has confirmed is not a
    belief however it was arrived at.

    Only atoms someone has actually stated are shown: a conjecture built on
    a conjecture would compound an error nobody has yet had the chance to
    correct. A broken exchange leaves the pass with nothing and the rest of
    sleep unaffected.
    """
    if not llm_client.is_configured():
        return

    stated = [a for a in store.all_atoms()
              if a.status in (AtomStatus.CANDIDATE, AtomStatus.CONFIRMED)]
    if len(stated) < 2:
        return
    stated.sort(key=lambda a: a.updated_at, reverse=True)
    shown = stated[:MAX_LLM_CONTEXT_ATOMS]

    try:
        payload = _extract_json(llm_client.chat(
            [
                {"role": "system", "content": _HYPOTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": "Facts:\n" + _atom_listing(shown)},
            ],
            temperature=0.0,  # consolidation should not vary run to run
            max_tokens=600,
        ))
    except (llm_client.LLMUnavailable, BadParse) as e:
        report.details.append(f"  no hypotheses from the model: {e}")
        return

    candidates = payload.get("hypotheses")
    if not isinstance(candidates, list):
        report.details.append("  no hypotheses from the model: reply carried no 'hypotheses' list")
        return

    by_id = {a.id: a for a in shown}
    existing = {_signature(a.subject, a.relation, a.object) for a in store.all_atoms()}
    kept = 0
    for cand in candidates:
        if kept >= MAX_LLM_HYPOTHESES_PER_SLEEP:
            report.details.append("  remaining proposals left for another night")
            break
        hyp, refusal = _validated_hypothesis(cand, by_id, existing)
        if hyp is None:
            report.details.append(f"  proposal refused: {refusal}")
            continue

        store.save_atom(hyp)
        existing.add(_signature(hyp.subject, hyp.relation, hyp.object))
        kept += 1
        report.hypotheses_generated += 1
        cited = ", ".join(f"#{int(s)}" for s in cand["sources"])
        report.details.append(f"  hypothesis: {hyp.statement()} (put together from {cited})")
        store.queue_question(
            f"Am I right that {hyp.statement()}?",
            reason="llm_hypothesis", related_atom_id=hyp.id,
        )
        report.verification_questions_queued += 1


def _queue_verification_for_weak_atoms(store, report: SleepReport) -> None:
    weak = [a for a in store.all_atoms() if a.status == AtomStatus.CANDIDATE and len(a.provenance) <= 1]
    weak.sort(key=lambda a: a.updated_at)
    for atom in weak[:MAX_VERIFICATION_QUESTIONS_PER_SLEEP]:
        queued_before = len(store.pending_questions())
        store.queue_question(
            f"You once told me {atom.statement()}. Is that still true?",
            reason="weak_candidate_recheck", related_atom_id=atom.id,
        )
        if len(store.pending_questions()) > queued_before:
            report.verification_questions_queued += 1
