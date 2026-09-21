"""
Offline consolidation -- "sleep" / "dreaming".

During inactivity CHLOE cross-matches existing ideas, derives possible new
ones, detects inconsistencies, compresses redundant knowledge, and prepares
the questions whose answers would tell it most. Conversation gathers
evidence; sleep produces understanding.

This pass runs over everything currently in the KnowledgeStore. It is
triggered by the "sleep" command in dialogue.py, and on a schedule in the
web deployment -- a distinct offline phase, without a process restart.

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

from . import entailment, grammar, llm_client, questions, trust as trust_mod
from .models import Atom, AtomStatus, Person
# Reused rather than re-implemented: the reply contract for a JSON-returning
# model is the same one the input parser enforces -- validate, never repair.
from .llm_nlu import BadParse, _clean_field, _extract_json

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
    implications_linked: int = 0
    hypotheses_generated: int = 0
    verification_questions_queued: int = 0
    details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Sleep complete. Contradictions flagged: {self.contradictions_flagged}, "
            f"merged duplicates: {self.merged}, implications linked: {self.implications_linked}, "
            f"hypotheses generated: {self.hypotheses_generated}, "
            f"new verification questions: {self.verification_questions_queued}."
        ]
        lines.extend(self.details)
        return "\n".join(lines)


def sleep(store) -> SleepReport:
    report = SleepReport()
    people_by_id = {p.id: p for p in store.all_people()}

    _apply_coreference(store, report)
    _merge_duplicates(store, report, people_by_id)
    _link_implications(store, report)
    _flag_contradictions(store, report)
    _generate_hypotheses(store, report, people_by_id)
    _generate_hypotheses_llm(store, report)
    _propose_coreference(store, report)
    _queue_verification_for_weak_atoms(store, report)

    return report


def _norm_relation(rel: str) -> str:
    return grammar.norm_relation(rel)


def _signature(subject: str, relation: str, obj: str) -> tuple:
    return grammar.identity(subject, relation, obj)[:3]


def _apply_coreference(store, report: SleepReport) -> None:
    """Move what is held about one subject onto another, where somebody has
    confirmed that the two name one thing. The proposal comes from an earlier
    sleep and is put to a person first (questions "coreference"), so nothing
    moves on the model's reading alone. Duplicates fold in the pass below."""
    for q in store.questions_by_reason("coreference"):
        atom = store.atom_by_id(q.get("related_atom_id"))
        if atom is None or atom.status == AtomStatus.HYPOTHESIS:
            continue
        people = {p.id: p for p in store.all_people()}
        if not any(prov.polarity > 0 and not questions.names_person(atom, people.get(prov.person_id))
                   for prov in atom.evidence):
            continue    # only somebody who is not the one being identified
        alias, canonical = atom.subject, atom.object
        if grammar.same_referent(alias, canonical):
            continue
        moved = 0
        for other in store.find_atoms_about(alias):
            if other.id == atom.id:
                continue
            other.subject = canonical
            store.save_atom(other)
            moved += 1
        if moved:
            report.details.append(
                f"  '{alias}' and '{canonical}' are one: {moved} belief(s) moved")


def _merge_duplicates(store, report: SleepReport, people_by_id: dict) -> None:
    """Atoms that state one belief in different words (grammar.identity)
    become one. The best-attested phrasing is kept as the wording -- the
    one the most people have given evidence on, the earliest on a tie --
    and takes every piece of evidence and every open question of the
    others, then has its confidence and status worked out again."""
    groups: Dict[tuple, List[Atom]] = {}
    for atom in store.all_atoms():
        groups.setdefault(atom.identity(), []).append(atom)
    for group in groups.values():
        if len(group) < 2:
            continue
        keeper = max(group, key=lambda a: (len(a.evidence), -a.id))
        for atom in group:
            if atom is keeper:
                continue
            for prov in atom.provenance:
                store.add_provenance(keeper.id, prov)
                keeper.provenance.append(prov)
            store.repoint_atom(atom.id, keeper.id)
            store.delete_atom(atom.id)
            report.merged += 1
            report.details.append(f"  merged '{atom.statement()}' into '{keeper.statement()}' (#{keeper.id})")
        if keeper.evidence:
            entailment.recompute(store, keeper, people_by_id)
        else:
            store.save_atom(keeper)


# ---------------------------------------------------------- implications

_IMPLICATION_SYSTEM_PROMPT = """\
You are part of the consolidation step of CHLOE, a knowledge system that
stores statements as subject-relation-object triples. You are shown
numbered pairs of statements made about the same thing. For each pair,
decide whether one of them entails the other: whenever it is true, the
other must be true as well, by the meaning of the words alone.

"Felix is a black cat" entails "Felix is a cat". "Poking a bear is risky
and dangerous" entails "Poking a bear is dangerous". "Hector is a dog" and
"Hector is a cat" entail nothing of each other, and neither do two
statements that merely could both be true.

Return ONLY a JSON object, no prose and no code fences:
{"implications": [{"pair": <number>, "specific": "A" or "B", "confidence": <number 0-1>}]}
List only the pairs where one statement entails the other; "specific" is
the one that entails. If there are none, return {"implications": []}.
"""


def _implication_candidates(store) -> List[Tuple[Atom, Atom]]:
    """Pairs of values somebody stated on the same subject, relation and
    scope, not yet settled either way -- the pairs that may have been taken
    for rivals."""
    groups: Dict[tuple, List[Atom]] = {}
    for atom in store.all_atoms():
        if atom.status == AtomStatus.HYPOTHESIS or atom.negative:
            continue
        s, r, _, sc = atom.identity()
        groups.setdefault((s, r, sc), []).append(atom)
    pairs = []
    for group in groups.values():
        group.sort(key=lambda a: a.id)
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if not store.judged(a.id, b.id):
                    pairs.append((a, b))
    return pairs[:MAX_LLM_CONTEXT_ATOMS]


def _link_implications(store, report: SleepReport) -> None:
    """Ask the LLM which apparent rivals are one value implying the other
    (entailment.py). The model proposes and the core checks that the pair
    is one it was shown. Where the words carry the implication -- the
    general value uses only words of the specific one -- the two are linked
    now. Where it rests on the model's knowledge of the world ("risky" ->
    "dangerous", "a poodle" -> "a dog"), it becomes a question, and a
    person's yes makes the link: the model's knowledge enters as a
    conjecture for somebody to confirm, never as a settlement. With the
    grounding switch off, every proposal is linked outright."""
    if not llm_client.is_configured():
        return
    pairs = _implication_candidates(store)
    if not pairs:
        return
    listing = "\n".join(f"{n}. A: {a.subject} {a.relation} {a.object}\n   B: {b.subject} {b.relation} {b.object}"
                         for n, (a, b) in enumerate(pairs, 1))
    try:
        payload = _extract_json(llm_client.chat(
            [{"role": "system", "content": _IMPLICATION_SYSTEM_PROMPT},
             {"role": "user", "content": "Pairs:\n" + listing}],
            temperature=0.0, max_tokens=400,
        ))
    except (llm_client.LLMUnavailable, BadParse) as e:
        report.details.append(f"  no implications from the model: {e}")
        return
    found = payload.get("implications")
    if not isinstance(found, list):
        report.details.append("  no implications from the model: reply carried no 'implications' list")
        return
    for item in found:
        proposal, refusal = _validated_implication(item, pairs)
        if proposal is None:
            report.details.append(f"  implication refused: {refusal}")
            continue
        specific, general, grounded = proposal
        if store.judged(specific.id, general.id):
            continue
        if grounded or not _grounded_only():
            voided = entailment.link(store, specific, general)
            report.implications_linked += 1
            report.details.append(f"  '{specific.statement()}' implies '{general.statement()}'"
                                  + (f" ({voided} dispute(s) withdrawn)" if voided else ""))
        elif questions.queue(store, "implication", specific, other=general):
            report.verification_questions_queued += 1
            report.details.append(f"  asking whether '{specific.statement()}' implies "
                                  f"'{general.statement()}' (not carried by the words alone)")


def _validated_implication(item, pairs) -> Tuple[Optional[Tuple[Atom, Atom, bool]], Optional[str]]:
    """A proposal checked against the contract: (specific, general, whether
    the words alone carry it), or None and the reason it was refused."""
    if not isinstance(item, dict):
        return None, "reply contained something that is not a pair"
    n, side, confidence = item.get("pair"), item.get("specific"), item.get("confidence")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(pairs):
        return None, f"pair {n!r} was not one of those shown"
    if side not in ("A", "B"):
        return None, f"pair {n}: 'specific' must be A or B, not {side!r}"
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) \
            or float(confidence) < MIN_LLM_HYPOTHESIS_CONFIDENCE:
        return None, f"pair {n}: confidence {confidence!r} is missing or below the floor"
    a, b = pairs[n - 1]
    specific, general = (a, b) if side == "A" else (b, a)
    grounded = not grammar.ungrounded_words(general.object, grammar.vocabulary([specific.object]))
    return (specific, general, grounded), None


def _flag_contradictions(store, report: SleepReport) -> None:
    for atom in store.all_atoms():
        if atom.status == AtomStatus.CONTRADICTED:
            report.contradictions_flagged += 1
            questions.queue(store, "unresolved_contradiction", atom)


def _generate_hypotheses(store, report: SleepReport, people_by_id: dict) -> None:
    atoms = [a for a in store.all_atoms() if a.status != AtomStatus.CONTRADICTED]
    by_relation: Dict[str, List[Atom]] = {}
    for a in atoms:
        by_relation.setdefault(_norm_relation(a.relation), []).append(a)

    for rel_name, group in by_relation.items():
        props = store.get_relation(rel_name)
        if not props.transitive:
            continue  # undeclared relations are not assumed transitive
        by_subject = {grammar.norm_phrase(a.subject): a for a in group}
        for a in group:
            bridge = by_subject.get(grammar.norm_phrase(a.object))
            if bridge is None or bridge is a:
                continue
            # a: X rel Y, bridge: Y rel Z  =>  hypothesis X rel Z
            existing = None
            for cand in group:
                if grammar.norm_phrase(cand.subject) == grammar.norm_phrase(a.subject) and \
                   grammar.norm_phrase(cand.object) == grammar.norm_phrase(bridge.object):
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
            questions.queue(store, "hypothesis", hyp)
            report.verification_questions_queued += 1


# --------------------------------------------------------- LLM hypotheses

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

    # A name CHLOE could not confirm carries its mark in the subject, and a
    # model shown it reads the mark as a word like any other.
    stated = [a for a in store.all_atoms()
              if a.status in (AtomStatus.CANDIDATE, AtomStatus.CONFIRMED)
              and not a.subject.endswith(Person.UNVERIFIED_SUFFIX)]
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
        questions.queue(store, "llm_hypothesis", hyp)
        report.verification_questions_queued += 1


# ------------------------------------------------- the proposal contract

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


# An atom is one triple with one scope, so a proposal naming two things at
# once is not a hypothesis this representation can hold -- it is two, and
# both are usually on file already.
_COMPOUND_RE = re.compile(r"[,;]|\s(?:and|or|but)\s", re.IGNORECASE)

# Conditions live in the scope field. One inside the object would be stored
# as part of the value and matched against rival values as if it were one.
_CONDITION_RE = re.compile(r"\s(?:when|if|while|during|unless|whenever)\s", re.IGNORECASE)

def _vocabulary(atoms: List[Atom]) -> set:
    return grammar.vocabulary(t for a in atoms for t in (a.subject, a.relation, a.object, a.scope))


def _ungrounded_words(text: str, vocab: set) -> List[str]:
    return grammar.ungrounded_words(text, vocab)


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


# ------------------------------------------------------ coreference

_COREFERENCE_SYSTEM_PROMPT = """\
You are part of the consolidation step of CHLOE, a knowledge system that
stores what people tell it as subject-relation-object triples. You are shown
numbered pairs of subjects, each with what CHLOE has been told about it. For
each pair, decide whether the two are two names for one and the same thing.

CHLOE writes "(unverified)" after a name somebody claimed and could not
confirm, so "Cora (unverified)" and "Cora" may well be one person. "Felix"
and "Hector" are two names for two different cats. Subjects that merely
resemble one another, or that could both be true of one thing without being
it, are not the same thing.

Return ONLY a JSON object, no prose and no code fences:
{"same": [{"pair": <number>, "canonical": "A" or "B", "confidence": <number 0-1>}]}
"canonical" is the name worth keeping, the fuller or more settled of the two.
List only pairs that name one thing. If there are none, return {"same": []}.
"""


def _propose_coreference(store, report: SleepReport) -> None:
    """Ask the model which subjects name one thing, and put each proposal to
    a person. Nothing here merges anything: the proposal is a HYPOTHESIS atom
    and a queued question, and _apply_coreference acts only once somebody who
    is not the subject of it has said yes."""
    if not llm_client.is_configured():
        return
    pairs = _coreference_candidates(store)
    if not pairs:
        return
    listing = "\n".join(f"{n}. A: {a}\n   {_subject_listing(store, a)}\n"
                        f"   B: {b}\n   {_subject_listing(store, b)}"
                        for n, (a, b) in enumerate(pairs, 1))
    try:
        payload = _extract_json(llm_client.chat(
            [{"role": "system", "content": _COREFERENCE_SYSTEM_PROMPT},
             {"role": "user", "content": "Pairs:\n" + listing}],
            temperature=0.0, max_tokens=400,
        ))
    except (llm_client.LLMUnavailable, BadParse) as e:
        report.details.append(f"  no coreference from the model: {e}")
        return
    found = payload.get("same")
    if not isinstance(found, list):
        report.details.append("  no coreference from the model: reply carried no 'same' list")
        return
    for item in found:
        proposal, refusal = _validated_coreference(item, pairs)
        if proposal is None:
            report.details.append(f"  coreference refused: {refusal}")
            continue
        alias, canonical = proposal
        hyp = store.save_atom(Atom(id=None, subject=alias, relation="is", object=canonical,
                                   status=AtomStatus.HYPOTHESIS,
                                   confidence=_coreference_floor(store, alias, canonical)))
        report.hypotheses_generated += 1
        report.details.append(f"  asking whether '{alias}' and '{canonical}' are one")
        if questions.queue(store, "coreference", hyp):
            report.verification_questions_queued += 1


def _subjects(store) -> Dict[str, List[Atom]]:
    """Stated beliefs, by the subject they are about."""
    by_subject: Dict[str, List[Atom]] = {}
    for atom in store.all_atoms():
        if atom.status in (AtomStatus.CANDIDATE, AtomStatus.CONFIRMED):
            by_subject.setdefault(atom.subject, []).append(atom)
    return by_subject


def _coreference_candidates(store) -> List[Tuple[str, str]]:
    """Pairs of subjects worth asking about: two spellings sharing a word,
    with neither the question nor the belief already on file. Sharing a word
    bounds what the model is shown and settles nothing -- two names for one
    thing that share no word ("Bob", "Robert") are not reached this way."""
    by_subject = _subjects(store)
    existing = {_signature(a.subject, a.relation, a.object) for a in store.all_atoms()}
    names = sorted(by_subject)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if grammar.same_referent(a, b) or not (grammar.vocabulary([a]) & grammar.vocabulary([b])):
                continue
            if any(_signature(x, "is", y) in existing for x, y in ((a, b), (b, a))):
                continue
            pairs.append((a, b))
    return pairs[:MAX_LLM_CONTEXT_ATOMS]


def _subject_listing(store, subject: str) -> str:
    atoms = _subjects(store).get(subject, [])
    return "; ".join(f"{a.relation} {a.object}" for a in atoms[:5]) or "nothing on file"


def _coreference_floor(store, alias: str, canonical: str) -> float:
    """A proposal is no more confident than the beliefs that suggested it."""
    by_subject = _subjects(store)
    held = by_subject.get(alias, []) + by_subject.get(canonical, [])
    return round(min([a.confidence for a in held] or [0.5]) * HYPOTHESIS_DAMPING, 4)


def _validated_coreference(item, pairs) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    """A proposal checked against the contract: (alias, canonical), or None
    and the reason it was refused."""
    if not isinstance(item, dict):
        return None, "reply contained something that is not a pair"
    n, side, confidence = item.get("pair"), item.get("canonical"), item.get("confidence")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(pairs):
        return None, f"pair {n!r} was not one of those shown"
    if side not in ("A", "B"):
        return None, f"pair {n}: 'canonical' must be A or B, not {side!r}"
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) \
            or float(confidence) < MIN_LLM_HYPOTHESIS_CONFIDENCE:
        return None, f"pair {n}: confidence {confidence!r} is missing or below the floor"
    a, b = pairs[n - 1]
    canonical, alias = (a, b) if side == "A" else (b, a)
    return (alias, canonical), None


def _queue_verification_for_weak_atoms(store, report: SleepReport) -> None:
    weak = [a for a in store.all_atoms() if a.status == AtomStatus.CANDIDATE
            and len({p.person_id for p in a.evidence}) <= 1]
    weak.sort(key=lambda a: a.updated_at)
    for atom in weak[:MAX_VERIFICATION_QUESTIONS_PER_SLEEP]:
        if questions.queue(store, "weak_candidate_recheck", atom):
            report.verification_questions_queued += 1
