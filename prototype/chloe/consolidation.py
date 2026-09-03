"""
Offline consolidation -- "sleep" / "dreaming".

From the design notes, 'Sleeping': during periods of inactivity CHLOE should
cross-match existing ideas, derive possible new ideas, detect
inconsistencies, compress redundant knowledge, and prepare questions whose
answers would maximise future information gain. "Conversation gathered
evidence. Sleep produced understanding."

This module runs that pass over everything currently in the KnowledgeStore.
It is triggered explicitly (the "sleep" command in dialogue.py), mirroring
the original chloe.sh's respawn-on-exit-code-66 idea of a distinct
offline phase, without needing an actual process restart.

Crucially, hypothesis generation only uses relation properties that have
been explicitly declared via store.declare_relation(...) -- see
models.RelationProperties and its docstring. CHLOE.md's 'Reasoning'
section is explicit that transitivity/symmetry must never be assumed
just because English suggests it.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from . import trust as trust_mod
from .models import Atom, AtomStatus

# treat these as the same underlying relation for dedup/merge purposes only
# (surface synonyms, not a claim about semantics)
RELATION_SYNONYMS = {"is": "is", "are": "is", "was": "is", "were": "is", "means": "is"}

MAX_VERIFICATION_QUESTIONS_PER_SLEEP = 3


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
    _queue_verification_for_weak_atoms(store, report)

    return report


def _norm_relation(rel: str) -> str:
    return RELATION_SYNONYMS.get(rel.strip().lower(), rel.strip().lower())


def _merge_duplicates(store, report: SleepReport) -> None:
    atoms = store.all_atoms()
    seen: Dict[tuple, Atom] = {}
    for atom in atoms:
        sig = (atom.subject.strip().lower(), _norm_relation(atom.relation), atom.object.strip().lower(), atom.scope.strip().lower())
        if sig not in seen:
            seen[sig] = atom
            continue
        keeper = seen[sig]
        # merge atom into keeper: combine provenance, drop the duplicate
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
            continue  # never assume transitivity for undeclared relations
        by_subject = {a.subject.strip().lower(): a for a in group}
        for a in group:
            bridge = by_subject.get(a.object.strip().lower())
            if bridge is None or bridge is a:
                continue
            # a: X rel Y, bridge: Y rel Z  => hypothesize X rel Z
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
                status=AtomStatus.HYPOTHESIS, confidence=min(a.confidence, bridge.confidence) * 0.8,
            )
            store.save_atom(hyp)
            report.hypotheses_generated += 1
            report.details.append(f"  hypothesis: {hyp.statement()} (derived via declared-transitive '{rel_name}')")
            store.queue_question(
                f"I worked out from what you've told me that {hyp.subject} {hyp.relation} {hyp.object} -- is that right?",
                reason="hypothesis", related_atom_id=hyp.id,
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
