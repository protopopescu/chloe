"""
Taking in the core of another CHLOE instance.

Merging is adding evidence. The other store's people, turns, beliefs,
evidence, implications, relations and questions come across with their
provenance, and each belief lands on the one here that states the same thing
(grammar.identity). A belief whose evidence changes is worked out again with
trust as it stands here. One that arrives with exactly the evidence it had
keeps its confidence and status.

Who is who follows greet(), since an ingested record is a name with no
secret word presented. A name new here comes over with its trust and secret
word. A name held on both sides with no secret on either is one person, who
keeps the trust held here. A name with a secret on either side comes over as
the unverified identity a failed sign-in gets, and its claims about itself
stay behind (dialogue._record).

Rival values meeting for the first time are not recorded as disputing each
other, since neither was said in the face of the other. The question of
which holds is queued instead.

Safe to repeat and to interrupt: turns are matched by fingerprint
(storage.interaction_fingerprint), and evidence or a question already on
file is never added again.
"""

import os
import sqlite3
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from . import entailment, questions
from .models import Atom, AtomStatus, Interaction, Person, Provenance
from .storage import KnowledgeStore


@dataclass
class IngestReport:
    source: str = ""
    people_added: int = 0
    people_matched: int = 0
    people_unverified: int = 0
    turns_added: int = 0
    beliefs_added: int = 0
    beliefs_reinforced: int = 0
    evidence_added: int = 0
    evidence_already_held: int = 0
    self_claims_left: int = 0
    implications_linked: int = 0
    relations_declared: int = 0
    questions_added: int = 0
    rivalries_queued: int = 0
    details: List[str] = field(default_factory=list)

    def counts(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in ("source", "details")}

    def summary(self) -> str:
        lines = [
            f"Ingested {self.source}.",
            f"People: {self.people_added} new, {self.people_matched} already known, "
            f"{self.people_unverified} unverified.",
            f"Turns added: {self.turns_added}. Beliefs: {self.beliefs_added} new, "
            f"{self.beliefs_reinforced} with new evidence.",
            f"Evidence: {self.evidence_added} added, {self.evidence_already_held} already held, "
            f"{self.self_claims_left} self-claims left behind.",
            f"Implications linked: {self.implications_linked}. Relations declared: {self.relations_declared}. "
            f"Questions added: {self.questions_added}, of which {self.rivalries_queued} between rival values.",
        ]
        lines.extend(self.details)
        return "\n".join(lines)


def open_snapshot(path: str, directory: str) -> KnowledgeStore:
    """A consistent copy of a store, opened as one. sqlite3's backup reads
    the source as a single transaction and leaves it untouched, and the copy
    can be brought up to this schema without writing to the original."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    copy = os.path.join(directory, "snapshot.db")
    src = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(copy)
        src.backup(dst)
        dst.close()
    finally:
        src.close()
    return KnowledgeStore(copy)


def ingest(store: KnowledgeStore, source_path: str) -> IngestReport:
    """Merge the store at `source_path` into `store`. The source is only
    read."""
    if os.path.exists(store.path) and os.path.samefile(store.path, source_path):
        raise ValueError("a store cannot ingest itself")
    with tempfile.TemporaryDirectory() as tmp:
        theirs = open_snapshot(source_path, tmp)
        try:
            return _Ingest(store, theirs, source_path).run()
        finally:
            theirs.close()


class _Ingest:
    def __init__(self, ours: KnowledgeStore, theirs: KnowledgeStore, source: str):
        self.ours = ours
        self.theirs = theirs
        self.report = IngestReport(source=source)
        self.people: Dict[int, Person] = {}      # their person id -> ours
        self.turns: Dict[int, int] = {}          # their interaction id -> ours
        self.atoms: Dict[int, Atom] = {}         # their atom id -> ours
        self.touched: Dict[int, Atom] = {}       # our atoms whose evidence changed
        self.created: Dict[int, Atom] = {}       # our atom id -> their atom it was made from
        self.landed: Dict[int, list] = {}        # our atom id -> their atoms whose evidence landed on it
        self.reinforced: set = set()             # our atoms already here, given new evidence

    def run(self) -> IngestReport:
        self.ingest_id = self.ours.begin_ingest(self.report.source)
        self._people()
        self._turns()
        self._relations()
        self._beliefs()
        self._implications()
        self._recompute()
        self._questions()
        self._rivalries()
        self.ours.finish_ingest(self.ingest_id, self.report.counts())
        return self.report

    # ----------------------------------------------------------- people
    def _people(self) -> None:
        for theirs in self.theirs.all_people():
            ours = self.ours.find_person_by_name(theirs.name)
            if ours is None:
                ours = self.ours.add_person(theirs)
                self.report.people_added += 1
            elif not ours.has_secret and not theirs.has_secret:
                self.report.people_matched += 1
            else:
                ours = self.ours.get_or_create_person(theirs.claimed_name + Person.UNVERIFIED_SUFFIX)
                self.report.people_unverified += 1
                self.report.details.append(f"  '{theirs.name}' holds a secret word on one side, "
                                           f"so comes over as '{ours.name}'")
            self.people[theirs.id] = ours

    def _person(self, their_id: int) -> Optional[Person]:
        return self.people.get(their_id)

    # ------------------------------------------------------------ turns
    def _turns(self) -> None:
        held = self.ours.fingerprints()
        names = {p.id: p.name for p in self.theirs.all_people()}
        for turn in self.theirs.all_interactions():
            mark = self.theirs.fingerprint_of(turn, names.get(turn.person_id, ""))
            if mark in held:
                self.turns[turn.id] = held[mark]
                continue
            person = self._person(turn.person_id)
            ours = self.ours.log_interaction(Interaction(
                id=None, person_id=person.id if person else 0, role=turn.role, text=turn.text,
                at=turn.at, origin=mark, ingest_id=self.ingest_id))
            held[mark] = self.turns[turn.id] = ours.id
            self.report.turns_added += 1

    # -------------------------------------------------------- relations
    def _relations(self) -> None:
        """A relation declared there and not here is declared here. Where
        both have declared it, this store's declaration stands."""
        ours = {r.name.casefold(): r for r in self.ours.declared_relations()}
        for rel in self.theirs.declared_relations():
            mine = ours.get(rel.name.casefold())
            if mine is None:
                self.ours.declare_relation(rel)
                self.report.relations_declared += 1
            elif (mine.transitive, mine.symmetric) != (rel.transitive, rel.symmetric):
                self.report.details.append(f"  relation '{rel.name}' is declared differently there; kept as here")

    # ---------------------------------------------------------- beliefs
    def _same_belief(self, atom: Atom) -> Optional[Atom]:
        wanted = atom.identity()
        for ours in self.ours.find_atoms(atom.subject, atom.relation):
            if ours.identity() == wanted:
                return ours
        return None

    def _self_claim(self, atom: Atom, prov: Provenance) -> bool:
        """A claim about themselves by someone whose identity here is
        unsettled, which dialogue._record() would not have stored."""
        person = self._person(prov.person_id)
        return person is not None and person.is_unverified and person.claims(atom.subject)

    def _beliefs(self) -> None:
        """A belief is made here when the first piece of its evidence lands,
        so one whose evidence is all left behind or already held never
        appears. A hypothesis with no evidence comes over as it is. Direct
        evidence goes first, since carried evidence and disputes name another
        belief that has to be here already."""
        theirs = sorted(self.theirs.all_atoms(), key=lambda a: a.id)
        for atom in theirs:
            ours = self._same_belief(atom)
            if ours is not None:
                self.atoms[atom.id] = ours
        for carried in (False, True):
            for atom in theirs:
                for prov in atom.provenance:
                    if (prov.via_atom_id is not None) == carried:
                        self._evidence(atom, prov)
        for atom in theirs:
            if not atom.provenance and atom.id not in self.atoms:
                self._create(atom)

    def _create(self, atom: Atom) -> Atom:
        ours = self.ours.save_atom(Atom(
            id=None, subject=atom.subject, relation=atom.relation, object=atom.object,
            scope=atom.scope, domain=atom.domain, status=atom.status, confidence=atom.confidence,
            created_at=atom.created_at, updated_at=atom.updated_at))
        self.atoms[atom.id] = ours
        self.created[ours.id] = atom
        self.report.beliefs_added += 1
        return ours

    def _evidence(self, atom: Atom, prov: Provenance) -> None:
        if self._self_claim(atom, prov):
            self.report.self_claims_left += 1
            return
        person, turn = self._person(prov.person_id), self.turns.get(prov.interaction_id)
        if person is None or turn is None:
            return      # a row pointing at nothing in its own store
        via = None
        if prov.via_atom_id is not None:
            carried_from = self.atoms.get(prov.via_atom_id)
            if carried_from is None:
                return  # it rests on a belief that did not come over
            via = carried_from.id
        ours = self.atoms.get(atom.id)
        mine = Provenance(person_id=person.id, interaction_id=turn, polarity=prov.polarity, at=prov.at,
                          parse_confidence=prov.parse_confidence, via_atom_id=via, void=prov.void)
        # Direct evidence is one act wherever it now sits, since sleep can
        # move a belief to another wording. Carried evidence is one act per
        # belief it was carried to.
        if via is None:
            held = self.ours.has_provenance(mine)
        else:
            held = ours is not None and self.ours.has_provenance(mine, atom_id=ours.id)
        if held:
            self.report.evidence_already_held += 1
            return
        if ours is None:
            ours = self._create(atom)
        self.ours.add_provenance(ours.id, mine)
        ours.provenance.append(mine)
        self.touched[ours.id] = ours
        self.landed.setdefault(ours.id, []).append(atom.id)
        self.report.evidence_added += 1
        if ours.id not in self.created and ours.id not in self.reinforced:
            self.reinforced.add(ours.id)
            self.report.beliefs_reinforced += 1

    def _copied_whole(self, atom_id: int) -> bool:
        """A belief made here that took all of one atom's evidence and
        nothing else: its evidence is unchanged, so its confidence and
        status stand as they were."""
        source = self.created.get(atom_id)
        landed = self.landed.get(atom_id, [])
        return source is not None and len(landed) == len(source.provenance) \
            and all(a == source.id for a in landed)

    # ----------------------------------------------------- implications
    def _implications(self) -> None:
        """Their links between beliefs, where this store has not judged the
        pair. Where either belief was here already, or took in evidence
        beyond the atom it came from, the link is made with entailment.link(),
        which withdraws the disputes it shows to be none and carries evidence
        across. Two beliefs that came over whole already have all of that."""
        for row in self.theirs.all_entailments():
            specific, general = self.atoms.get(row["specific_id"]), self.atoms.get(row["general_id"])
            if specific is None or general is None or specific.id == general.id:
                continue
            if self.ours.judged(specific.id, general.id):
                continue
            confirmed = self._person(row["confirmed_by"]) if row["confirmed_by"] is not None else None
            confirmed_by = confirmed.id if confirmed else None
            if row["holds"] and self._copied_whole(specific.id) and self._copied_whole(general.id):
                self.ours.add_entailment(specific.id, general.id, holds=True, confirmed_by=confirmed_by)
                self.report.implications_linked += 1
            elif row["holds"]:
                entailment.link(self.ours, specific, general, confirmed_by=confirmed_by)
                self.report.implications_linked += 1
                for atom_id in (specific.id, general.id):
                    self.touched[atom_id] = self.ours.atom_by_id(atom_id)
            else:
                self.ours.add_entailment(specific.id, general.id, holds=False, confirmed_by=confirmed_by)

    def _recompute(self) -> None:
        """Confidence and status worked out again wherever the evidence has
        changed, with trust as it stands here."""
        people = {p.id: p for p in self.ours.all_people()}
        for atom_id in list(self.touched):
            if self._copied_whole(atom_id):
                continue
            atom = self.ours.atom_by_id(atom_id)
            if atom is not None and atom.evidence:
                entailment.recompute(self.ours, atom, people)
                self.touched[atom_id] = atom

    # -------------------------------------------------------- questions
    def _questions(self) -> None:
        """Their questions, as they stand. An open one is opened here unless
        something here is already open on the same subject of doubt, and is
        put to people here as never asked. A closed one is kept as a record,
        which is what sleep reads a confirmed coreference from."""
        for q in self.theirs.all_questions():
            related = other = None
            if q["related_atom_id"] is not None:
                related = self.atoms.get(q["related_atom_id"])
                if related is None:
                    continue
            if q["other_atom_id"] is not None:
                other = self.atoms.get(q["other_atom_id"])
                if other is None:
                    continue
            rid, oid = (related.id if related else None), (other.id if other else None)
            if q["answered"]:
                if not self.ours.has_question(q["reason"], rid, q["term"], oid):
                    self.ours.add_closed_question(q["question"], q["reason"], rid, q["term"], oid,
                                                  q["created_at"])
                continue
            if self.ours.has_open_question(rid, q["term"], oid):
                continue
            self.ours.queue_question(q["question"], q["reason"], related_atom_id=rid, term=q["term"],
                                     other_atom_id=oid)
            self.report.questions_added += 1

    def _rivalries(self) -> None:
        """Rival values brought together by this ingest -- one that was here
        and one that came over -- get the question of which holds, unless
        one of them already has it open. Rivals that came over together
        bring whatever questions they had."""
        asked = set()
        for atom_id in self.touched:
            atom = self.ours.atom_by_id(atom_id)
            if atom is None or atom.negative or atom.status not in (AtomStatus.CANDIDATE, AtomStatus.CONFIRMED):
                continue
            standing = questions.rivals(self.ours, atom)
            ids = frozenset(a.id for a in standing)
            if len(ids) < 2 or ids in asked or not (ids & self.created.keys()) or ids <= self.created.keys():
                continue
            asked.add(ids)
            if any(q["reason"] == "contradiction" and q["related_atom_id"] in ids
                   for q in self.ours.pending_questions()):
                continue
            incumbent = max(standing, key=lambda a: (a.confidence, -a.id))
            if questions.queue(self.ours, "contradiction", incumbent):
                self.report.questions_added += 1
                self.report.rivalries_queued += 1
                self.report.details.append(
                    f"  rival values for '{incumbent.subject} {incumbent.relation}': "
                    + ", ".join(sorted(f"'{a.object}'" for a in standing)))
