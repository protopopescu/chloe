"""
Implication between beliefs: "Felix is a black cat" entails "Felix is a
cat", "poking a bear is risky and dangerous" entails "poking a bear is
dangerous".

Two such beliefs are not the same belief, so they are never merged into
one: whichever wording survived, it would either credit the general
belief's sources with the detail they never gave, or lose the detail. They
stay separate atoms, linked, and the link does three things:

  - they stop competing: neither is a rival value of the other
    (questions.rivals);
  - evidence flows the way implication runs -- support for the specific
    belief supports the general one, and a denial of the general one
    denies the specific one;
  - the disputes recorded when the second arrived, back when nothing could
    tell a compatible value from a rival, are voided and the trust they
    cost is given back. Voided evidence stays in the record.

A link is made during sleep when the words carry the implication ("a black
cat" -> "a cat"), and otherwise only when a person, asked, says it holds
("risky" -> "dangerous"): the LLM's reading of the words is a conjecture,
and a conjecture waits for somebody to confirm it (consolidation.
_link_implications, questions "implication"). This module is what a link
means, used there and by every claim made afterwards.
"""

from typing import Dict, Optional

from . import trust as trust_mod
from .models import Atom, Provenance


def recompute(store, atom: Atom, people_by_id: Optional[Dict] = None) -> None:
    people_by_id = people_by_id or {p.id: p for p in store.all_people()}
    atom.confidence = trust_mod.recompute_confidence(atom, people_by_id)
    atom.status = trust_mod.status_from_confidence(atom.confidence, len(atom.evidence))
    store.save_atom(atom)


def carry(store, atom: Atom, prov: Provenance) -> None:
    """Carry one piece of direct evidence on `atom` to the atoms linked to
    it: support upward to what it entails, denial downward to what entails
    it. A person already heard on the target with the same verdict is not
    counted a second time."""
    if prov.via_atom_id is not None or prov.void:
        return
    targets = store.entails(atom.id) if prov.polarity > 0 else store.entailed_by(atom.id)
    for target_id in targets:
        target = store.atom_by_id(target_id)
        if target is None or any(p.person_id == prov.person_id and p.polarity == prov.polarity
                                 for p in target.evidence):
            continue
        carried = Provenance(person_id=prov.person_id, interaction_id=prov.interaction_id,
                             polarity=prov.polarity, at=prov.at, via_atom_id=atom.id)
        store.add_provenance(target.id, carried)
        target.provenance.append(carried)
        recompute(store, target)


def link(store, specific: Atom, general: Atom, confirmed_by: Optional[int] = None) -> int:
    """Record that `specific` entails `general`, and bring the evidence on
    both into line with it. `confirmed_by` is the person who said so, if it
    was not read from the words alone. Returns the number of disputes voided."""
    store.add_entailment(specific.id, general.id, holds=True, confirmed_by=confirmed_by)
    voided = 0
    for disputed, via in ((general, specific), (specific, general)):
        for person_id in store.void_disputes(disputed.id, via.id):
            person = store.person_by_id(person_id)
            if person is not None:
                trust_mod.undo_contradiction(person, disputed.domain)
                store.save_person(person)
            voided += 1
    specific, general = store.atom_by_id(specific.id), store.atom_by_id(general.id)
    for prov in list(specific.evidence):
        if prov.polarity > 0:
            carry(store, specific, prov)
    for prov in list(general.evidence):
        if prov.polarity < 0:
            carry(store, general, prov)
    people = {p.id: p for p in store.all_people()}
    for atom_id in (specific.id, general.id):
        recompute(store, store.atom_by_id(atom_id), people)
    return voided


def reject(store, specific: Atom, general: Atom, person_id: int) -> None:
    """Record that a person, asked, said `general` does not follow from
    `specific`. The two go on competing as before, and the pair is not put
    to the LLM again."""
    store.add_entailment(specific.id, general.id, holds=False, confirmed_by=person_id)
