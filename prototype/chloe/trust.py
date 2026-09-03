"""
Trust and confidence.

From the design notes, 'Trust Models': trust evolves with internal consistency,
corroboration, later verification, and historical reliability, and should
be domain-specific rather than one global score. This module implements a
simple version of that: each Person carries a trust score per domain
(models.Person.trust), nudged up on corroboration and down on
contradiction. Atom confidence is then a trust-weighted vote across all
the provenance attached to it, not a raw source count -- so one highly
trusted source can outweigh several unreliable ones, and a single
contradiction from a trusted source can flip an atom's status.
"""

from .models import Atom, AtomStatus, Person, Provenance

LEARNING_RATE = 0.15
CONFIRM_THRESHOLD = 0.75
CONTRADICT_THRESHOLD = 0.35


def update_trust_on_corroboration(person: Person, domain: str) -> None:
    current = person.trust_in(domain)
    person.set_trust(current + LEARNING_RATE * (1.0 - current), domain)


def update_trust_on_contradiction(person: Person, domain: str) -> None:
    current = person.trust_in(domain)
    person.set_trust(current - LEARNING_RATE * current, domain)


def recompute_confidence(atom: Atom, people_by_id: dict) -> float:
    """Trust-weighted vote: +polarity*trust for corroborating provenance,
    -polarity*trust for contradicting. Squashed into [0, 1] around 0.5."""
    if not atom.provenance:
        return atom.confidence

    score = 0.0
    weight_total = 0.0
    for prov in atom.provenance:
        person = people_by_id.get(prov.person_id)
        trust = person.trust_in(atom.domain) if person else 0.5
        score += prov.polarity * trust
        weight_total += trust

    if weight_total == 0:
        return 0.5

    normalized = score / weight_total  # in [-1, 1]
    confidence = 0.5 + 0.5 * normalized  # map to [0, 1]
    return max(0.0, min(1.0, confidence))


def status_from_confidence(confidence: float, n_provenance: int) -> AtomStatus:
    if n_provenance <= 1:
        return AtomStatus.CANDIDATE
    if confidence >= CONFIRM_THRESHOLD:
        return AtomStatus.CONFIRMED
    if confidence <= CONTRADICT_THRESHOLD:
        return AtomStatus.CONTRADICTED
    return AtomStatus.CANDIDATE
