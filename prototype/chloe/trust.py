"""
Trust and confidence.

Trust is domain-specific rather than one global score: each Person carries
a score per domain (models.Person.trust), nudged up on corroboration and
down on contradiction.

Atom confidence is a trust-weighted vote over the provenance attached to
it, not a source count, so one trusted source can outweigh several
unreliable ones and a single trusted contradiction can flip an atom's
status. The vote is damped by the total weight of evidence behind it, so
confidence approaches its extremes only as trusted evidence accumulates,
not on unanimity alone.
"""

from .models import Atom, AtomStatus, Person, Provenance

LEARNING_RATE = 0.15

# Weight of the "no evidence yet" prior in the confidence damping term. Larger
# values need more accumulated trust-weight before a belief can approach 0 or 1.
# 0.5 -- one neutral-trust source -- keeps the demo's contradicted belief below
# CONTRADICT_THRESHOLD; at 1.0 it no longer registers as contradicted at all.
EVIDENCE_PRIOR = 0.5
CONFIRM_THRESHOLD = 0.75
CONTRADICT_THRESHOLD = 0.35


def update_trust_on_corroboration(person: Person, domain: str) -> None:
    current = person.trust_in(domain)
    person.set_trust(current + LEARNING_RATE * (1.0 - current), domain)


def update_trust_on_contradiction(person: Person, domain: str) -> None:
    current = person.trust_in(domain)
    person.set_trust(current - LEARNING_RATE * current, domain)


def undo_contradiction(person: Person, domain: str) -> None:
    """The inverse of update_trust_on_contradiction at the current value, for
    a dispute found afterwards to have been no disagreement at all."""
    person.set_trust(person.trust_in(domain) / (1.0 - LEARNING_RATE), domain)


def recompute_confidence(atom: Atom, people_by_id: dict) -> float:
    """Trust-weighted vote, damped by how much evidence stands behind it.

    Direction comes from the vote: +polarity*trust corroborating,
    -polarity*trust contradicting, normalised by total weight into [-1, 1].
    Magnitude is damped by weight_total / (weight_total + EVIDENCE_PRIOR),
    so the vote decides which way a belief leans and accumulated weight
    decides how far it can go. 0.5 is the no-information point.

    Undamped, a normalised vote reaches its extremes on unanimity alone,
    however little the agreeing sources are trusted.
    """
    if not atom.evidence:
        return atom.confidence

    score = 0.0
    weight_total = 0.0
    for prov in atom.evidence:
        person = people_by_id.get(prov.person_id)
        trust = person.trust_in(atom.domain) if person else 0.5
        score += prov.polarity * trust
        weight_total += trust

    if weight_total == 0:
        return 0.5

    normalized = score / weight_total  # in [-1, 1]
    damping = weight_total / (weight_total + EVIDENCE_PRIOR)
    confidence = 0.5 + 0.5 * normalized * damping  # map to [0, 1]
    return max(0.0, min(1.0, confidence))


def status_from_confidence(confidence: float, n_provenance: int) -> AtomStatus:
    if n_provenance <= 1:
        return AtomStatus.CANDIDATE
    if confidence >= CONFIRM_THRESHOLD:
        return AtomStatus.CONFIRMED
    if confidence <= CONTRADICT_THRESHOLD:
        return AtomStatus.CONTRADICTED
    return AtomStatus.CANDIDATE
