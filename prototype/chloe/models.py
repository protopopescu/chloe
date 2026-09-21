"""
Core data model.

Everything CHLOE believes is stored as an Atom: a subject-relation-object
triple, plus the context that makes it meaningful (scope), who said it and
when (provenance), and how confident CHLOE is that it is true.

Knowledge is never an opaque number: every belief is inspectable, and
facts carry explicit scope rather than pretending to be universal.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from . import grammar


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Stance(str, Enum):
    """What CHLOE's reply does with a yes/no question -- decided by the
    engine, and passed to the output-side guard so a naturalisation cannot
    quietly answer the other way. AFFIRM and DENY are the two answers;
    UNKNOWN is a reply that gives neither, and must not come back as one.
    """

    AFFIRM = "affirm"
    DENY = "deny"
    UNKNOWN = "unknown"


class AtomStatus(str, Enum):
    HYPOTHESIS = "hypothesis"      # generated during sleep, not yet from a person
    CANDIDATE = "candidate"        # stated once, not yet corroborated
    CONFIRMED = "confirmed"        # corroborated by enough trusted sources
    CONTRADICTED = "contradicted"  # conflicting evidence exists, unresolved


@dataclass
class Person:
    """An interlocutor. Trust is domain-specific (domain -> score) rather
    than one global reliability number.

    secret_hash/secret_salt back the optional secret-word identity check.
    Without it, anyone typing a known name would inherit that person's
    accumulated trust and belief history."""

    id: int
    name: str
    trust: dict = field(default_factory=dict)  # domain -> float in [0, 1]
    created_at: str = field(default_factory=now_iso)
    secret_hash: Optional[str] = None
    secret_salt: Optional[str] = None

    DEFAULT_DOMAIN = "general"
    DEFAULT_TRUST = 0.5
    # The identity given to someone who claimed a protected name and could
    # not confirm it. Their testimony counts; their claims about themselves
    # do not, since which self they are is the thing not established.
    UNVERIFIED_SUFFIX = " (unverified)"

    @property
    def has_secret(self) -> bool:
        return bool(self.secret_hash and self.secret_salt)

    @property
    def is_unverified(self) -> bool:
        return self.name.endswith(self.UNVERIFIED_SUFFIX)

    @property
    def claimed_name(self) -> str:
        """The name without the mark, for someone whose claim to it is not
        settled. Their own name for anyone else."""
        return self.name[:-len(self.UNVERIFIED_SUFFIX)] if self.is_unverified else self.name

    def claims(self, subject: str) -> bool:
        """Whether a subject is about this person, under the name they hold
        or the one they gave. Which of the two they are is what an unverified
        identity leaves open, so both are claims about themselves."""
        return any(grammar.refers_to(subject, name)
                   for name in {self.name, self.claimed_name})

    def trust_in(self, domain: Optional[str] = None) -> float:
        domain = domain or self.DEFAULT_DOMAIN
        return self.trust.get(domain, self.trust.get(self.DEFAULT_DOMAIN, self.DEFAULT_TRUST))

    def set_trust(self, value: float, domain: Optional[str] = None) -> None:
        domain = domain or self.DEFAULT_DOMAIN
        self.trust[domain] = max(0.0, min(1.0, value))


@dataclass
class Provenance:
    """A single piece of evidence for or against an atom: who said it, in
    which interaction, whether it corroborated or contradicted, and when.

    parse_confidence is how sure the input parser was of its reading (set
    by llm_nlu.py; None for the deterministic pattern parser). It is kept
    out of the belief-confidence maths on purpose: "CHLOE misunderstood
    you" and "your source was wrong" must stay distinguishable."""

    person_id: int
    interaction_id: int
    polarity: int  # +1 corroborates, -1 contradicts
    at: str = field(default_factory=now_iso)
    parse_confidence: Optional[float] = None
    # Evidence this person gave about another atom, carried here: a
    # disputed rival value, or support or denial that an implication
    # between the two carries across (entailment.py). None for evidence
    # given about this atom directly.
    via_atom_id: Optional[int] = None
    # Kept and ignored, never deleted: evidence later found to rest on a
    # mistake -- a dispute recorded against a value that turned out to be
    # compatible -- stays in the record and out of the confidence.
    void: bool = False


@dataclass
class Atom:
    """A single idea, e.g. 'the sky is blue'.

    subject / relation / object generalise the original 'X is Y' pattern:
    the relation is not always "is".

    scope carries the condition under which the statement holds -- "the sky
    is blue" is only true relative to illumination, atmosphere, observer.
    Free text here rather than a formalised condition language.
    """

    id: Optional[int]
    subject: str
    relation: str
    object: str
    scope: str = ""  # e.g. "during daytime, on Earth"
    domain: str = Person.DEFAULT_DOMAIN
    status: AtomStatus = AtomStatus.CANDIDATE
    confidence: float = 0.5
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    provenance: list = field(default_factory=list)  # list[Provenance]

    @property
    def evidence(self) -> list:
        """The provenance that counts: everything not voided."""
        return [p for p in self.provenance if not p.void]

    def statement(self) -> str:
        s = f"{self.subject} {self.relation} {self.object}"
        if self.scope:
            s += f" ({self.scope})"
        return s

    def identity(self) -> tuple:
        """Which belief this is, whatever its surface form (grammar.identity)."""
        return grammar.identity(self.subject, self.relation, self.object, self.scope)

    @property
    def negative(self) -> bool:
        """'X is not Y', stored as a belief in its own right."""
        return grammar.norm_phrase(self.object).startswith("not ")


@dataclass
class Interaction:
    """One turn of dialogue, logged so an atom can be traced back to the
    exact conversational moment it came from."""

    id: Optional[int]
    person_id: int
    role: str  # "source" or "chloe"
    text: str
    at: str = field(default_factory=now_iso)


@dataclass
class RelationProperties:
    """Declared algebraic properties of a relation.

    Transitivity and symmetry are never assumed just because English
    suggests them. They are declared explicitly, per relation, and default
    to False.
    """

    name: str
    transitive: bool = False
    symmetric: bool = False
