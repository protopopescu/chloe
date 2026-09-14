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

    @property
    def has_secret(self) -> bool:
        return bool(self.secret_hash and self.secret_salt)

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

    def statement(self) -> str:
        s = f"{self.subject} {self.relation} {self.object}"
        if self.scope:
            s += f" ({self.scope})"
        return s

    def key(self) -> str:
        """Normalised identity used to find 'the same idea' regardless of
        surface phrasing / scope, for contradiction & duplicate detection."""
        return f"{self.subject.strip().lower()}|{self.relation.strip().lower()}"


@dataclass
class Interaction:
    """One turn of dialogue, logged so an atom can be traced back to the
    exact conversational moment it came from."""

    id: Optional[int]
    person_id: int
    role: str  # "human" or "chloe"
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
