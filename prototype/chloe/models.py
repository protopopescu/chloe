"""
Core data model.

Everything CHLOE believes is stored as an Atom: a subject-relation-object
triple, plus the context that makes it meaningful (scope), who said it and
when (provenance), and how confident CHLOE is that it's true.

This directly implements the "Knowledge Representation" and "Central
Philosophy" sections of the design notes: knowledge is never an opaque
number, every belief is inspectable, and facts carry explicit scope rather
than pretending to be universal.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AtomStatus(str, Enum):
    HYPOTHESIS = "hypothesis"      # generated during sleep, not yet from a person
    CANDIDATE = "candidate"        # stated once, not yet corroborated
    CONFIRMED = "confirmed"        # corroborated by enough trusted sources
    CONTRADICTED = "contradicted"  # conflicting evidence exists, unresolved


@dataclass
class Person:
    """An interlocutor. Trust is domain-specific (a dict of domain -> score)
    rather than a single global reliability number, per the 'Trust Models'
    section of CHLOE.md.

    secret_hash/secret_salt back the optional secret-word identity check:
    someone who opts in on first meeting can later be asked to confirm the
    word before CHLOE re-binds their name back to this Person's history and
    trust score -- otherwise anyone who types a known name would silently
    inherit that person's accumulated trust and belief history."""

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
    """A single piece of evidence for or against an atom: who said it,
    in which interaction, whether it corroborated or contradicted, and
    when.

    parse_confidence records how sure the input parser was that its
    structured reading faithfully represents what the person said (set by
    the LLM parser, llm_nlu.py; None for the deterministic pattern
    parser, which reports no such number). It is deliberately kept
    separate from source trust and is NOT folded into belief confidence:
    "CHLOE misunderstood you" and "your source was wrong" are different
    failures and must stay distinguishable in the record."""

    person_id: int
    interaction_id: int
    polarity: int  # +1 corroborates, -1 contradicts
    at: str = field(default_factory=now_iso)
    parse_confidence: Optional[float] = None


@dataclass
class Atom:
    """A single 'idea' / 'atom of reasoning', e.g. 'the sky is blue'.

    subject / relation / object mirror CHLOE's original 'X is Y' pattern
    but generalise it (relation is not always "is").

    scope holds the contextual qualifiers CHLOE.md calls out explicitly:
    a statement like "the sky is blue" is only meaningful relative to
    atmosphere / illumination / observer / etc. We store scope as free-text
    key:value-ish conditions rather than trying to formalise them fully in
    this prototype.
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
    """One turn of dialogue, logged for provenance and replay -- CHLOE.md's
    'every belief should remain inspectable' extends to being able to trace
    an atom back to the exact conversational moment it came from."""

    id: Optional[int]
    person_id: int
    role: str  # "human" or "chloe"
    text: str
    at: str = field(default_factory=now_iso)


@dataclass
class RelationProperties:
    """Declared algebraic properties of a relation.

    CHLOE.md's 'Reasoning' principle: inference engines must never silently
    assume transitivity/symmetry just because natural language suggests it.
    Those properties must be declared explicitly, here, per relation -- and
    default to False.
    """

    name: str
    transitive: bool = False
    symmetric: bool = False
