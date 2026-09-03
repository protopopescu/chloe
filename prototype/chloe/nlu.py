"""
Minimal natural-language front end.

The 2000 code hardcoded a handful of English patterns directly into the
control flow (cGrammar.hh, cData.hh). We do the same thing here, in spirit,
but a bit more robustly and in one place -- this module's whole job is to
turn a line of text into a structured Utterance, nothing more. It is
deliberately NOT an LLM: this keeps the prototype self-contained and
runnable with no API key or network access. The interface here
(`parse(text) -> Utterance`) is the seam where a real LLM-based parser
could be swapped in later, exactly as CHLOE.md's "Possible Modern
Architecture" describes ("LLM for language understanding" sitting on top
of the symbolic core). Nothing else in the package needs to change if you
do that swap.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class UtteranceType(str, Enum):
    STATEMENT = "statement"          # "X is Y"
    NEGATION = "negation"            # "X is not Y"
    WH_QUESTION = "wh_question"      # "what is X"
    YN_QUESTION = "yn_question"      # "is X Y"
    COMMAND = "command"              # "sleep", "exit", "who do you trust", ...
    UNKNOWN = "unknown"


_SCOPE_SPLIT = re.compile(r"\bwhen\b|\bif\b", re.IGNORECASE)
_COPULAS = ("is", "are", "was", "were", "means", "means that")

# Direct-address greeting stripped before classification, e.g. "Hi Chloe,"
# or "Chloe,". The 2000 C++ parser (cGrammar.hh AnalyseSentence) never
# actually stripped this -- it required tword[1] to be a copula, so
# "Hi Chloe, ..." simply matched nothing and was silently dropped. Our
# regex parser is more permissive (it hunts for "is" anywhere in the
# sentence), so an unstripped vocative gets swallowed into the subject.
# A bare "chloe" with no greeting word and no following comma/colon is
# left alone, since that's more likely a real statement about Chloe
# herself (e.g. "Chloe is a robot") than an address.
_VOCATIVE_RE = re.compile(
    r"^\s*(?:(?:hi|hey|hello|hiya|yo)\b\s*,?\s*chloe\b\s*[,:]?\s*|chloe\b\s*[,:]\s*)",
    re.IGNORECASE,
)


def _strip_vocative(s: str) -> str:
    return _VOCATIVE_RE.sub("", s, count=1)


@dataclass
class Utterance:
    raw: str
    type: UtteranceType
    subject: Optional[str] = None
    relation: Optional[str] = None
    obj: Optional[str] = None
    scope: str = ""
    negated: bool = False
    extra: dict = field(default_factory=dict)


def _strip_punct(s: str) -> str:
    return s.strip().rstrip(".?! ").strip()


def _split_scope(s: str):
    parts = _SCOPE_SPLIT.split(s, maxsplit=1)
    if len(parts) == 2:
        return _strip_punct(parts[0]), _strip_punct(parts[1])
    return _strip_punct(s), ""


COMMANDS = {
    "sleep": "sleep",
    "exit": "exit",
    "quit": "exit",
    "bye": "exit",
    "goodbye": "exit",
    "who do you trust": "trust_report",
    "what do you know": "knowledge_report",
    "show open questions": "open_questions",
}


def parse(text: str) -> Utterance:
    raw = text
    text = _strip_vocative(text)
    t = _strip_punct(text)
    low = t.lower()

    if low in COMMANDS:
        return Utterance(raw=raw, type=UtteranceType.COMMAND, extra={"command": COMMANDS[low]})

    is_question = raw.strip().endswith("?")

    # wh- question: "what is X", "who is X", "what does X mean"
    m = re.match(r"^(what|who)\s+(is|are|means)\s+(.+)$", low)
    if m and is_question:
        subject = _strip_punct(m.group(3))
        return Utterance(raw=raw, type=UtteranceType.WH_QUESTION, subject=subject, relation="is")

    # yes/no question: "is X Y", "are X Y", optionally with a "when/if" scope clause
    m = re.match(r"^(is|are|was|were)\s+(.+)$", low)
    if m and is_question:
        rest, scope = _split_scope(m.group(2))
        # naive split: "the sky blue" -> subject="the sky", object="blue"
        tokens = rest.split()
        if len(tokens) >= 2:
            subject = " ".join(tokens[:-1])
            obj = tokens[-1]
            return Utterance(raw=raw, type=UtteranceType.YN_QUESTION, subject=subject, relation=m.group(1), obj=obj, scope=scope)
        return Utterance(raw=raw, type=UtteranceType.UNKNOWN)

    # Any other question ("what colour is the sky?") doesn't fit the narrow
    # wh/yn templates above. Previously this fell through to the statement
    # regex below, which doesn't check is_question and will happily match
    # the "is" inside the question -- turning a question into a false
    # remembered belief. Bail out to UNKNOWN instead of guessing.
    if is_question:
        return Utterance(raw=raw, type=UtteranceType.UNKNOWN)

    # negated statement: "X is not Y"
    body, scope = _split_scope(t)
    m = re.match(r"^(.+?)\s+(is|are|was|were|means)\s+not\s+(.+)$", body, re.IGNORECASE)
    if m:
        return Utterance(
            raw=raw, type=UtteranceType.NEGATION,
            subject=_strip_punct(m.group(1)), relation=m.group(2).lower(),
            obj=_strip_punct(m.group(3)), scope=scope, negated=True,
        )

    # plain statement: "X is Y"
    m = re.match(r"^(.+?)\s+(is|are|was|were|means)\s+(.+)$", body, re.IGNORECASE)
    if m:
        return Utterance(
            raw=raw, type=UtteranceType.STATEMENT,
            subject=_strip_punct(m.group(1)), relation=m.group(2).lower(),
            obj=_strip_punct(m.group(3)), scope=scope,
        )

    return Utterance(raw=raw, type=UtteranceType.UNKNOWN)
