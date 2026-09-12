"""
Minimal natural-language front end.

Turns a line of text into a structured Utterance, and nothing more. The
2000 code hardcoded its English patterns into the control flow
(cGrammar.hh, cData.hh); they live in one place here.

Deliberately not an LLM, so the prototype runs with no API key and no
network. `parse(text) -> Utterance` is the linguistic input interface:
llm_nlu.py implements the same signature over a language model, and
nothing else in the package changes between the two.
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
_COPULAS = ("is", "are", "am", "was", "were", "means", "means that")
_DETERMINERS = ("a", "an", "the")

# Direct address stripped before classification, e.g. "Hi Chloe," or
# "Chloe,". The statement pattern below looks for a copula anywhere in the
# sentence, so an unstripped vocative is swallowed into the subject.
# A bare "chloe" with no greeting word and no comma or colon after it is
# left alone: more likely a statement about Chloe ("Chloe is a robot")
# than an address.
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
    m = re.match(r"^(what|who)\s+(is|are|am|means)\s+(.+)$", low)
    if m and is_question:
        subject = _strip_punct(m.group(3))
        return Utterance(raw=raw, type=UtteranceType.WH_QUESTION, subject=subject, relation="is")

    # yes/no question: "is X Y", "are X Y", optionally with a "when/if" scope clause
    m = re.match(r"^(is|are|was|were)\s+(.+)$", low)
    if m and is_question:
        rest, scope = _split_scope(m.group(2))
        # Naive split, object-last: "the sky blue" -> "the sky" / "blue".
        # A determiner immediately before the final word belongs to the
        # object, not the subject: "Felix a cat" -> "Felix" / "a cat".
        tokens = rest.split()
        if len(tokens) >= 2:
            cut = -2 if len(tokens) >= 3 and tokens[-2] in _DETERMINERS else -1
            subject = " ".join(tokens[:cut])
            obj = " ".join(tokens[cut:])
            return Utterance(raw=raw, type=UtteranceType.YN_QUESTION, subject=subject, relation=m.group(1), obj=obj, scope=scope)
        return Utterance(raw=raw, type=UtteranceType.UNKNOWN)

    # Any other question ("what colour is the sky?") fits neither template
    # above. The statement pattern below does not test is_question and
    # would match the copula inside the question, storing it as a belief,
    # so refuse rather than guess.
    if is_question:
        return Utterance(raw=raw, type=UtteranceType.UNKNOWN)

    # negated statement: "X is not Y"
    body, scope = _split_scope(t)
    m = re.match(r"^(.+?)\s+(is|are|am|was|were|means)\s+not\s+(.+)$", body, re.IGNORECASE)
    if m:
        return Utterance(
            raw=raw, type=UtteranceType.NEGATION,
            subject=_strip_punct(m.group(1)), relation=m.group(2).lower(),
            obj=_strip_punct(m.group(3)), scope=scope, negated=True,
        )

    # plain statement: "X is Y"
    m = re.match(r"^(.+?)\s+(is|are|am|was|were|means)\s+(.+)$", body, re.IGNORECASE)
    if m:
        return Utterance(
            raw=raw, type=UtteranceType.STATEMENT,
            subject=_strip_punct(m.group(1)), relation=m.group(2).lower(),
            obj=_strip_punct(m.group(3)), scope=scope,
        )

    return Utterance(raw=raw, type=UtteranceType.UNKNOWN)
