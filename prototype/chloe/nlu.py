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

import os
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
    SMALL_TALK = "small_talk"        # "hello", "how are you?", "thanks"
    UNKNOWN = "unknown"


_SCOPE_SPLIT = re.compile(r"\bwhen\b|\bif\b", re.IGNORECASE)
# Part of the parse() contract rather than of any one implementation: a
# line longer than this is refused unread, by the patterns and by the LLM
# parser alike. A single assertion in the forms CHLOE stores fits well
# inside it, so the length itself is evidence the input is something else.
# It also keeps the LLM from ever being handed a long block of text.
MAX_INPUT_CHARS = int(os.getenv("CHLOE_MAX_INPUT_CHARS", "128"))

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


# The same address at the end: "..., Chloe?". A comma is required, so a
# statement ending in the name ("the robot is Chloe") is left alone.
_TRAILING_VOCATIVE_RE = re.compile(r"\s*,\s*chloe\s*([!.?]*)\s*$", re.IGNORECASE)

# Discourse markers and hedges. The belief is the proposition, not the
# packaging -- but only stripped when something follows, so "I think" alone
# or "no" alone is left to be classified on its own terms.
_LEADING_MARKER_RE = re.compile(
    r"^\s*(?:(?:no|yes|yeah|well|so|actually|honestly|right|ok(?:ay)?|but|and)\s*,\s*|"
    r"(?:i\s+(?:think|believe|reckon|guess)|you\s+know|to\s+be\s+fair|i\s+mean)\s+)+",
    re.IGNORECASE,
)
_TRAILING_MARKER_RE = re.compile(
    r"\s*,\s*(?:actually|really|though|indeed|you\s+know|i\s+think|i\s+believe|to\s+be\s+fair)"
    r"\s*([!.?]*)\s*$",
    re.IGNORECASE,
)


def _strip_vocative(s: str) -> str:
    s = _VOCATIVE_RE.sub("", s, count=1)
    return _TRAILING_VOCATIVE_RE.sub(r"\1", s, count=1)


def _strip_markers(s: str) -> str:
    """Remove discourse packaging from both ends, leaving the claim."""
    stripped = _TRAILING_MARKER_RE.sub(r"\1", s, count=1)
    stripped = _LEADING_MARKER_RE.sub("", stripped, count=1)
    return stripped if stripped.strip() else s


# Phatic openings and closings: not assertions, not failures to understand.
# Deliberately narrow -- anything carrying a claim must fall through to the
# patterns below, so this never swallows evidence.
_SMALL_TALK_RE = re.compile(
    r"^\s*(?:"
    r"h(?:i|ello|ey|iya)|yo|greetings|good\s+(?:morning|afternoon|evening|day)|"
    r"how\s+(?:are|r)\s+(?:you|u)(?:\s+doing)?|how(?:'s| is)\s+it\s+going|"
    r"what'?s\s+up|thank(?:s| you)(?:\s+very\s+much)?|cheers|"
    r"nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you|"
    r"good\s+(?:night|bye)|see\s+you|ok(?:ay)?|sure|cool|nice|great"
    r")\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def _is_small_talk(s: str) -> bool:
    return bool(_SMALL_TALK_RE.match(s))


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
    "sleep, chloe": "sleep",
    "sleep chloe": "sleep",
    "go to sleep": "sleep",
    "go to sleep, chloe": "sleep",
    "stop": "stop",
    "ask": "ask",
    "ask more": "ask",
    "ask more questions": "ask",
    "ask me": "ask",
    "ask me more": "ask",
    "ask me more questions": "ask",
    "ask me something": "ask",
    "ask me anything": "ask",
    "ask me questions": "ask",
    "ask something": "ask",
    "ask away": "ask",
    "any more questions": "ask",
    "more questions": "ask",
    "exit": "exit",
    "quit": "exit",
    "bye": "exit",
    "goodbye": "exit",
    "who do you trust": "trust_report",
    "what do you know": "knowledge_report",
    "show open questions": "open_questions",
}


# A subject names the thing a belief is about. One that opens with a
# question word ("how to know", "why it rains") is a question, not a thing,
# and nothing can be believed or asked about it.
_INTERROGATIVES = {"what", "who", "whom", "whose", "which", "how", "why", "where", "when"}


def is_referring(phrase: Optional[str]) -> bool:
    words = (phrase or "").strip().lower().split()
    return bool(words) and words[0] not in _INTERROGATIVES


def check_subject(utt: "Utterance") -> "Utterance":
    """Part of the parse() contract, applied by both parsers: an utterance
    whose subject names nothing is refused."""
    if utt.type in (UtteranceType.STATEMENT, UtteranceType.NEGATION,
                    UtteranceType.YN_QUESTION, UtteranceType.WH_QUESTION) \
            and not is_referring(utt.subject):
        return Utterance(raw=utt.raw, type=UtteranceType.UNKNOWN,
                         extra={**utt.extra, "reason": "not_parseable"})
    return utt


def parse(text: str) -> Utterance:
    return check_subject(_parse_patterns(text))


def _parse_patterns(text: str) -> Utterance:
    raw = text
    if len(text) > MAX_INPUT_CHARS:
        return Utterance(raw=raw, type=UtteranceType.UNKNOWN,
                         extra={"reason": "too_long"})
    text = _strip_vocative(text)
    t = _strip_punct(text)
    if _is_small_talk(t):
        return Utterance(raw=raw, type=UtteranceType.SMALL_TALK)
    text = _strip_markers(text)
    t = _strip_punct(text)
    low = t.lower()

    if low in COMMANDS:
        return Utterance(raw=raw, type=UtteranceType.COMMAND, extra={"command": COMMANDS[low]})

    is_question = raw.strip().endswith("?")

    # wh- question: "what is X", "who is X", "what does X mean".
    # Matched against the original-case text so the subject is echoed back
    # as the person wrote it ("who is Felix?", not "who is felix?").
    m = re.match(r"^(what|who)\s+(is|are|am|means)\s+(.+)$", t, re.IGNORECASE)
    if m and is_question:
        subject = _strip_punct(m.group(3))
        return Utterance(raw=raw, type=UtteranceType.WH_QUESTION, subject=subject,
                         relation="is", extra={"wh": m.group(1).lower()})

    # yes/no question: "is X Y", "are X Y", optionally with a "when/if" scope clause
    m = re.match(r"^(is|are|am|was|were)\s+(.+)$", t, re.IGNORECASE)
    if m and is_question:
        rest, scope = _split_scope(m.group(2))
        # Naive split, object-last: "the sky blue" -> "the sky" / "blue".
        # A determiner immediately before the final word belongs to the
        # object, not the subject: "Felix a cat" -> "Felix" / "a cat".
        tokens = rest.split()
        if len(tokens) >= 2:
            cut = -2 if len(tokens) >= 3 and tokens[-2].lower() in _DETERMINERS else -1
            subject = " ".join(tokens[:cut])
            obj = " ".join(tokens[cut:])
            return Utterance(raw=raw, type=UtteranceType.YN_QUESTION, subject=subject,
                             relation=m.group(1).lower(), obj=obj, scope=scope)
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
