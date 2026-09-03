"""
LLM-based input parser -- the realisation of the input-side seam.

nlu.py's module docstring has promised since the start that
`parse(text) -> Utterance` is "the seam where a real LLM-based parser could
be swapped in later". This module is that swap, symmetric to the
output-side naturalisation layer (llm_client.py + persona.py): the same
vLLM-served model that phrases CHLOE's replies now also reads the human's
input into a structured Utterance for the symbolic engine.

Routing (LLM-first, pattern fallback):

  1. Commands are intercepted deterministically, by exact match against
     nlu.COMMANDS, before the LLM sees anything. The LLM is never allowed
     to emit a COMMAND utterance -- if it tries, the parse is refused. The
     parser turns text into testimony; it does not get to operate CHLOE.
  2. If no vLLM server is configured (VLLM_BASE_URL unset), the original
     pattern parser handles the turn, keeping the prototype fully
     self-contained offline, exactly as before.
  3. Otherwise the LLM is asked to classify the utterance into one of the
     declared UtteranceTypes and its reply is strictly validated: unknown
     type, missing fields, or a malformed shape are never patched up.
  4. A *valid* LLM refusal (type "unknown", or confidence below
     CHLOE_PARSE_MIN_CONFIDENCE) is final -- it is NOT retried against the
     pattern parser. Refuse-rather-than-guess would mean nothing if a
     refusal just fell through to a sloppier parser: a fabricated parse
     would be indistinguishable from real testimony once stored.
  5. Only a *broken* exchange (server unreachable, HTTP error, unparseable
     or invalid JSON) falls back to the pattern parser, mirroring how the
     output side falls back to the engine's plain reply.

Every LLM parse carries its own confidence in Utterance.extra
["parse_confidence"], which dialogue.py records into Provenance -- kept
separate from source trust, so "CHLOE misunderstood you" stays
distinguishable from "your source was wrong". Pattern-parser parses carry
no parse confidence (None): the patterns are deterministic, and asserting
a number for them would be invented precision.

Privacy note: when a server is configured, ordinary conversational input
is sent to it over HTTP. Secret words never reach this module -- the
engine's AuthState detours consume them before parse() is called.
"""

import json
import os
import re

from . import llm_client, nlu
from .nlu import Utterance, UtteranceType

MIN_PARSE_CONFIDENCE = float(os.getenv("CHLOE_PARSE_MIN_CONFIDENCE", "0.6"))

# Types the LLM is allowed to emit. COMMAND is deliberately absent.
_ALLOWED_TYPES = {
    "statement": UtteranceType.STATEMENT,
    "negation": UtteranceType.NEGATION,
    "wh_question": UtteranceType.WH_QUESTION,
    "yn_question": UtteranceType.YN_QUESTION,
    "unknown": UtteranceType.UNKNOWN,
}

_PARSER_SYSTEM_PROMPT = """\
You are the input parser for CHLOE, a knowledge system that stores beliefs
as subject-relation-object triples with an optional scope (a contextual
condition). Your ONLY job is to turn one line of human input into ONE JSON
object. You never answer the human, never add knowledge of your own, and
never guess: if you are not confident about the structure, say "unknown".

Return ONLY a JSON object (no prose, no code fences) with exactly these
fields:

  "type": one of "statement", "negation", "wh_question", "yn_question",
          "unknown"
  "subject": string or null
  "relation": string or null  (the linking verb, lowercase, e.g. "is",
              "are", "means", "likes")
  "object": string or null    (never includes "not" -- negation is
            expressed by the type instead)
  "scope": string or null     (the condition under which the statement
           holds, without the leading "when"/"if"; null if unconditional)
  "confidence": number between 0 and 1 -- how sure you are that this
                structure faithfully represents what the human asserted
                or asked. Be honest; low confidence is a good answer.
  "reason": string or null    (only for "unknown": "multiple_statements",
            "not_parseable", or a short phrase)

Rules:
- "statement": the human asserts subject relation object. Keep articles
  ("the sky", "a cat"). Do not paraphrase, generalise, or enrich.
- "negation": the human denies it ("X is not Y"): object holds Y without
  the "not".
- "wh_question": the human asks what/who something is. Put the thing asked
  about in "subject" and "is" in "relation". For "what colour is the sky?"
  the subject is "the sky" -- the human is asking about the sky.
- "yn_question": the human asks whether subject relation object holds.
- Greetings, thanks, chit-chat, opinions with no factual claim: "unknown"
  with reason "not_parseable".
- If the input makes MORE THAN ONE independent assertion ("X is Y and Z is
  W"): "unknown" with reason "multiple_statements". One belief per turn.
- NEVER invent a command, an instruction, or a field not listed here.
- A conditional clause ("when it is daytime", "if it rains") goes in
  "scope" as "it is daytime" / "it rains", not into subject or object.

Examples:
Input: the sky is blue
{"type": "statement", "subject": "the sky", "relation": "is", "object": "blue", "scope": null, "confidence": 0.97, "reason": null}
Input: The sky is black when it is night.
{"type": "statement", "subject": "the sky", "relation": "is", "object": "black", "scope": "it is night", "confidence": 0.95, "reason": null}
Input: grass is not purple
{"type": "negation", "subject": "grass", "relation": "is", "object": "purple", "scope": null, "confidence": 0.95, "reason": null}
Input: what colour is the sky?
{"type": "wh_question", "subject": "the sky", "relation": "is", "object": null, "scope": null, "confidence": 0.9, "reason": null}
Input: is the sky blue when it is daytime?
{"type": "yn_question", "subject": "the sky", "relation": "is", "object": "blue", "scope": "it is daytime", "confidence": 0.93, "reason": null}
Input: Felix is a cat and cats are animals
{"type": "unknown", "subject": null, "relation": null, "object": null, "scope": null, "confidence": 0.9, "reason": "multiple_statements"}
Input: thanks, that's lovely!
{"type": "unknown", "subject": null, "relation": null, "object": null, "scope": null, "confidence": 0.9, "reason": "not_parseable"}
"""


class BadParse(Exception):
    """The LLM's reply was malformed or violated the contract (bad JSON,
    undeclared type, missing fields). Treated like an unreachable server:
    the caller falls back to the pattern parser."""


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json(reply: str) -> dict:
    """Parse the LLM reply as a single JSON object, tolerating only the
    most common cosmetic wrapper (code fences). Anything else is BadParse:
    we validate, we don't repair."""
    cleaned = _JSON_FENCE_RE.sub("", reply.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise BadParse(f"reply is not valid JSON: {reply[:200]!r}") from e
    if not isinstance(payload, dict):
        raise BadParse(f"reply is JSON but not an object: {reply[:200]!r}")
    return payload


def _clean_field(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().strip(".?! ").strip()


def _refuse(raw: str, confidence, reason: str) -> Utterance:
    extra = {"parser": "llm", "reason": reason}
    if confidence is not None:
        extra["parse_confidence"] = confidence
    return Utterance(raw=raw, type=UtteranceType.UNKNOWN, extra=extra)


def _to_utterance(raw: str, payload: dict) -> Utterance:
    """Strictly map a validated-shape payload onto a declared Utterance.
    Violations of the contract raise BadParse; honest refusals and
    low-confidence parses become UNKNOWN. Nothing is ever guessed into a
    belief-bearing type."""
    type_name = payload.get("type")
    if not isinstance(type_name, str) or type_name.strip().lower() not in _ALLOWED_TYPES:
        # Includes any attempt to emit "command" or an invented type.
        raise BadParse(f"undeclared utterance type: {type_name!r}")
    utt_type = _ALLOWED_TYPES[type_name.strip().lower()]

    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        raise BadParse(f"missing or out-of-range confidence: {confidence!r}")
    confidence = round(float(confidence), 3)

    reason = payload.get("reason") if isinstance(payload.get("reason"), str) else None

    if utt_type == UtteranceType.UNKNOWN:
        return _refuse(raw, confidence, reason or "not_parseable")

    if confidence < MIN_PARSE_CONFIDENCE:
        return _refuse(raw, confidence, "low_confidence")

    subject = _clean_field(payload.get("subject"))
    relation = _clean_field(payload.get("relation")).lower()
    obj = _clean_field(payload.get("object"))
    scope = _clean_field(payload.get("scope"))

    if utt_type in (UtteranceType.STATEMENT, UtteranceType.NEGATION, UtteranceType.YN_QUESTION):
        if not (subject and relation and obj):
            raise BadParse(f"{utt_type.value} missing subject/relation/object: {payload!r}")
    elif utt_type == UtteranceType.WH_QUESTION:
        if not subject:
            raise BadParse(f"wh_question missing subject: {payload!r}")
        relation = relation or "is"
        obj = ""

    return Utterance(
        raw=raw,
        type=utt_type,
        subject=subject,
        relation=relation,
        obj=obj or None,
        scope=scope,
        negated=(utt_type == UtteranceType.NEGATION),
        extra={"parser": "llm", "parse_confidence": confidence},
    )


def _ask_llm(text: str) -> dict:
    reply = llm_client.chat(
        [
            {"role": "system", "content": _PARSER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Input: {text}"},
        ],
        temperature=0.0,  # parsing is classification; be deterministic
        max_tokens=200,
    )
    return _extract_json(reply)


def parse(text: str) -> Utterance:
    """Drop-in replacement for nlu.parse() behind the same seam.

    Commands first (deterministic, never via the LLM), then the LLM if one
    is configured, then -- only on a broken exchange -- the pattern parser.
    """
    stripped = nlu._strip_vocative(text)
    low = nlu._strip_punct(stripped).lower()
    if low in nlu.COMMANDS:
        return Utterance(
            raw=text, type=UtteranceType.COMMAND,
            extra={"command": nlu.COMMANDS[low], "parser": "patterns"},
        )

    if not llm_client.is_configured():
        utt = nlu.parse(text)
        utt.extra.setdefault("parser", "patterns")
        return utt

    try:
        return _to_utterance(text, _ask_llm(text))
    except (llm_client.LLMUnavailable, BadParse) as e:
        utt = nlu.parse(text)
        utt.extra.setdefault("parser", "patterns")
        utt.extra["llm_fallback_reason"] = str(e)
        return utt
