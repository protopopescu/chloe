"""
The output-side linguistic interface: prompt construction, and the contract
its replies must satisfy.

The symbolic engine (dialogue.ChloeEngine) decides what happened -- a fact
learned, corroborated or contradicted, a question pending, a trust score
changed -- and produces a short factual reply. The LLM only phrases it.

A naturalisation is checked before it is shown. One that repeats the
prompt's scaffolding, echoes the person's words back at them, runs far
longer than the reply it was given, swaps the speakers, introduces a name or
number the core never supplied, leaves out a name it did, or answers a
yes/no question the other way, is rejected in favour of the engine's own
text -- the same fallback taken when the server is unreachable. A claim
phrased into a reply would be attributed to nobody and could never be
doubted later.

Some of what the core writes is not phrased at all. A question put to the
person is what their next answer will be recorded against, and a list of
beliefs has one item per belief, which the checks do not count. Both reach
the person in the core's words (split_verbatim).
"""

import re

CHLOE_NAME = "Chloe"

CHLOE_DESCRIPTION = (
    "You are CHLOE (Computer-Human Language Oriented Experiment), an AI "
    "project started in 2000. You learn concepts and facts from "
    "conversation, track who told you what and how much you trust them, "
    "and flag contradictions instead of silently picking a side. You do "
    "not pretend to know things you weren't told."
)

STYLE_GUIDE = (
    "Speak in one or two short, natural sentences. Warm but plain -- no "
    "corporate assistant tone, no excessive enthusiasm, no emoji. Never "
    "invent facts, numbers or claims of your own, and never correct the "
    "person from your own knowledge.\n"
    "Write only the sentence you would say out loud. Keep every name and "
    "number exactly as given, and keep who is who: what is said about the "
    "person stays about the person. Do not repeat the person's message "
    "back to them, do not restate or quote these instructions, and never "
    "write a label, heading, prefix or anything in brackets before your "
    "reply."
)

# A sentence that has already been decided needs no sampling diversity.
NATURALISE_TEMPERATURE = 0.0

# A backstop. The defence is the request's structure: no label in the user
# turn, nothing in the prompt that would read as instructions if echoed. A
# small model can still produce these phrases unprompted. New leakage calls
# for tightening the structure, and this list should not grow.
SCAFFOLD_MARKERS = (
    "ground truth",
    "the person just said",
    "reply to the person",
    "do not contradict",
    "in your own words",
    "as an ai",
    "system prompt",
)

MIN_REPLY_CHARS = 1
# A short message is a poor echo test: "Noted..." begins with "no". Only
# treat a repetition as an echo when the message is long enough for it to
# be deliberate.
MIN_ECHO_CHARS = 12
LENGTH_SLACK = 4        # a naturalisation may elaborate, but not by this much
LENGTH_FLOOR = 240      # ... and short replies are never rejected for length

# Words, contractions and numbers. A capital is only evidence of a name
# when it falls inside a sentence -- at the start it is just orthography.
_TOKEN_RE = re.compile(r"[A-Za-z][\w'’-]*|\d[\d.,:%/-]*")
_SENTENCE_BREAK = ".!?:;\n"
_CORE_SENTENCE_BREAK = ".!?;\n"
# First person and Chloe's own name may appear anywhere.
_ALWAYS_PERMITTED = {"i", "i'm", "i've", "i'll", "i'd", CHLOE_NAME.lower()}

# How a reply opens, when it opens by answering. The core knows whether it
# said yes, no or neither (models.Stance); these say what the naturalisation
# did, so the two can be compared. Only the opening is examined, since an
# answer to a yes/no question leads with it.
_AFFIRM_OPENER_RE = re.compile(
    r"^\W*(?:yes|yeah|yep|yup|correct|true|indeed|right|that(?:'|\u2019)?s right|absolutely|of course)\b",
    re.IGNORECASE)
_DENY_OPENER_RE = re.compile(
    r"^\W*(?:no|nope|nah|not quite|i don(?:'|\u2019)?t think so|i do not think so|actually,?\s+no)\b",
    re.IGNORECASE)


def _opening_stance(text: str):
    """Return "affirm", "deny", or None if the reply does not open with
    an answer at all."""
    if _AFFIRM_OPENER_RE.match(text):
        return "affirm"
    if _DENY_OPENER_RE.match(text):
        return "deny"
    return None


# Negation particles, for comparing a denial with the atom it denies.
_NEGATION_RE = re.compile(r"\b(?:not|never)\b|n't\b", re.IGNORECASE)


def _denegate(text: str) -> str:
    """`text` with negation removed and whitespace collapsed.

    A denial and the atom it denies are the same proposition in opposite
    polarity, so this is what lets the echo test tell "repeating the
    person" apart from "naming the belief under discussion".
    """
    return " ".join(_NEGATION_RE.sub(" ", text.lower()).split())


def _words(text: str) -> list:
    return re.findall(r"[\w'’-]+", text.lower())


def _is_negation(word: str) -> bool:
    return word in ("not", "never") or word.endswith("n't")


def _polarities(text: str, atom_words: list) -> set:
    """Polarities at which `atom_words` occurs in `text`.

    True means the occurrence was negated. Negation words inside a run are
    skipped over and noted, so 'Claire is an artist' and 'Claire is not an
    artist' are the same proposition found at opposite polarities rather
    than two unrelated strings.
    """
    if not atom_words:
        return set()
    words = _words(text)
    found = set()
    for start in range(len(words)):
        i, j, negated = start, 0, False
        while i < len(words) and j < len(atom_words):
            if _is_negation(words[i]):
                negated = True
                i += 1
                continue
            if words[i] != atom_words[j]:
                break
            i += 1
            j += 1
        if j == len(atom_words):
            found.add(negated)
    return found


# "you are X" / "I am X" -- the same predicate attached to the other party.
_YOU_ARE_RE = re.compile(r"\byou(?:'re|’re| are)\s+(?:an?|the)?\s*([\w'’-]+)", re.IGNORECASE)
_I_AM_RE = re.compile(r"\bi(?:'m|’m| am)\s+(?:an?|the)?\s*([\w'’-]+)", re.IGNORECASE)


def system_prompt(ground_truth_reply: str = "") -> str:
    """The persona, plus -- for a given turn -- what CHLOE has to say.

    The per-turn content goes in the system role deliberately. Carried in
    the user turn behind a label, a small model tends to reproduce the
    label along with the content.
    """
    parts = [CHLOE_DESCRIPTION, STYLE_GUIDE]
    if ground_truth_reply:
        parts.append(
            "Say this to the person now, adding nothing and leaving nothing "
            f"out:\n{ground_truth_reply}"
        )
    return "\n\n".join(parts)


def naturalise_request(user_message: str, ground_truth_reply: str) -> list:
    """Messages for asking the model to phrase a reply the engine has
    already decided.

    Deliberately three turns and no conversation history. Shown its own
    previous naturalisations as assistant turns, a small model continues
    that thread in preference to the instruction, and the reply ends up
    saying something the core never decided. The user turn carries only what
    the person said, so there is no scaffolding in it to echo.
    """
    return [
        {"role": "system", "content": system_prompt(ground_truth_reply)},
        {"role": "user", "content": user_message},
    ]


def _referents(text: str, strict: bool = True, breaks: str = _SENTENCE_BREAK) -> set:
    """Names and numbers in `text`, as a set of lowercased tokens.

    `strict` asks which tokens *count as* a name, and is used on the reply:
    a sentence-initial capital is ordinary orthography and carries no
    evidence of one. On the sources -- the core's reply and the person's
    message -- the same set is an allowlist, so it is gathered permissively
    and every capital counts. Over-collecting there only ever permits more.
    """
    found = set()
    for m in _TOKEN_RE.finditer(text):
        token = m.group(0)
        if token[0].isdigit():
            found.add(token.rstrip(".,:;").lower())
            continue
        if not token[0].isupper():
            continue
        if strict:
            back = text[: m.start()].rstrip(" \t\"'([‘“")
            if not back or back[-1] in breaks:
                continue    # start of a sentence: orthography, not a name
        found.add(token.lower())
    return found


def _predicate(pattern, text: str) -> set:
    return {m.group(1).lower() for m in pattern.finditer(text)}


def rejection_reason(reply: str, user_message: str, ground_truth_reply: str,
                     stance=None):
    """Why this naturalisation must not be shown, or None if it may be.

    Conservative by design: a false rejection costs only the engine's plainer
    wording, while a false acceptance puts text in front of a visitor that
    the epistemic core never decided and cannot account for.

    `stance` is the core's own answer to a yes/no question -- a models.Stance
    value, or None where the turn was not one. It is compared with the
    stance the reply opens with, a case the polarity check cannot see: to
    "Is Hector an animal?" the core may answer "I don't think so -- I
    believe Hector is a dog", which never contains the person's words.
    """
    text = (reply or "").strip()
    if len(text) < MIN_REPLY_CHARS:
        return "empty reply"

    low = text.lower()
    for marker in SCAFFOLD_MARKERS:
        if marker in low:
            return f"leaked instruction text ({marker!r})"

    # The person's words may appear in the reply only where the engine put
    # them there. Anywhere else they are the model quoting the conversation
    # back -- as an opening restatement, or inside a narration of the turn
    # ("The conversation was: ...") -- which is scaffolding by another route.
    echo = user_message.strip().rstrip(".!?").lower()
    # A denial is the exception: "Claire is not an artist" against a core
    # reply ending "I'll double check: Claire is an artist" names the same
    # atom. It is allowed only where the core itself named the proposition.
    core_named_it = (echo in ground_truth_reply.lower()
                     or _denegate(echo) in _denegate(ground_truth_reply))
    if (len(echo) >= MIN_ECHO_CHARS
            and not core_named_it
            and re.search(rf"\b{re.escape(echo)}\b", low)):
        return "echoed the person's own words back at them"

    # Who is who. "you're Sam" must not come back as "I'm Sam".
    said_of_person = _predicate(_YOU_ARE_RE, ground_truth_reply)
    said_of_self = _predicate(_I_AM_RE, ground_truth_reply)
    inverted = (_predicate(_I_AM_RE, text) & said_of_person - said_of_self) \
        | (_predicate(_YOU_ARE_RE, text) & said_of_self - said_of_person)
    if inverted:
        return f"swapped the speakers ({sorted(inverted)[0]!r})"

    # Names and numbers must come from the core or from the person. A model
    # that supplies its own is answering a question the core left open, and
    # doing it through a channel that records no source for the answer.
    supplied = (_referents(ground_truth_reply, strict=False)
                | _referents(user_message, strict=False))
    invented = _referents(text) - supplied - _ALWAYS_PERMITTED
    if invented:
        return f"introduced a name or number the core did not supply ({sorted(invented)[0]!r})"

    # Polarity. The echo exemption above lets a naturalisation quote a
    # denial; it must not let it turn the core's own belief around. If the
    # core stated the proposition one way and the reply states it only the
    # other way, the reply has changed the outcome.
    atom_words = _words(_denegate(echo))
    if len(echo) >= MIN_ECHO_CHARS:
        core = _polarities(ground_truth_reply, atom_words)
        shown = _polarities(text, atom_words)
        if core and shown and not (core & shown):
            return "reversed the polarity of what the core said"

    # Stance. The core has already decided yes, no or neither; the model may
    # word that decision but not take another one. "Neither" is included
    # deliberately: an "I don't know" that comes back as "Yes" is the same
    # failure, and the likelier one.
    if stance is not None:
        shown_stance = _opening_stance(text)
        if shown_stance is not None and shown_stance != stance:
            core_stance = getattr(stance, "value", stance)
            return f"answered {shown_stance!r} where the core answered {core_stance!r}"

    limit = max(LENGTH_FLOOR, LENGTH_SLACK * len(ground_truth_reply.strip()))
    if len(text) > limit:
        return f"far longer than the reply it was given ({len(text)} > {limit})"

    # Nothing left out. The names in the core's reply are who and what it
    # decided something about, and a phrasing that drops one has told the
    # person something else ("You're welcome" for "that matches:
    # Humuhumunukunukapuaa is a fish"). The core writes no labels, so a
    # capital after its colons is a name ("I'll double check: Hector is a
    # dog"). A number may be left out -- "(confidence 0.63)" is detail -- but
    # never changed, which the check above already covers.
    required = {r for r in _referents(ground_truth_reply, strict=True, breaks=_CORE_SENTENCE_BREAK)
                if not r[0].isdigit()} - _ALWAYS_PERMITTED
    missing = sorted(r for r in required
                     if not re.search(rf"(?<![\w'’]){re.escape(r)}(?![\w'’])", low))
    if missing:
        return f"left out a name the core gave ({missing[0]!r})"


    return None


def split_verbatim(ground_truth_reply: str, verbatim: str):
    """Split the core's reply into the part the model may phrase and the
    part it may not.

    Returns (to_phrase, verbatim). `to_phrase` may be empty -- a reply that
    is only a question has nothing to phrase. When the reply does not end
    with the text it names, nothing is phrased at all: the whole reply goes
    out as the core wrote it, since the split cannot be trusted."""
    if not verbatim:
        return ground_truth_reply, ""
    text = ground_truth_reply.rstrip()
    if not text.endswith(verbatim):
        return "", text
    return text[: -len(verbatim)].rstrip(), verbatim
