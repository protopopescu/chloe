"""
Person-relative pronoun handling.

Ports `SwapPronoun()` and `AccordTheVerb()` from the 2000 C++ (cGrammar.hh).
The original swapped first- and second-person pronouns just before CHLOE
spoke, and agreed the copula with the result, so that a question about
"I" came back as an answer about "you".

The same job splits in two here, because beliefs are now stored rather
than echoed:

    resolve_referents()  inbound. "I"/"me"/"my" become the speaker's name
        and "you"/"your" become Chloe's, so a belief is recorded about a
        named entity. Stored literally, "I" would be one atom shared by
        every interlocutor who ever used the word.

    clause() / wh_clause()  outbound, the inverse. The speaker's own name
        renders as "you", Chloe's as "I", and the copula is agreed with
        whichever pronoun came out.

Queued questions deliberately do not use these: a question stored now may
be asked of someone else later, so it keeps the real names.

Only first- and second-person singular is handled. "we"/"our" have no
single referent to resolve to and are left alone.
"""

import re
from typing import Optional

CHLOE_NAME = "Chloe"

_FIRST_PERSON = {"i", "me", "myself"}
_SECOND_PERSON = {"you", "yourself"}
_FIRST_POSSESSIVE = {"my", "mine"}
_SECOND_POSSESSIVE = {"your", "yours"}

PRONOUNS = _FIRST_PERSON | _SECOND_PERSON | _FIRST_POSSESSIVE | _SECOND_POSSESSIVE

_WORD_RE = re.compile(r"[A-Za-z']+")


def _possessive(name: str) -> str:
    return name + "'s"


def _resolve_word(word: str, speaker: str) -> Optional[str]:
    low = word.lower()
    if low in _FIRST_PERSON:
        return speaker
    if low in _SECOND_PERSON:
        return CHLOE_NAME
    if low in _FIRST_POSSESSIVE:
        return _possessive(speaker)
    if low in _SECOND_POSSESSIVE:
        return _possessive(CHLOE_NAME)
    return None


def resolve_text(text: str, speaker: str) -> str:
    """Rewrite first- and second-person words to the entities they denote."""
    if not text or not speaker:
        return text

    def sub(m):
        return _resolve_word(m.group(0), speaker) or m.group(0)

    return _WORD_RE.sub(sub, text)


def is_pronoun(text: str) -> bool:
    return bool(text) and text.strip().lower() in PRONOUNS


def resolve_referents(utt, speaker: str):
    """Resolve pronouns in an Utterance in place, and return it.

    A copula agreed with a pronoun subject ("I am", "you are") is stored in
    its third-person form, so that "I am a person" and "Dan is a person"
    reach the same atom. Third-person "are" is left alone: "cats are
    animals" is plural, not a mis-agreed singular.
    """
    if not speaker:
        return utt

    subject_was_pronoun = is_pronoun(utt.subject or "")

    utt.subject = resolve_text(utt.subject, speaker) if utt.subject else utt.subject
    utt.obj = resolve_text(utt.obj, speaker) if utt.obj else utt.obj
    utt.scope = resolve_text(utt.scope, speaker) if utt.scope else utt.scope

    if subject_was_pronoun and (utt.relation or "").lower() in ("am", "are"):
        utt.relation = "is"

    return utt


def swap_pronoun(word: str, speaker: str) -> str:
    """cGrammar.hh SwapPronoun(): a stored name as the listener would say it."""
    if not word or not speaker:
        return word
    low = word.strip().lower()
    if low == speaker.strip().lower():
        return "you"
    if low == CHLOE_NAME.lower():
        return "I"
    if low == _possessive(speaker).lower():
        return "your"
    if low == _possessive(CHLOE_NAME).lower():
        return "my"
    return word


def swap_pronouns_in(text: str, speaker: str) -> str:
    """swap_pronoun() applied to every word of a phrase, so that an object
    or scope clause reads back in the second person too."""
    if not text or not speaker:
        return text
    names = {
        speaker.strip().lower(): "you",
        CHLOE_NAME.lower(): "I",
        _possessive(speaker).lower(): "your",
        _possessive(CHLOE_NAME).lower(): "my",
    }
    return _WORD_RE.sub(lambda m: names.get(m.group(0).lower(), m.group(0)), text)


def same_referent(a: str, b: str) -> bool:
    """Whether two resolved phrases denote the same thing. Used to catch
    statements that say nothing: "I am me" and "I am Dan" both resolve to
    "Dan is Dan"."""
    return bool(a) and bool(b) and a.strip().lower() == b.strip().lower()


def accord_verb(verb: str, pronoun: str) -> str:
    """cGrammar.hh AccordTheVerb(): agree a copula with the pronoun in front
    of it. Anything that is not a first/second-person pronoun keeps the verb
    as stored."""
    low = (pronoun or "").strip().lower()
    if low == "you":
        return "are"
    if low == "i":
        return "am"
    return verb


def clause(subject: str, relation: str, obj: str, scope: str = "", speaker: str = "") -> str:
    """Render a stored triple for the person being spoken to."""
    subj = swap_pronoun(subject, speaker)
    rel = accord_verb(relation, subj)
    text = f"{subj} {rel} {swap_pronouns_in(obj, speaker)}"
    if scope:
        text += f" ({swap_pronouns_in(scope, speaker)})"
    return text


def wh_clause(subject: str, speaker: str = "", wh: str = "what") -> str:
    """Render a wh-question with X in the right person, keeping the
    wh-word the asker used: "who are you", "what am I", "what is Felix"."""
    subj = swap_pronoun(subject, speaker)
    return f"{wh} {accord_verb('is', subj)} {subj}"


def capitalise(text: str) -> str:
    """Sentence-initial capital that leaves an already-correct "I" alone."""
    return text[:1].upper() + text[1:] if text else text
