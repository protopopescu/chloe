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

A queued question stores no wording. It is rendered with clause() when it
is asked, for whoever is being asked, so the same question reads "you"
to one person and a name to another.

Only first- and second-person singular is handled. "we"/"our" have no
single referent to resolve to and are left alone.

The module also defines when two phrasings are the same belief
(identity()). Every lookup, merge and duplicate check uses it, so a
statement meets the atom it corroborates at the moment it is made rather
than only after a sleep.
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


_COPULA_PRESENT = {"is", "are", "am"}
_COPULA_PAST = {"was", "were"}


def accord_verb(verb: str, pronoun: str) -> str:
    """cGrammar.hh AccordTheVerb(): agree a verb with the pronoun in front of
    it. A copula takes that person's form and keeps its tense. Any other verb
    is marked only for the third person. Anything that is not a
    first/second-person pronoun keeps the verb as stored."""
    low = (pronoun or "").strip().lower()
    if low not in ("you", "i"):
        return verb
    stem = verb.strip().lower()
    if stem in _COPULA_PRESENT:
        return "are" if low == "you" else "am"
    if stem in _COPULA_PAST:
        return "were" if low == "you" else "was"
    return _lemma(stem)


def clause(subject: str, relation: str, obj: str, scope: str = "", speaker: str = "") -> str:
    """Render a stored triple for the person being spoken to."""
    subj = swap_pronoun(subject, speaker)
    return f"{subj} {predicate(relation, obj, scope, speaker, subj)}"


def predicate(relation: str, obj: str, scope: str = "", speaker: str = "",
              subject_pronoun: str = "") -> str:
    """The part of a clause after its subject. The verb agrees with
    `subject_pronoun`, the subject as swap_pronoun() has already rendered
    it."""
    text = f"{accord_verb(relation, subject_pronoun)} {swap_pronouns_in(obj, speaker)}"
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


# ------------------------------------------------------------- identity

# Forms of one relation, for identity only -- surface variants, not a claim
# about their semantics.
RELATION_SYNONYMS = {"is": "is", "are": "is", "am": "is", "was": "is", "were": "is",
                     "means": "is", "has": "have"}


def _lemma(verb: str) -> str:
    """Third-person -s removed: 'likes' and 'like' are one relation. Crude,
    and only ever applied to the verb that heads a relation."""
    if verb in RELATION_SYNONYMS:
        return RELATION_SYNONYMS[verb]
    if len(verb) > 4 and verb.endswith("ies"):
        return verb[:-3] + "y"
    if len(verb) > 3 and verb.endswith(("sses", "shes", "ches", "xes", "zes", "oes")):
        return verb[:-2]
    if len(verb) > 2 and verb.endswith("s") and not verb.endswith(("ss", "us", "is")):
        return verb[:-1]
    return verb


def norm_phrase(text: str) -> str:
    """Case, spacing and trailing punctuation are not part of what was said."""
    return " ".join((text or "").casefold().split()).strip(".,;:!? ")


def norm_relation(relation: str) -> str:
    words = norm_phrase(relation).split()
    if not words:
        return ""
    return " ".join([_lemma(words[0])] + words[1:])


def identity(subject: str, relation: str, obj: str, scope: str = "") -> tuple:
    """When two phrasings are one belief: 'Cora like to cook' and 'cora
    likes to cook', 'hector' and 'Hector'."""
    return (norm_phrase(subject), norm_relation(relation), norm_phrase(obj), norm_phrase(scope))


def refers_to(subject: str, speaker: str) -> bool:
    """Whether a resolved subject is about the speaker: their name, or
    something of theirs ("Bob's cat")."""
    if not subject or not speaker:
        return False
    s, name = norm_phrase(subject), norm_phrase(speaker)
    return s == name or s.startswith(norm_phrase(_possessive(speaker)) + " ")


# ------------------------------------------------------------- grounding

_CONTENT_RE = re.compile(r"[A-Za-z0-9']+")

# Words a reading may use without their appearing in what it was read
# from: they carry no content of their own, and demanding them back would
# refuse "a black cat" for the sake of its article.
FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "and", "or", "is", "are", "was", "were", "be", "been", "am", "not",
    "that", "this", "these", "those", "its", "his", "her", "their", "some", "it",
}


def fold(word: str) -> str:
    """Crude singular fold, enough to let 'cats' and 'a cat' meet. Nothing
    here is linguistics; it is the smallest normalisation that stops a
    grounding check refusing a reading over a plural."""
    w = word.strip().lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def vocabulary(texts) -> set:
    """Folded content words, with a contraction also giving its stem, so
    that "I'm" grounds "I" and "Felix's" grounds "Felix"."""
    vocab = set()
    for t in texts:
        for w in _CONTENT_RE.findall((t or "").lower()):
            vocab.add(fold(w))
            vocab.add(fold(w.split("'")[0]))
    return vocab


def ungrounded_words(text: str, vocab: set) -> list:
    """Content words of `text` that appear nowhere in `vocab`."""
    return [w for w in _CONTENT_RE.findall((text or "").lower())
            if w not in FUNCTION_WORDS and fold(w) not in vocab]


def in_source_case(text: str, source: str) -> str:
    """Each word of `text` spelled as `source` spells it, where it does:
    a reading keeps the person's own capitals ("Hector", not "hector")."""
    spelling = {w.lower(): w for w in _CONTENT_RE.findall(source or "")}
    return _CONTENT_RE.sub(lambda m: spelling.get(m.group(0).lower(), m.group(0)), text or "")
