"""
Tests for the conversational half of the linguistic interface: pleasantries
are answered rather than refused, discourse packaging is stripped from
claims, and neither writes anything CHLOE wasn't actually told.

Stdlib only, offline (the pattern parser; the LLM path is prompted for the
same distinctions and covered by test_llm_parser.py):

    python3 -m unittest tests.test_smalltalk
"""

import os
import tempfile
import unittest

from chloe.dialogue import ChloeEngine
from chloe.nlu import UtteranceType, parse
from chloe.storage import KnowledgeStore


class ParsingTests(unittest.TestCase):
    def test_pleasantries_are_small_talk_not_failures(self):
        for text in ["Hello", "hi", "Good morning", "How are you, Chloe?",
                     "how are you doing?", "thanks!", "Thank you very much",
                     "cheers", "nice to meet you"]:
            self.assertEqual(parse(text).type, UtteranceType.SMALL_TALK, text)

    def test_real_nonsense_is_still_unknown(self):
        # small talk must not become a dustbin for anything unparsed
        for text in ["asdf qwer zxcv", "the quick brown fox jumped"]:
            self.assertEqual(parse(text).type, UtteranceType.UNKNOWN, text)

    def test_hedges_and_markers_are_stripped_from_claims(self):
        cases = {
            "The sky is blue, actually.": ("The sky", "blue", UtteranceType.STATEMENT),
            # "indeed" is confirmation, not part of what is being confirmed:
            # stored unstripped it makes a corroboration look like a rival value.
            "Claire is an artist, indeed.": ("Claire", "an artist", UtteranceType.STATEMENT),
            "Well, I think Felix is a cat": ("Felix", "a cat", UtteranceType.STATEMENT),
            "Honestly, grass is green": ("grass", "green", UtteranceType.STATEMENT),
            "No, the sky is not green": ("the sky", "green", UtteranceType.NEGATION),
        }
        for text, (subj, obj, typ) in cases.items():
            u = parse(text)
            self.assertEqual((u.type, u.subject, u.obj), (typ, subj, obj), text)

    def test_a_claim_wrapped_in_a_greeting_is_still_a_claim(self):
        u = parse("Hi Chloe, the sky is blue")
        self.assertEqual((u.type, u.subject, u.obj),
                         (UtteranceType.STATEMENT, "the sky", "blue"))

    def test_trailing_vocative_needs_a_comma(self):
        # "..., Chloe?" is address; "is Chloe" is a claim about Chloe
        self.assertEqual(parse("How are you, Chloe?").type, UtteranceType.SMALL_TALK)
        u = parse("Chloe is a robot")
        self.assertEqual((u.type, u.subject, u.obj),
                         (UtteranceType.STATEMENT, "Chloe", "a robot"))

    def test_bare_markers_are_not_swallowed(self):
        # stripping only applies when a claim follows
        self.assertEqual(parse("no").type, UtteranceType.UNKNOWN)
        self.assertEqual(parse("I think").type, UtteranceType.UNKNOWN)


class DialogueTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.store = KnowledgeStore(self.path)
        self.dan = ChloeEngine(self.store)
        self.dan.greet("Dan")
        self.dan.turn("no")

    def tearDown(self):
        self.store.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_small_talk_is_answered_not_refused(self):
        for text in ["Hello", "How are you?", "thanks"]:
            reply = self.dan.turn(text)
            self.assertNotIn("rephrase", reply, text)
            self.assertNotIn("don't understand", reply, text)

    def test_small_talk_writes_nothing(self):
        for text in ["Hello", "How are you, Chloe?", "thanks!", "good morning"]:
            self.dan.turn(text)
        self.assertEqual(self.store.all_atoms(), [])

    def test_she_greets_by_name(self):
        self.assertIn("Dan", self.dan.turn("Hello"))

    def test_how_are_you_reports_what_she_holds(self):
        self.assertIn("don't know much yet", self.dan.turn("How are you?"))
        self.dan.turn("the sky is blue")
        self.assertIn("1 thing on file", self.dan.turn("How are you?"))
        self.dan.turn("grass is green")
        self.assertIn("2 things on file", self.dan.turn("How are you?"))

    def test_a_hedged_claim_is_still_recorded(self):
        self.dan.turn("Well, I think the sky is blue, actually.")
        self.assertEqual([a.statement() for a in self.store.all_atoms()],
                         ["the sky is blue"])

    def test_unparseable_input_still_refuses(self):
        self.assertIn("rephrase", self.dan.turn("asdf qwer zxcv"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
