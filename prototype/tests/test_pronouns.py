"""
Tests for grammar.py: pronoun resolution inbound, pronoun swapping and
copula agreement outbound, the fact that what lands in the store is
third-person and person-neutral, and that statements which resolve to a
tautology are refused rather than stored.

Runs offline against the pattern parser. Stdlib only:

    python3 -m unittest tests.test_pronouns
"""

import os
import tempfile
import unittest

from chloe import grammar
from chloe.dialogue import ChloeEngine
from chloe.nlu import parse
from chloe.storage import KnowledgeStore


class GrammarUnitTests(unittest.TestCase):
    def test_resolve_first_and_second_person(self):
        self.assertEqual(grammar.resolve_text("I", "Dan"), "Dan")
        self.assertEqual(grammar.resolve_text("me", "Dan"), "Dan")
        self.assertEqual(grammar.resolve_text("you", "Dan"), "Chloe")
        self.assertEqual(grammar.resolve_text("my cat", "Dan"), "Dan's cat")
        self.assertEqual(grammar.resolve_text("your name", "Dan"), "Chloe's name")

    def test_third_person_untouched(self):
        self.assertEqual(grammar.resolve_text("the sky", "Dan"), "the sky")
        self.assertEqual(grammar.resolve_text("we", "Dan"), "we")

    def test_swap_pronoun_is_listener_relative(self):
        self.assertEqual(grammar.swap_pronoun("Dan", "Dan"), "you")
        self.assertEqual(grammar.swap_pronoun("Chloe", "Dan"), "I")
        self.assertEqual(grammar.swap_pronoun("Dan", "Alice"), "Dan")
        self.assertEqual(grammar.swap_pronoun("Felix", "Dan"), "Felix")

    def test_same_referent(self):
        self.assertTrue(grammar.same_referent("Dan", "dan"))
        self.assertFalse(grammar.same_referent("Dan", "Alice"))
        self.assertFalse(grammar.same_referent("Dan", ""))

    def test_accord_verb(self):
        self.assertEqual(grammar.accord_verb("is", "you"), "are")
        self.assertEqual(grammar.accord_verb("is", "I"), "am")
        self.assertEqual(grammar.accord_verb("is", "Felix"), "is")
        self.assertEqual(grammar.accord_verb("likes", "Felix"), "likes")

    def test_clause_agrees(self):
        self.assertEqual(grammar.clause("Dan", "is", "a person", speaker="Dan"),
                         "you are a person")
        self.assertEqual(grammar.clause("Chloe", "is", "a machine", speaker="Dan"),
                         "I am a machine")
        self.assertEqual(grammar.clause("Felix", "is", "Dan's cat", speaker="Dan"),
                         "Felix is your cat")
        self.assertEqual(grammar.clause("the sky", "is", "blue", "it is night", "Dan"),
                         "the sky is blue (it is night)")

    def test_wh_clause_agrees(self):
        self.assertEqual(grammar.wh_clause("Chloe", "Dan"), "what am I")
        self.assertEqual(grammar.wh_clause("Dan", "Dan"), "what are you")
        self.assertEqual(grammar.wh_clause("Felix", "Dan"), "what is Felix")

    def test_wh_clause_keeps_the_askers_wh_word(self):
        self.assertEqual(grammar.wh_clause("Dan", "Dan", "who"), "who are you")
        self.assertEqual(grammar.wh_clause("Chloe", "Dan", "who"), "who am I")

    def test_parser_reports_the_wh_word(self):
        self.assertEqual(parse("who is Felix?").extra.get("wh"), "who")
        self.assertEqual(parse("what is Felix?").extra.get("wh"), "what")

    def test_pronoun_subject_stores_third_person_copula(self):
        utt = grammar.resolve_referents(parse("I am a person"), "Dan")
        self.assertEqual((utt.subject, utt.relation, utt.obj), ("Dan", "is", "a person"))

    def test_plural_are_is_not_de_accorded(self):
        utt = grammar.resolve_referents(parse("cats are animals"), "Dan")
        self.assertEqual((utt.subject, utt.relation), ("cats", "are"))


class DialoguePronounTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.store = KnowledgeStore(self.path)

    def tearDown(self):
        self.store.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _engine(self, name):
        e = ChloeEngine(self.store)
        e.greet(name)
        e.turn("no")  # decline the secret-word offer
        return e

    def test_statement_and_question_about_the_speaker(self):
        dan = self._engine("Dan")
        self.assertIn("you are a person", dan.turn("I am a person"))
        self.assertTrue(dan.turn("who am I?").startswith("You are a person"))

    def test_statement_and_question_about_chloe(self):
        dan = self._engine("Dan")
        self.assertIn("I am a machine", dan.turn("you are a machine"))
        self.assertTrue(dan.turn("what are you?").startswith("I am a machine"))

    def test_unknown_subject_question_still_agrees(self):
        dan = self._engine("Dan")
        self.assertEqual(dan.turn("what are you?"), "Only that I'm Chloe -- what am I?")

    def test_a_who_question_is_answered_as_a_who_question(self):
        dan = self._engine("Dan")
        self.assertEqual(dan.turn("who am I?"), "Only that you're Dan -- who are you?")
        self.assertEqual(dan.turn("who is Felix?"), "I don't know yet -- who is Felix?")
        self.assertEqual(dan.turn("what is Felix?"), "I don't know yet -- what is Felix?")

    def test_a_name_is_not_reported_as_total_ignorance(self):
        # greet() binds the name on the Person, not as an atom, but answering
        # a flat "I don't know" right after using it reads as a contradiction.
        dan = self._engine("Dan")
        self.assertTrue(dan.turn("who am I?").startswith("Only that you're Dan"))

    def test_possessive_in_object(self):
        dan = self._engine("Dan")
        self.assertIn("Felix is your cat", dan.turn("Felix is my cat"))

    def test_store_holds_third_person_only(self):
        dan = self._engine("Dan")
        dan.turn("I am a person")
        dan.turn("you are a machine")
        dan.turn("Felix is my cat")
        stored = sorted(a.statement() for a in self.store.all_atoms())
        self.assertEqual(stored, ["Chloe is a machine", "Dan is a person", "Felix is Dan's cat"])

    def test_another_person_is_not_addressed_as_you(self):
        self._engine("Dan").turn("I am a person")
        alice = self._engine("Alice")
        self.assertTrue(alice.turn("what is Dan?").startswith("Dan is a person"))
        self.assertEqual(alice.turn("who am I?"), "Only that you're Alice -- who are you?")

    def test_self_identification_is_not_stored_as_a_belief(self):
        # Pronoun resolution turns "I am Dan", said by Dan, into "Dan is Dan".
        dan = self._engine("Dan")
        self.assertIn("I have you as Dan", dan.turn("I am Dan"))
        self.assertIn("I have you as Dan", dan.turn("I am me"))
        self.assertEqual(self.store.all_atoms(), [])

    def test_tautology_is_refused(self):
        dan = self._engine("Dan")
        self.assertIn("tells me nothing about Felix", dan.turn("Felix is Felix"))
        self.assertIn("fail to be itself", dan.turn("Felix is not Felix"))
        self.assertEqual(dan.turn("am I me?"), "Yes -- necessarily.")
        self.assertEqual(self.store.all_atoms(), [])

    def test_real_beliefs_still_stored_after_the_guards(self):
        dan = self._engine("Dan")
        dan.turn("I am a physicist")
        self.assertEqual([a.statement() for a in self.store.all_atoms()], ["Dan is a physicist"])
        self.assertTrue(dan.turn("who am I?").startswith("You are a physicist"))

    def test_queued_question_keeps_the_real_name(self):
        dan = self._engine("Dan")
        dan.turn("who am I?")
        self.assertEqual([q["question"] for q in self.store.pending_questions()],
                         ["What is Dan?"])

    def test_another_persons_name_is_not_a_tautology(self):
        dan = self._engine("Dan")
        self.assertIn("Alice", dan.turn("Felix is Alice"))
        self.assertEqual([a.statement() for a in self.store.all_atoms()], ["Felix is Alice"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
