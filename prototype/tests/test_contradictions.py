"""
Tests for what happens to a belief under dispute: the rival value is kept as
a belief of its own, and a belief whose evidence has turned against it stops
answering while staying in the store.

Stdlib only, offline (the pattern parser):

    python3 -m unittest tests.test_contradictions
"""

import os
import tempfile
import unittest

from chloe.dialogue import ChloeEngine
from chloe.models import AtomStatus, Stance
from chloe.storage import KnowledgeStore


class ContradictionTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.store = KnowledgeStore(self.path)
        self.ben = self._speaker("Ben")
        self.ben.turn("Bob is a scientist")
        self.dan = self._speaker("Dan")

    def tearDown(self):
        self.store.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _speaker(self, name):
        engine = ChloeEngine(self.store)
        engine.greet(name)
        return engine

    def _atom(self, statement):
        for atom in self.store.all_atoms():
            if atom.statement() == statement:
                return atom
        return None

    def _deny_until_contradicted(self):
        for name in ["Alice", "Eve", "Carl"]:
            self._speaker(name).turn("Bob is not a scientist")
        self.assertEqual(self._atom("Bob is a scientist").status, AtomStatus.CONTRADICTED)

    def test_a_rival_value_is_stored_as_a_belief_of_its_own(self):
        self.dan.turn("Bob is a builder")
        self.assertIsNotNone(self._atom("Bob is a builder"))
        self.assertEqual(self._atom("Bob is a builder").status, AtomStatus.CANDIDATE)

    def test_the_incumbent_still_takes_the_evidence_against_it(self):
        before = self._atom("Bob is a scientist").confidence
        self.dan.turn("Bob is a builder")
        self.assertLess(self._atom("Bob is a scientist").confidence, before)

    def test_repeating_a_rival_carries_it_rather_than_only_eroding_the_incumbent(self):
        for _ in range(3):
            self.dan.turn("Bob is a builder")
        rival = self._atom("Bob is a builder")
        self.assertEqual(len(rival.provenance), 3)
        self.assertGreater(rival.confidence, self._atom("Bob is a scientist").confidence)
        self.assertIn("builder", self.dan.turn("what is Bob?"))

    def test_a_contradicted_belief_does_not_answer(self):
        self._deny_until_contradicted()
        self.assertNotIn("scientist", self.dan.turn("what is Bob?"))

    def test_a_contradicted_belief_does_not_answer_yes_or_no(self):
        self._deny_until_contradicted()
        reply = self.dan.turn("is Bob a scientist?")
        self.assertIn("conflicting", reply)
        self.assertEqual(self.dan.last_stance, Stance.UNKNOWN)

    def test_a_value_that_still_stands_answers_in_its_place(self):
        self._deny_until_contradicted()
        self.dan.turn("Bob is a builder")
        self.assertIn("builder", self.dan.turn("what is Bob?"))
        self.assertIn("builder", self.dan.turn("is Bob a scientist?"))

    def test_a_contradicted_belief_keeps_its_row_and_its_provenance(self):
        self._deny_until_contradicted()
        self.dan.turn("what is Bob?")
        atom = self._atom("Bob is a scientist")
        self.assertIsNotNone(atom)
        self.assertEqual(len(atom.provenance), 4)

    def test_further_evidence_still_reaches_a_contradicted_belief(self):
        self._deny_until_contradicted()
        self._speaker("Fay").turn("Bob is a scientist")
        self.assertEqual(len(self._atom("Bob is a scientist").provenance), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
