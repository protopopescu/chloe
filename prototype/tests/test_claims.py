"""
Tests for how a claim is recorded: a statement, a denial, and a yes or no
to a question all land on the belief they are about, and the reply and
the change in trust are judged against that belief's standing before the
claim arrived.

Stdlib only, offline (the pattern parser):

    python3 -m unittest tests.test_claims
"""

import os
import tempfile
import unittest

from chloe import grammar
from chloe.dialogue import ChloeEngine
from chloe.models import Atom, AtomStatus
from chloe.storage import KnowledgeStore


class ClaimTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.store = KnowledgeStore(self.path)

    def tearDown(self):
        self.store.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _speaker(self, name):
        engine = ChloeEngine(self.store)
        engine.greet(name)
        engine.turn("no")          # decline the secret-word offer
        return engine

    def _atoms(self, subject):
        return [a for a in self.store.all_atoms() if a.subject.lower() == subject.lower()]

    def _contradict_bob_the_scientist(self):
        self._speaker("Ben").turn("Bob is a scientist")
        for name in ["Alice", "Eve", "Carl"]:
            self._speaker(name).turn("Bob is not a scientist")
        atom = self._atoms("Bob")[0]
        self.assertEqual(atom.status, AtomStatus.CONTRADICTED)
        return atom

    # --------------------------------------------- standing, not polarity
    def test_denying_a_contradicted_belief_agrees_with_it(self):
        self._contradict_bob_the_scientist()
        bob = self._speaker("Bob")
        before, open_before = bob.person.trust_in(), self.store.pending_questions()
        reply = bob.turn("I am not a scientist")
        self.assertIn("I don't believe that you are a scientist", reply)
        self.assertNotIn("conflicts", reply)
        self.assertGreater(bob.person.trust_in(), before)
        self.assertEqual(self.store.pending_questions(), open_before, "nothing new to double check")

    def test_asserting_a_contradicted_belief_does_not_match_it(self):
        self._contradict_bob_the_scientist()
        fay = self._speaker("Fay")
        before = fay.person.trust_in()
        reply = fay.turn("Bob is a scientist")
        self.assertNotIn("matches", reply)
        self.assertIn("others have denied it", reply)
        self.assertLess(fay.person.trust_in(), before)

    def test_corroborating_a_hypothesis_is_not_a_prior_belief(self):
        self.store.save_atom(Atom(id=None, subject="Cora", relation="is",
                                  object="a computer scientist", status=AtomStatus.HYPOTHESIS,
                                  confidence=0.6))
        reply = self._speaker("Bob").turn("Cora is a computer scientist")
        self.assertNotIn("already believed", reply)
        self.assertIn("worked that out myself", reply)

    # ------------------------------------------------- one claim, one atom
    def test_a_denial_and_a_negative_statement_are_one_belief(self):
        self._speaker("Dan").turn("George is not busy")
        self._speaker("Eve").turn("George is not busy")
        atoms = self._atoms("George")
        self.assertEqual(len(atoms), 1)
        self.assertEqual(len(atoms[0].provenance), 2)

    def test_asserting_what_a_negative_belief_rules_out_disputes_it(self):
        self._speaker("Dan").turn("George is not busy")
        reply = self._speaker("Eve").turn("George is busy")
        atoms = self._atoms("George")
        self.assertEqual(len(atoms), 1, [a.statement() for a in atoms])
        self.assertEqual([p.polarity for p in atoms[0].provenance], [1, -1])
        self.assertIn("conflicts", reply)

    def test_a_negative_fact_is_no_rival_value(self):
        self._speaker("Dan").turn("Felix is a cat")
        reply = self._speaker("Eve").turn("Felix is not a dog")
        self.assertNotIn("contradicts", reply)
        cat = [a for a in self._atoms("Felix") if a.object == "a cat"][0]
        self.assertEqual([p.polarity for p in cat.provenance], [1])

    def test_one_belief_whatever_the_surface_form(self):
        self.assertEqual(grammar.identity("Cora", "likes", "to cook"),
                         grammar.identity("cora", "like", "to  Cook."))
        self.assertEqual(grammar.identity("Bob's cat", "is called", "hector"),
                         grammar.identity("Bob's cat", "is called", "Hector"))
        self.assertNotEqual(grammar.identity("Felix", "is", "a cat"),
                            grammar.identity("Felix", "is", "a black cat"))

    def test_surface_variants_meet_when_they_are_said(self):
        self._speaker("Sam").turn("Cora is a cook")
        reply = self._speaker("Bob").turn("cora is a Cook.")
        atoms = self._atoms("Cora")
        self.assertEqual(len(atoms), 1)
        self.assertEqual(len(atoms[0].provenance), 2)
        self.assertIn("matches", reply)

    # ------------------------------------------------ unverified identity
    def _unverified_cora(self):
        cora = ChloeEngine(self.store)
        cora.greet("Cora")
        cora.turn("yes")                          # set a secret word
        cora.turn("aubergine")
        impostor = ChloeEngine(self.store)
        impostor.greet("Cora")
        for guess in ["one", "two", "three"]:
            impostor.turn(guess)
        self.assertTrue(impostor.person.is_unverified)
        return impostor

    def test_an_unverified_claim_about_oneself_is_not_stored(self):
        impostor = self._unverified_cora()
        reply = impostor.turn("I am a computer scientist")
        self.assertIn("can't be sure you're Cora", reply)
        self.assertEqual(self.store.all_atoms(), [])
        impostor.turn("my cat is black")
        self.assertEqual(self.store.all_atoms(), [])

    def test_an_unverified_claim_under_the_claimed_name_is_not_stored_either(self):
        impostor = self._unverified_cora()
        reply = impostor.turn("Cora is a computer scientist")
        self.assertIn("can't be sure you're Cora", reply)
        self.assertEqual(self.store.all_atoms(), [], "naming herself in the third person "
                                                     "is the same claim")

    def test_an_unverified_person_is_still_a_source_about_others(self):
        impostor = self._unverified_cora()
        impostor.turn("Felix is a cat")
        self.assertEqual(len(self._atoms("Felix")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
