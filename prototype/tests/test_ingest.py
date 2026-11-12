"""
Tests for taking in another instance's core (chloe/ingest.py): evidence
from both lands on one belief, identity follows the sign-in rules, and
ingesting again, or ingesting a core that has passed through a third
instance, never counts the same evidence twice.

Stdlib only, offline (the pattern parser):

    python3 -m unittest tests.test_ingest
"""

import contextlib
import hashlib
import io
import os
import shutil
import tempfile
import unittest

from chloe import entailment, questions
from chloe.dialogue import ChloeEngine
from chloe.ingest import ingest
from chloe.models import Atom, AtomStatus, RelationProperties
from chloe.storage import KnowledgeStore

import ingest_core


def _state(store):
    """What a store believes, and on what evidence."""
    return sorted((a.statement(), a.status.value, round(a.confidence, 9), len(a.provenance))
                  for a in store.all_atoms())


class IngestTestCase(unittest.TestCase):
    def setUp(self):
        os.environ.pop("VLLM_BASE_URL", None)
        self.dir = tempfile.mkdtemp()
        self.stores = []
        self.here = self.store("here.db")
        self.there = self.store("there.db")

    def tearDown(self):
        for store in self.stores:
            store.close()
        shutil.rmtree(self.dir)

    def store(self, name):
        store = KnowledgeStore(os.path.join(self.dir, name))
        self.stores.append(store)
        return store

    def talk(self, store, name, *lines, secret=None):
        engine = ChloeEngine(store)
        engine.greet(name)
        if secret is not None:
            engine.turn("yes")
            engine.turn(secret)
        for line in lines:
            engine.turn(line)
        return engine

    def take_in(self, store=None, source=None):
        return ingest(store or self.here, (source or self.there).path)

    def belief(self, store, statement):
        for atom in store.all_atoms():
            if atom.statement() == statement:
                return atom
        return None

    def person(self, store, atom_or_prov):
        return store.person_by_id(atom_or_prov.person_id).name

    # ------------------------------------------------------- faithfulness
    def test_a_core_taken_into_an_empty_store_holds_what_it_held(self):
        store = self.there
        store.declare_relation(RelationProperties(name="is", transitive=True))
        self.talk(store, "Ben", "Bob is a scientist", "Felix is a cat")
        self.talk(store, "Dan", "Bob is a builder", "Bob is not a scientist")
        self.talk(store, "Eve", "Bob is not a scientist", "Felix is a cat")
        self.take_in()
        self.assertEqual(_state(self.here), _state(self.there))
        self.assertEqual(sorted((p.name, p.trust) for p in self.here.all_people()),
                         sorted((p.name, p.trust) for p in self.there.all_people()))
        self.assertEqual(len(self.here.pending_questions()), len(self.there.pending_questions()))
        self.assertTrue(self.here.get_relation("is").transitive)

    def test_a_hypothesis_comes_over_as_a_hypothesis_with_its_question(self):
        hyp = self.there.save_atom(Atom(id=None, subject="Felix", relation="is", object="an animal",
                                        status=AtomStatus.HYPOTHESIS, confidence=0.6))
        questions.queue(self.there, "hypothesis", hyp)
        self.take_in()
        ours = self.belief(self.here, "Felix is an animal")
        self.assertEqual(ours.status, AtomStatus.HYPOTHESIS)
        self.assertEqual([q["reason"] for q in self.here.pending_questions()], ["hypothesis"])

    def test_the_source_is_only_read(self):
        self.talk(self.there, "Ben", "Bob is a scientist")
        self.there.close()
        with open(self.there.path, "rb") as f:
            before = hashlib.sha256(f.read()).hexdigest()
        self.take_in()
        with open(self.there.path, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), before)

    def test_a_persons_conversation_comes_with_them(self):
        self.talk(self.there, "Ben", "Bob is a scientist", "what is Bob?")
        self.take_in()
        ben = self.here.find_person_by_name("Ben")
        said = [i.text for i in self.here.interactions_for_person(ben.id) if i.role == "source"]
        self.assertEqual(said, ["Bob is a scientist", "what is Bob?"])

    # ---------------------------------------------------------- merging
    def test_evidence_from_both_lands_on_one_belief(self):
        self.talk(self.here, "Ben", "Bob is a scientist")
        self.talk(self.there, "Dan", "bob is a Scientist")
        report = self.take_in()
        bob = [a for a in self.here.all_atoms() if a.subject.lower() == "bob"]
        self.assertEqual(len(bob), 1)
        self.assertEqual(sorted(self.person(self.here, p) for p in bob[0].provenance), ["Ben", "Dan"])
        self.assertEqual(bob[0].status, AtomStatus.CONFIRMED, "worked out again from both")
        self.assertEqual(report.beliefs_reinforced, 1)

    def test_rival_values_from_the_two_get_a_question_and_no_dispute(self):
        self.talk(self.here, "Ben", "Bob is a scientist")
        self.talk(self.there, "Dan", "Bob is a builder")
        report = self.take_in()
        for statement in ("Bob is a scientist", "Bob is a builder"):
            self.assertEqual([p.polarity for p in self.belief(self.here, statement).provenance], [1])
        self.assertEqual(report.rivalries_queued, 1)
        self.assertEqual([q["reason"] for q in self.here.pending_questions()], ["contradiction"])

    def test_rivals_that_came_over_together_bring_only_their_own_questions(self):
        self.talk(self.there, "Ben", "Bob is a scientist")
        self.talk(self.there, "Dan", "Bob is a builder")
        for q in self.there.pending_questions():
            self.there.mark_question_answered(q["id"])
        report = self.take_in()
        self.assertEqual(report.rivalries_queued, 0)
        self.assertEqual(self.here.pending_questions(), [])

    def test_a_link_made_there_reaches_evidence_held_here(self):
        self.talk(self.here, "Ben", "Felix is a black cat")
        self.talk(self.there, "Dan", "Felix is a black cat")
        self.talk(self.there, "Eve", "Felix is a cat")
        entailment.link(self.there, self.belief(self.there, "Felix is a black cat"),
                        self.belief(self.there, "Felix is a cat"))
        self.take_in()
        specific = self.belief(self.here, "Felix is a black cat")
        general = self.belief(self.here, "Felix is a cat")
        self.assertTrue(self.here.linked(specific.id, general.id))
        self.assertIn("Ben", [self.person(self.here, p) for p in general.evidence],
                      "support for the specific belief held here carries to the general one")

    # ----------------------------------------------------------- people
    def test_a_bare_name_on_both_sides_is_one_person(self):
        self.talk(self.here, "Dan", "Bob is a scientist")
        self.talk(self.there, "Dan", "Bob is a builder")
        report = self.take_in()
        self.assertEqual(report.people_matched, 1)
        self.assertEqual([p.name for p in self.here.all_people()], ["Dan"])

    def test_a_name_with_a_secret_word_comes_over_unverified(self):
        self.talk(self.here, "Dan", secret="blue whale")
        self.talk(self.there, "Dan", "Dan is a physicist", "Bob is a builder")
        report = self.take_in()
        self.assertEqual(report.people_unverified, 1)
        builder = self.belief(self.here, "Bob is a builder")
        self.assertEqual(self.person(self.here, builder.provenance[0]), "Dan (unverified)")
        self.assertIsNone(self.belief(self.here, "Dan is a physicist"),
                          "a claim about themselves by someone not shown to be them")
        self.assertEqual(report.self_claims_left, 1)

    def test_a_new_name_keeps_its_secret_word(self):
        self.talk(self.there, "Eve", "Bob is a builder", secret="blue whale")
        self.take_in()
        eve = self.here.find_person_by_name("Eve")
        self.assertTrue(self.here.verify_person_secret(eve, "Blue Whale"))
        engine = ChloeEngine(self.here)
        self.assertIn("Welcome back", engine.greet("Eve"))

    def test_a_person_known_here_keeps_the_trust_held_here(self):
        self.talk(self.here, "Dan", "Bob is a scientist")
        before = self.here.find_person_by_name("Dan").trust
        self.talk(self.there, "Ann", "Bob is a builder")
        self.talk(self.there, "Dan", "Bob is not a builder")
        self.take_in()
        self.assertEqual(self.here.find_person_by_name("Dan").trust, before)

    # ------------------------------------------------------- repetition
    def test_ingesting_again_adds_nothing(self):
        self.talk(self.here, "Ben", "Bob is a scientist")
        self.talk(self.there, "Dan", "Bob is a builder", "Felix is a cat")
        self.take_in()
        state, asked = _state(self.here), self.here.pending_questions()
        report = self.take_in()
        self.assertEqual((report.turns_added, report.evidence_added, report.questions_added), (0, 0, 0))
        self.assertEqual(_state(self.here), state)
        self.assertEqual(self.here.pending_questions(), asked)

    def test_a_later_snapshot_brings_only_what_is_new(self):
        self.talk(self.there, "Dan", "Bob is a builder")
        self.take_in()
        self.talk(self.there, "Eve", "Bob is a builder")
        report = self.take_in()
        self.assertEqual(report.evidence_added, 1)
        self.assertEqual(len(self.belief(self.here, "Bob is a builder").provenance), 2)

    def test_a_copy_of_this_store_adds_nothing(self):
        self.talk(self.here, "Ben", "Bob is a scientist", "Felix is a cat")
        copy = os.path.join(self.dir, "copy.db")
        shutil.copy(self.here.path, copy)
        state = _state(self.here)
        report = ingest(self.here, copy)
        self.assertEqual((report.turns_added, report.evidence_added), (0, 0))
        self.assertEqual(_state(self.here), state)

    def test_two_copies_that_went_their_own_ways_merge_once(self):
        self.talk(self.here, "Ben", "Bob is a scientist")
        fork = os.path.join(self.dir, "fork.db")
        shutil.copy(self.here.path, fork)
        forked = KnowledgeStore(fork)
        self.stores.append(forked)
        self.talk(self.here, "Ann", "Bob is a scientist")
        self.talk(forked, "Dan", "Bob is a scientist")
        ingest(self.here, fork)
        sources = sorted(self.person(self.here, p) for p in self.belief(self.here, "Bob is a scientist").provenance)
        self.assertEqual(sources, ["Ann", "Ben", "Dan"])

    def test_a_core_that_passed_through_a_third_store_is_recognised(self):
        self.talk(self.there, "Dan", "Bob is a builder")
        third = self.store("third.db")
        self.talk(third, "Eve", "Bob is a builder")
        ingest(third, self.there.path)
        self.take_in()
        ingest(self.here, third.path)
        sources = sorted(self.person(self.here, p) for p in self.belief(self.here, "Bob is a builder").provenance)
        self.assertEqual(sources, ["Dan", "Eve"])

    def test_a_store_cannot_ingest_itself(self):
        with self.assertRaises(ValueError):
            ingest(self.here, self.here.path)

    # -------------------------------------------------------- the script
    def test_a_dry_run_changes_nothing(self):
        self.talk(self.there, "Dan", "Bob is a builder")
        self.here.close()
        with open(self.here.path, "rb") as f:
            before = f.read()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ingest_core.main([self.here.path, self.there.path, "--dry-run"]), 0)
        with open(self.here.path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_each_ingest_is_on_record(self):
        self.talk(self.there, "Dan", "Bob is a builder")
        self.take_in()
        (row,) = self.here.ingests()
        self.assertEqual(row["source"], self.there.path)
        self.assertEqual(row["summary"]["evidence_added"], 1)


if __name__ == "__main__":
    unittest.main()
