"""
Tests for the inspection path: the interaction log a transcript is built
from, and inspect_db.py's read-only collectors.

Stdlib only, no server, no network:

    python3 -m unittest tests.test_inspect
"""

import os
import tempfile
import unittest

import inspect_db
from chloe.dialogue import CONVERSATION_ROLES, ROLE_EVIDENCE, ChloeEngine
from chloe.storage import KnowledgeStore


class InspectionTests(unittest.TestCase):
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

    def _turns(self, name="Dan"):
        person = self.store.find_person_by_name(name)
        return self.store.interactions_for_person(person.id)

    # ------------------------------------------------------------ the log
    def test_greeting_is_logged_once_and_verbatim(self):
        greeting = [i for i in self._turns() if i.role == "chloe"][0]
        self.assertTrue(greeting.text.startswith("Hi! I am Chloe."))
        self.assertEqual(sum(1 for i in self._turns() if i.text.startswith("Hi! I am Chloe.")), 1)

    def test_provenance_anchor_is_not_a_conversational_turn(self):
        self.dan.turn("the sky is blue")
        roles = [i.role for i in self._turns()]
        self.assertIn(ROLE_EVIDENCE, roles)
        spoken = [i.text for i in self._turns() if i.role in CONVERSATION_ROLES]
        # the person's own words appear once; the canonical statement doesn't
        self.assertEqual(spoken.count("the sky is blue"), 1)

    def test_transcript_is_only_what_was_said(self):
        self.dan.turn("the sky is blue")
        data = inspect_db.transcript(self.store, "Dan")
        self.assertTrue(all(t["role"] in CONVERSATION_ROLES for t in data["turns"]))
        self.assertEqual(data["person"], "Dan")

    def test_transcript_of_an_unknown_person_is_refused(self):
        with self.assertRaises(SystemExit):
            inspect_db.transcript(self.store, "Nobody")

    def test_interaction_by_id_round_trips(self):
        first = self._turns()[0]
        self.assertEqual(self.store.interaction_by_id(first.id).text, first.text)
        self.assertIsNone(self.store.interaction_by_id(999999))

    # ------------------------------------------------------- the collectors
    def test_beliefs_carry_named_evidence(self):
        self.dan.turn("the sky is blue")
        alice = ChloeEngine(self.store)
        alice.greet("Alice")
        alice.turn("no")
        alice.turn("the sky is green")

        data = inspect_db.collect(self.store, {"beliefs", "people", "questions"})
        sky = [b for b in data["beliefs"] if b["statement"] == "the sky is blue"][0]
        self.assertEqual(
            sorted((e["person"], e["effect"]) for e in sky["evidence"]),
            [("Alice", "disputes"), ("Dan", "supports")],
        )
        self.assertEqual(sorted(p["name"] for p in data["people"]), ["Alice", "Dan"])
        self.assertTrue(data["open_questions"])

    def test_beliefs_are_ordered_by_confidence(self):
        self.dan.turn("the sky is blue")
        self.dan.turn("grass is green")
        bob = ChloeEngine(self.store)
        bob.greet("Bob")
        bob.turn("no")
        bob.turn("grass is green")          # corroborates, so it should sort first
        confidences = [b["confidence"] for b in inspect_db.collect(self.store, {"beliefs"})["beliefs"]]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_collect_is_read_only(self):
        self.dan.turn("the sky is blue")
        before = [a.statement() for a in self.store.all_atoms()]
        inspect_db.collect(self.store, {"beliefs", "people", "questions"})
        inspect_db.transcript(self.store, "Dan")
        self.assertEqual([a.statement() for a in self.store.all_atoms()], before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
