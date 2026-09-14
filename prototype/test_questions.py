"""
Tests for the sleep/wake question detour: the "Sleep, Chloe" command, the
consent step, how answers are routed onto the beliefs the questions hang
on, and what happens when someone declines or changes the subject.

Stdlib only, offline:

    python3 test_questions.py
"""

import os
import tempfile
import unittest

from chloe.dialogue import ChloeEngine, QuestionState, YES_NO_REASONS
from chloe.models import RelationProperties
from chloe.nlu import parse
from chloe.storage import KnowledgeStore


class QuestionModeTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.store = KnowledgeStore(self.path)
        self.store.declare_relation(RelationProperties(name="is", transitive=True))
        self.dan = self._person("Dan")
        for line in ["the sky is blue", "Felix is a cat", "a cat is an animal"]:
            self.dan.turn(line)

    def tearDown(self):
        self.store.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _person(self, name):
        e = ChloeEngine(self.store)
        e.greet(name)
        e.turn("no")          # decline the secret-word offer
        return e

    def _contradict(self):
        alice = self._person("Alice")
        alice.turn("the sky is green")

    # ------------------------------------------------------------- command
    def test_sleep_chloe_is_a_command(self):
        for phrasing in ["Sleep, Chloe", "sleep chloe", "Sleep", "Go to sleep, Chloe"]:
            self.assertEqual(parse(phrasing).extra.get("command"), "sleep", phrasing)

    def test_sleep_consolidates_and_offers(self):
        reply = self.dan.turn("Sleep, Chloe")
        self.assertIn("Sleep complete", reply)
        self.assertIn("Can I ask you some questions?", reply)
        self.assertEqual(self.dan.question_state, QuestionState.AWAITING_CONSENT)

    def test_no_offer_when_nothing_is_open(self):
        empty = ChloeEngine(KnowledgeStore(":memory:"))
        empty.greet("Solo")
        empty.turn("no")
        self.assertIsNone(empty.offer_questions())

    # ------------------------------------------------------------- consent
    def test_declining_is_honoured_for_the_session(self):
        self.dan.turn("Sleep, Chloe")
        self.assertIn("keep them to myself", self.dan.turn("no thanks"))
        self.assertEqual(self.dan.question_state, QuestionState.NONE)
        self.assertIsNone(self.dan.offer_questions())

    def test_changing_the_subject_is_not_an_answer(self):
        self.dan.turn("Sleep, Chloe")
        reply = self.dan.turn("grass is green")
        self.assertIn("I'll remember that grass is green", reply)
        self.assertIsNone(self.dan.offer_questions())   # it doesn't nag

    # ------------------------------------------------------------- answers
    def _reach_a_yes_no_question(self):
        self.dan.turn("Sleep, Chloe")
        asked = self.dan.turn("yes")
        while self.dan._current_question and \
                self.dan._current_question["reason"] not in YES_NO_REASONS:
            asked = self.dan.turn("the sky is blue")
        return asked

    def test_yes_supports_the_related_belief(self):
        self._reach_a_yes_no_question()
        q = self.dan._current_question
        before = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        reply = self.dan.turn("yes")
        after = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        self.assertIn("that supports", reply)
        self.assertGreater(after.confidence, before.confidence)
        self.assertEqual(len(after.provenance), len(before.provenance) + 1)

    def test_no_is_an_answer_not_a_refusal(self):
        self._reach_a_yes_no_question()
        q = self.dan._current_question
        before = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        reply = self.dan.turn("no")
        after = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        self.assertIn("disputing", reply)
        self.assertLess(after.confidence, before.confidence)

    def test_answered_questions_do_not_come_back(self):
        self._reach_a_yes_no_question()
        qid = self.dan._current_question["id"]
        self.dan.turn("yes")
        self.assertNotIn(qid, [q["id"] for q in self.store.pending_questions()])

    def test_an_open_question_is_answered_by_a_statement(self):
        self._contradict()
        self.dan.turn("Sleep, Chloe")
        first = self.dan.turn("yes")
        self.assertIn("which is it?", first)
        qid = self.dan._current_question["id"]
        reply = self.dan.turn("the sky is blue")
        self.assertIn("matches what I already believed", reply)
        self.assertNotIn(qid, [q["id"] for q in self.store.pending_questions()])

    def test_a_yes_no_question_answered_in_their_own_words(self):
        # "Claire is an artist, indeed" settles the question as surely as
        # "yes": it closes, the statement is recorded by the ordinary path,
        # and the next question follows in the same reply.
        asked = self._reach_a_yes_no_question()
        q = self.dan._current_question
        atom = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        self.assertIn(atom.subject.lower(), asked.lower())
        before = atom.confidence
        reply = self.dan.turn(f"{atom.subject} {atom.relation} {atom.object}")
        after = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        self.assertIn("matches what I already believed", reply)
        self.assertGreater(after.confidence, before)
        self.assertNotIn(q["id"], [p["id"] for p in self.store.pending_questions()])
        self.assertFalse(self.dan._questions_declined, "answering is not declining")

    def test_a_statement_about_something_else_still_ends_it(self):
        self._reach_a_yes_no_question()
        qid = self.dan._current_question["id"]
        reply = self.dan.turn("grass is green")
        self.assertIn("I'll remember that grass is green", reply)
        self.assertTrue(self.dan._questions_declined)
        self.assertIn(qid, [p["id"] for p in self.store.pending_questions()],
                      "a change of subject leaves the question open")

    def test_stop_ends_it_and_leaves_the_question_open(self):
        self._reach_a_yes_no_question()
        qid = self.dan._current_question["id"]
        self.assertIn("I'll stop there", self.dan.turn("stop"))
        self.assertEqual(self.dan.question_state, QuestionState.NONE)
        still_open = [q for q in self.store.pending_questions() if q["id"] == qid]
        self.assertTrue(still_open, "an unanswered question must stay open")
        self.assertEqual(still_open[0]["asked"], 1)

    def test_an_asked_but_unanswered_question_comes_round_again(self):
        self._reach_a_yes_no_question()
        qid = self.dan._current_question["id"]
        self.dan.turn("stop")
        # everything else answered or exhausted, this one is still owed
        for q in self.store.pending_questions():
            if q["id"] != qid:
                self.store.mark_question_answered(q["id"])
        self.assertEqual(self.store.next_question()["id"], qid)

    def test_stop_command_works_outside_the_detour(self):
        self.assertIn("stop asking", self.dan.turn("stop"))
        self.assertIsNone(self.dan.offer_questions())

    # ------------------------------------------------------------ dedupe
    def test_open_questions_are_not_duplicated(self):
        self._contradict()
        before = len(self.store.pending_questions())
        self.dan.turn("Sleep, Chloe")
        self.dan.turn("Sleep, Chloe")
        counts = {}
        for q in self.store.pending_questions():
            counts[q["question"]] = counts.get(q["question"], 0) + 1
        self.assertTrue(all(n == 1 for n in counts.values()), counts)
        self.assertGreaterEqual(len(self.store.pending_questions()), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
