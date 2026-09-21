"""
Tests for the sleep/wake question detour: the "Sleep, Chloe" command, the
consent step, how answers are routed onto the beliefs the questions hang
on, and what happens when someone declines or changes the subject.

Stdlib only, offline:

    python3 -m unittest tests.test_questions
"""

import os
import tempfile
import unittest

from chloe import questions
from chloe.dialogue import ChloeEngine, QuestionState
from chloe.models import AtomStatus, RelationProperties
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
                questions.kind_of(self.dan._current_question) != questions.Kind.CONFIRM:
            asked = self.dan.turn("the sky is blue")
        return asked

    def test_yes_supports_the_related_belief(self):
        self._reach_a_yes_no_question()
        q = self.dan._current_question
        before = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        reply = self.dan.turn("yes")
        after = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        self.assertIn("Good", reply)
        self.assertGreater(after.confidence, before.confidence)
        self.assertEqual(len(after.provenance), len(before.provenance) + 1)

    def test_no_is_an_answer_not_a_refusal(self):
        self._reach_a_yes_no_question()
        q = self.dan._current_question
        before = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        reply = self.dan.turn("no")
        after = [a for a in self.store.all_atoms() if a.id == q["related_atom_id"]][0]
        self.assertIn("think again", reply)
        self.assertLess(after.confidence, before.confidence)

    def test_answered_questions_do_not_come_back(self):
        self._reach_a_yes_no_question()
        qid = self.dan._current_question["id"]
        self.dan.turn("yes")
        self.assertNotIn(qid, [q["id"] for q in self.store.pending_questions()])

    def test_an_open_question_is_answered_by_a_statement(self):
        self._contradict()
        eve = self._person("Eve")          # has said nothing about the sky
        eve.turn("Sleep, Chloe")
        first = eve.turn("yes")
        self.assertIn("which is it?", first)
        qid = eve._current_question["id"]
        reply = eve.turn("the sky is blue")
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
        self.assertIn("worked that out myself", reply)
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


class OnTheTableTests(unittest.TestCase):
    """A reply is read against the question it answers; a question is
    checked and worded when it is asked, for the person it is put to."""

    def setUp(self):
        self.store = KnowledgeStore(":memory:")

    def _person(self, name):
        e = ChloeEngine(self.store)
        e.greet(name)
        e.turn("no")
        return e

    def _asked(self, engine):
        """Consent, and return the first question put."""
        self.assertIsNotNone(engine.offer_questions())
        return engine.turn("yes")

    def _conflict(self):
        self._person("Dario").turn("Hector is a dog")
        self._person("Bob").turn("Hector is a cat")

    # ------------------------------------------------------- reply words
    def test_a_bare_yes_does_not_answer_which_is_it(self):
        self._conflict()
        eve = self._person("Eve")
        self.assertIn("which is it?", self._asked(eve))
        atoms_before = len(self.store.all_atoms())
        qid = eve._current_question["id"]
        reply = eve.turn("yes")
        self.assertIn("more than a yes or no", reply)
        self.assertIn("a dog or a cat", reply)
        self.assertEqual(eve._current_question["id"], qid)
        self.assertIn(qid, [q["id"] for q in self.store.pending_questions()])
        self.assertEqual(len(self.store.all_atoms()), atoms_before)

    def test_a_bare_yes_or_no_with_nothing_asked_is_not_testimony(self):
        dan = self._person("Dan")
        for word in ["No", "yes", "no."]:
            dan.turn(word)
        self.assertEqual(self.store.all_atoms(), [])

    def test_yes_settles_an_unresolved_contradiction(self):
        self._person("Ben").turn("Bob is a scientist")
        for name in ["Alice", "Eve", "Carl"]:
            self._person(name).turn("Bob is not a scientist")
        dan = self._person("Dan")
        dan.turn("Sleep, Chloe")
        self.assertIn("Is it true that Bob is a scientist?", dan.turn("yes"))
        before = len(self.store.all_atoms()[0].provenance)
        dan.turn("no")
        self.assertEqual(len(self.store.all_atoms()[0].provenance), before + 1)

    def test_reply_words_are_read_whatever_their_case(self):
        self._person("Ben").turn("Felix is a cat")
        self._person("Ben").turn("Sleep, Chloe")
        eve = self._person("Eve")
        eve.offer_questions()
        self.assertEqual(eve.question_state, QuestionState.AWAITING_CONSENT)
        eve.turn("YES!")
        self.assertEqual(eve.question_state, QuestionState.AWAITING_ANSWER)
        before = len(self.store.all_atoms()[0].evidence)
        eve.turn("Yes.")
        self.assertEqual(len(self.store.all_atoms()[0].evidence), before + 1)

    def test_ask_more_takes_up_the_questions_after_stop(self):
        for line in ["Felix is a cat", "Claire is an artist"]:
            self._person("Ben").turn(line)
        self._person("Ben").turn("Sleep, Chloe")
        eve = self._person("Eve")
        self._asked(eve)
        eve.turn("stop")
        self.assertIsNone(eve.offer_questions())
        for phrasing in ["Ask more", "ask more questions, Chloe", "Ask something."]:
            self.assertEqual(parse(phrasing).extra.get("command"), "ask", phrasing)
        reply = eve.turn("Ask more")
        self.assertEqual(eve.question_state, QuestionState.AWAITING_ANSWER)
        self.assertIn("Is that right?", reply)

    def test_ask_me_with_nothing_left_says_so(self):
        self.assertIn("nothing to ask", self._person("Eve").turn("Ask me something"))

    # ------------------------------------------------------- who is asked
    def test_nobody_is_asked_what_they_have_already_answered(self):
        self._conflict()
        self.assertIsNone(self._person("Bob").offer_questions())
        self.assertIsNone(self._person("Dario").offer_questions())

    def test_you_once_told_me_is_said_only_to_the_source(self):
        self._person("Sam").turn("Cora is a cook")
        self._person("Dan").turn("Sleep, Chloe")
        self.assertIn("You once told me that Cora is a cook", self._asked(self._person("Sam")))
        self.assertIn("I've been told that Cora is a cook", self._asked(self._person("Bob")))

    def test_a_question_about_the_person_asked_is_put_to_them_as_you(self):
        self._person("Ben").turn("Bob is a scientist")
        self._person("Dan").turn("Sleep, Chloe")
        self.assertIn("that you are a scientist", self._asked(self._person("Bob")))

    # ---------------------------------------------------- still worth it?
    def test_a_settled_recheck_is_closed_not_asked(self):
        self._person("Sam").turn("Cora is a cook")
        self._person("Dan").turn("Sleep, Chloe")          # queues the recheck
        self._person("Bob").turn("Cora is a cook")        # ...which a second source settles
        self.assertIsNone(self._person("Eve").offer_questions())
        self.assertEqual(self.store.pending_questions(), [])

    def test_one_open_question_per_belief(self):
        self._person("Ben").turn("Bob is a scientist")
        self._person("Alice").turn("Bob is not a scientist")      # denial
        for name in ["Eve", "Carl"]:
            self._person(name).turn("Bob is not a scientist")
        dan = self._person("Dan")
        dan.turn("Sleep, Chloe")                                   # unresolved contradiction
        dan.turn("Sleep, Chloe")
        open_ = self.store.pending_questions()
        self.assertEqual(len(open_), 1, [q["question"] for q in open_])
        self.assertEqual(open_[0]["reason"], "unresolved_contradiction")

    def test_an_unknown_term_is_answered_by_saying_what_it_is(self):
        self._person("Ann").turn("what is a zorb?")
        eve = self._person("Eve")
        self.assertIn("What is a zorb?", self._asked(eve))
        eve.turn("a zorb is a ball")
        self.assertEqual(self.store.pending_questions(), [])

    # ----------------------------------------------------------- the log
    def test_every_turn_is_logged_once_and_as_said(self):
        self._person("Ben").turn("Felix is a cat")
        self._person("Ben").turn("Sleep, Chloe")
        eve = self._person("Eve")
        eve.offer_questions()
        replies = [eve.turn("yes"), eve.turn("no")]
        chloe = [i.text for i in self.store.interactions_for_person(eve.person.id) if i.role == "chloe"]
        for reply in replies:
            self.assertEqual(chloe.count(reply), 1, reply)
        for line in chloe:
            self.assertEqual(chloe.count(line), 1, f"logged more than once: {line!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
