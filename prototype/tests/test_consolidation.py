"""
Tests for the sleep pass, and in particular for the LLM hypothesis
generator: what it keeps, what it refuses, and what it does when the server
is not there.

The model is stubbed throughout -- these check the contract around it, not
the model. Stdlib only, offline:

    python3 -m unittest tests.test_consolidation
"""

import json
import os
import tempfile
import unittest

from chloe import consolidation, llm_client, questions
from chloe.dialogue import ChloeEngine
from chloe.questions import REASONS, Kind
from chloe.models import Atom, AtomStatus, Provenance, RelationProperties
from chloe.storage import KnowledgeStore


def reply(hypotheses):
    return json.dumps({"hypotheses": hypotheses})


class StubbedModel:
    """Stands in for llm_client.chat, and makes is_configured() true."""

    def __init__(self, response):
        self.response = response
        self.requests = []

    def __call__(self, messages, **kwargs):
        self.requests.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    @property
    def facts_shown(self):
        return self.requests[-1][-1]["content"]


class ConsolidationTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.store = KnowledgeStore(self.path)
        self._env = {k: os.environ.get(k) for k in
                     ("VLLM_BASE_URL", "CHLOE_HYPOTHESIS_GROUNDED")}
        self._real_chat = llm_client.chat

    def tearDown(self):
        llm_client.chat = self._real_chat
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.store.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    # ------------------------------------------------------------ fixtures
    def atom(self, subject, relation, obj, scope="", status=AtomStatus.CANDIDATE,
             confidence=0.5):
        a = self.store.save_atom(Atom(id=None, subject=subject, relation=relation,
                                      object=obj, scope=scope, status=status,
                                      confidence=confidence))
        self.store.add_provenance(a.id, Provenance(person_id=1, interaction_id=1, polarity=1))
        return a

    def felix(self):
        return self.atom("Felix", "is", "a cat"), self.atom("Felix", "is", "black")

    def with_model(self, response):
        os.environ["VLLM_BASE_URL"] = "http://stub/v1"
        stub = StubbedModel(response)
        llm_client.chat = stub
        return stub

    def llm_questions(self):
        return [q for q in self.store.pending_questions() if q["reason"] == "llm_hypothesis"]

    def hypotheses(self):
        return [a for a in self.store.all_atoms() if a.status == AtomStatus.HYPOTHESIS]

    BLACK_CAT = {"subject": "Felix", "relation": "is", "object": "a black cat",
                 "scope": None, "sources": [1, 2], "confidence": 0.85}

    # ------------------------------------------------------------ merging
    def test_surface_variants_become_one_belief(self):
        os.environ.pop("VLLM_BASE_URL", None)
        kept = self.atom("Cora", "likes", "to cook")
        self.store.add_provenance(kept.id, Provenance(person_id=2, interaction_id=2, polarity=1))
        variant = self.atom("cora", "like", "to cook")
        self.store.queue_question("label", "weak_candidate_recheck", related_atom_id=variant.id)

        report = consolidation.sleep(self.store)

        atoms = self.store.all_atoms()
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0].statement(), "Cora likes to cook", "best-attested wording kept")
        self.assertEqual(len(atoms[0].provenance), 3)
        self.assertEqual(atoms[0].status, AtomStatus.CONFIRMED, "status worked out again")
        self.assertEqual(report.merged, 1)
        self.assertTrue(all(q["related_atom_id"] == atoms[0].id
                            for q in self.store.pending_questions()))

    # -------------------------------------------------------- implication
    def _said(self, name, line):
        e = ChloeEngine(self.store)
        e.greet(name)
        e.turn("no")
        e.turn(line)
        return e

    def _implication_model(self, *found):
        """Answers the implication prompt with `found`, and proposes no
        hypotheses."""
        def chat(messages, **kwargs):
            if "entails" in messages[0]["content"]:
                return json.dumps({"implications": list(found)})
            return json.dumps({"hypotheses": []})
        os.environ["VLLM_BASE_URL"] = "http://stub/v1"
        llm_client.chat = chat

    def _felix_black_cat(self):
        self._said("Ben", "Felix is a cat")
        bob = self._said("Bob", "Felix is a black cat")
        cat, black = sorted(self.store.all_atoms(), key=lambda a: a.id)
        self.assertEqual([p.polarity for p in cat.evidence], [1, -1], "taken for a rival at first")
        return cat, black, bob

    def test_a_value_that_implies_another_is_linked_not_disputed(self):
        cat, black, bob = self._felix_black_cat()
        self._implication_model({"pair": 1, "specific": "B", "confidence": 0.95})
        report = consolidation.sleep(self.store)

        self.assertEqual(report.implications_linked, 1)
        self.assertTrue(self.store.linked(black.id, cat.id))
        cat = self.store.atom_by_id(cat.id)
        voided = [p for p in cat.provenance if p.void]
        self.assertEqual(len(voided), 1, "the dispute stays in the record")
        self.assertEqual({(p.person_id, p.polarity) for p in cat.evidence},
                         {(1, 1), (bob.person.id, 1)}, "Bob's black cat now supports the cat")
        self.assertAlmostEqual(self.store.person_by_id(bob.person.id).trust_in(), 0.5)
        self.assertFalse([q for q in self.store.pending_questions() if q["reason"] == "contradiction"
                          and questions.still_open(self.store, q)])

    def test_a_denial_of_the_general_value_reaches_the_specific_one(self):
        cat, black, _ = self._felix_black_cat()
        self._implication_model({"pair": 1, "specific": "B", "confidence": 0.95})
        consolidation.sleep(self.store)
        os.environ.pop("VLLM_BASE_URL", None)
        eve = self._said("Eve", "Felix is not a cat")
        black = self.store.atom_by_id(black.id)
        self.assertIn((eve.person.id, -1), {(p.person_id, p.polarity) for p in black.evidence})

    def test_an_implication_the_words_do_not_carry_is_asked_not_made(self):
        self._felix_black_cat()
        self._implication_model({"pair": 1, "specific": "A", "confidence": 0.95})   # cat => black cat
        report = consolidation.sleep(self.store)
        self.assertEqual(report.implications_linked, 0)
        [q] = [q for q in self.store.pending_questions() if q["reason"] == "implication"]
        self.assertEqual(questions.render(self.store, q),
                         "If Felix is a cat, does that mean Felix is a black cat?")

    def test_the_models_own_knowledge_does_not_link_values(self):
        self._said("Ann", "Rex is a poodle")
        self._said("Tom", "Rex is a dog")
        self._implication_model({"pair": 1, "specific": "A", "confidence": 0.99})
        report = consolidation.sleep(self.store)
        self.assertEqual(report.implications_linked, 0)
        self.assertEqual([q["reason"] for q in self.store.pending_questions()
                          if q["other_atom_id"]], ["implication"])

    # ------------------------------------------- implication, confirmed by asking
    def _asked_about_rex(self, name="Eve"):
        self._said("Ann", "Rex is a poodle")
        self._said("Tom", "Rex is a dog")
        self._implication_model({"pair": 1, "specific": "A", "confidence": 0.99})
        consolidation.sleep(self.store)
        os.environ.pop("VLLM_BASE_URL", None)
        asker = ChloeEngine(self.store)
        asker.greet(name)
        asker.turn("no")
        asker.offer_questions()
        reply = asker.turn("yes")
        for _ in range(4):                  # step past any recheck of either value
            if "does that mean" in reply:
                break
            self.assertNotIn("which is it", reply, "the choice waits on the implication")
            reply = asker.turn("yes")
        self.assertIn("If Rex is a poodle, does that mean Rex is a dog?", reply)
        return asker

    def _rex(self, obj):
        return next(a for a in self.store.all_atoms() if a.object == obj)

    def test_yes_makes_the_link_and_records_who_said_so(self):
        eve = self._asked_about_rex()
        reply = eve.turn("yes")
        poodle, dog = self._rex("a poodle"), self._rex("a dog")
        self.assertIn("I'll take it that Rex is a dog whenever Rex is a poodle", reply)
        self.assertTrue(self.store.linked(poodle.id, dog.id))
        row = self.store.conn.execute("SELECT confirmed_by FROM entailments").fetchone()
        self.assertEqual(row["confirmed_by"], eve.person.id)
        self.assertFalse([p for p in dog.evidence if p.polarity < 0], "Tom's value is no rival now")

    def test_no_keeps_them_apart_and_is_not_asked_again(self):
        eve = self._asked_about_rex()
        self.assertIn("keep the two apart", eve.turn("no"))
        poodle, dog = self._rex("a poodle"), self._rex("a dog")
        self.assertFalse(self.store.linked(poodle.id, dog.id))
        self.assertTrue(self.store.judged(poodle.id, dog.id))
        self._implication_model({"pair": 1, "specific": "A", "confidence": 0.99})
        consolidation.sleep(self.store)
        self.assertFalse([q for q in self.store.pending_questions() if q["reason"] == "implication"])

    def test_which_is_it_is_asked_once_the_implication_is_refused(self):
        eve = self._asked_about_rex()
        reply = eve.turn("no")
        self.assertIn("keep the two apart", reply)
        for _ in range(4):
            if "which is it" in reply:
                break
            reply = eve.turn("yes")
        self.assertIn("whether Rex is a poodle or a dog -- which is it?", reply)

    def test_the_people_who_stated_the_values_may_be_asked(self):
        self._asked_about_rex(name="Ann")

    # -------------------------------------------------------- the good case
    def test_conjunction_the_rules_cannot_reach(self):
        self.felix()
        self.with_model(reply([self.BLACK_CAT]))
        report = consolidation.sleep(self.store)

        hyps = self.hypotheses()
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0].object, "a black cat")
        self.assertEqual(hyps[0].status, AtomStatus.HYPOTHESIS)
        self.assertEqual(report.hypotheses_generated, 1)

    def test_it_is_asked_before_it_is_believed(self):
        self.felix()
        self.with_model(reply([self.BLACK_CAT]))
        consolidation.sleep(self.store)

        questions = self.llm_questions()
        self.assertEqual(len(questions), 1)
        self.assertIn("Felix is a black cat", questions[0]["question"])
        self.assertEqual(questions[0]["related_atom_id"], self.hypotheses()[0].id)

    def test_confidence_never_exceeds_the_weakest_source(self):
        self.atom("Felix", "is", "a cat", confidence=0.9)
        self.atom("Felix", "is", "black", confidence=0.4)
        self.with_model(reply([dict(self.BLACK_CAT, confidence=0.99)]))
        consolidation.sleep(self.store)

        self.assertAlmostEqual(self.hypotheses()[0].confidence,
                               0.4 * consolidation.HYPOTHESIS_DAMPING, places=4)

    def test_hypotheses_are_not_built_on_hypotheses(self):
        self.felix()
        self.atom("Felix", "is", "a mouser", status=AtomStatus.HYPOTHESIS)
        stub = self.with_model(reply([]))
        consolidation.sleep(self.store)

        self.assertNotIn("mouser", stub.facts_shown)

    # ------------------------------------------------------------ refusals
    def test_a_word_from_nowhere_is_refused(self):
        self.felix()
        self.with_model(reply([{"subject": "Felix", "relation": "is", "object": "a mammal",
                                "scope": None, "sources": [1], "confidence": 0.95}]))
        report = consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])
        self.assertTrue(any("mammal" in d for d in report.details))

    def test_the_grounding_check_can_be_switched_off(self):
        self.felix()
        os.environ["CHLOE_HYPOTHESIS_GROUNDED"] = "0"
        self.with_model(reply([{"subject": "Felix", "relation": "is", "object": "a mammal",
                                "scope": None, "sources": [1], "confidence": 0.95}]))
        consolidation.sleep(self.store)

        self.assertEqual([a.object for a in self.hypotheses()], ["a mammal"])

    def test_a_plural_source_still_grounds_a_singular_proposal(self):
        self.atom("Felix", "is", "a cat")
        self.atom("cats", "are", "animals")
        self.with_model(reply([{"subject": "Felix", "relation": "is", "object": "an animal",
                                "scope": None, "sources": [1, 2], "confidence": 0.8}]))
        consolidation.sleep(self.store)

        self.assertEqual([a.object for a in self.hypotheses()], ["an animal"])

    def test_two_conditions_cannot_be_combined(self):
        self.atom("the sky", "is", "blue", scope="it is daytime")
        self.atom("the sky", "is", "grey", scope="it is overcast")
        self.with_model(reply([{"subject": "the sky", "relation": "is",
                                "object": "blue in daytime, and grey when overcast",
                                "scope": None, "sources": [1, 2], "confidence": 0.9}]))
        report = consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])
        self.assertTrue(any("more than one thing" in d for d in report.details))

    def test_a_compound_object_is_refused_even_under_one_condition(self):
        self.atom("Claire", "is", "an artist", scope="she is at work")
        self.atom("Claire", "is", "tired", scope="she is at work")
        self.with_model(reply([{"subject": "Claire", "relation": "is",
                                "object": "an artist and tired", "scope": "she is at work",
                                "sources": [1, 2], "confidence": 0.8}]))
        consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])

    def test_a_condition_in_the_object_is_refused(self):
        self.atom("the sky", "is", "blue", scope="it is daytime")
        self.atom("the sky", "is", "bright")
        self.with_model(reply([{"subject": "the sky", "relation": "is",
                                "object": "bright when it is daytime", "scope": None,
                                "sources": [1, 2], "confidence": 0.8}]))
        report = consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])
        self.assertTrue(any("condition in its object" in d for d in report.details))

    def test_a_shared_condition_is_carried(self):
        self.atom("Claire", "is", "an artist", scope="she is at work")
        self.atom("Claire", "is", "tired", scope="she is at work")
        self.with_model(reply([{"subject": "Claire", "relation": "is",
                                "object": "a tired artist", "scope": "she is at work",
                                "sources": [1, 2], "confidence": 0.8}]))
        consolidation.sleep(self.store)

        self.assertEqual([(a.object, a.scope) for a in self.hypotheses()],
                         [("a tired artist", "she is at work")])

    def test_dropping_a_condition_is_refused(self):
        self.atom("Claire", "is", "an artist", scope="she is at work")
        self.atom("Claire", "is", "tired", scope="she is at work")
        self.with_model(reply([{"subject": "Claire", "relation": "is",
                                "object": "a tired artist", "scope": None,
                                "sources": [1, 2], "confidence": 0.8}]))
        report = consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])
        self.assertTrue(any("drops or alters the condition" in d for d in report.details))

    def test_an_invented_condition_is_refused(self):
        self.felix()
        self.with_model(reply([dict(self.BLACK_CAT, scope="it is a cat")]))
        report = consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])
        self.assertTrue(any("nobody stated" in d for d in report.details))

    def test_an_uncited_proposal_is_refused(self):
        self.felix()
        self.with_model(reply([dict(self.BLACK_CAT, sources=[])]))
        report = consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])
        self.assertTrue(any("cites no facts" in d for d in report.details))

    def test_a_proposal_citing_something_never_shown_is_refused(self):
        self.felix()
        self.with_model(reply([dict(self.BLACK_CAT, sources=[999])]))
        report = consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])
        self.assertTrue(any("not shown" in d for d in report.details))

    def test_low_confidence_is_refused(self):
        self.felix()
        self.with_model(reply([dict(self.BLACK_CAT, confidence=0.2)]))
        consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])

    def test_restating_a_known_fact_is_refused(self):
        self.felix()
        self.with_model(reply([{"subject": "Felix", "relation": "is", "object": "a cat",
                                "scope": None, "sources": [1], "confidence": 0.99}]))
        report = consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])
        self.assertTrue(any("already on file" in d for d in report.details))

    def test_a_malformed_proposal_does_not_take_the_good_one_with_it(self):
        self.felix()
        self.with_model(reply([{"subject": "Felix"}, self.BLACK_CAT]))
        consolidation.sleep(self.store)

        self.assertEqual([a.object for a in self.hypotheses()], ["a black cat"])

    def test_the_queue_is_capped(self):
        for n in range(8):
            self.atom("Felix", "is", f"thing{n}")
        proposals = [{"subject": "Felix", "relation": "is", "object": f"thing{n} thing{n + 1}",
                      "scope": None, "sources": [n + 1, n + 2], "confidence": 0.8}
                     for n in range(7)]
        self.with_model(reply(proposals))
        consolidation.sleep(self.store)

        self.assertEqual(len(self.hypotheses()), consolidation.MAX_LLM_HYPOTHESES_PER_SLEEP)

    # --------------------------------------------------- when there is none
    def test_no_server_means_no_llm_pass(self):
        os.environ.pop("VLLM_BASE_URL", None)
        self.felix()
        report = consolidation.sleep(self.store)

        self.assertEqual(self.hypotheses(), [])
        self.assertEqual(report.hypotheses_generated, 0)

    def test_a_broken_exchange_does_not_break_sleep(self):
        self.felix()
        self.with_model(llm_client.LLMUnavailable("server down"))
        report = consolidation.sleep(self.store)

        self.assertIn("Sleep complete", report.summary())
        self.assertTrue(any("no hypotheses from the model" in d for d in report.details))

    def test_a_reply_that_is_not_json_does_not_break_sleep(self):
        self.felix()
        self.with_model("Sure! Here are some ideas about Felix.")
        report = consolidation.sleep(self.store)

        self.assertIn("Sleep complete", report.summary())
        self.assertEqual(self.hypotheses(), [])

    def test_the_declared_transitivity_pass_still_runs(self):
        self.store.declare_relation(RelationProperties(name="is", transitive=True))
        self.atom("Felix", "is", "a cat")
        self.atom("a cat", "is", "an animal")
        self.with_model(reply([]))
        consolidation.sleep(self.store)

        self.assertEqual([a.object for a in self.hypotheses()], ["an animal"])
        self.assertTrue([q for q in self.store.pending_questions() if q["reason"] == "hypothesis"])

    # ------------------------------------------------------- the wake side
    def test_a_yes_settles_it(self):
        self.felix()
        self.with_model(reply([self.BLACK_CAT]))
        consolidation.sleep(self.store)
        hyp_id = self.hypotheses()[0].id

        os.environ.pop("VLLM_BASE_URL", None)   # pattern parser for the dialogue
        dan = ChloeEngine(self.store)
        dan.greet("Dan")
        dan.turn("no")                          # decline the secret-word offer
        self.assertIsNotNone(dan.offer_questions())
        question = dan.turn("yes")              # consent to the questions
        for _ in range(6):                      # step past any earlier question
            if "Felix is a black cat" in question:
                break
            question = dan.turn("no")
        self.assertIn("Felix is a black cat", question)
        self.assertIn(REASONS["llm_hypothesis"].preamble, question)

        dan.turn("yes")
        settled = [a for a in self.store.all_atoms() if a.id == hyp_id][0]
        self.assertNotEqual(settled.status, AtomStatus.HYPOTHESIS)
        self.assertTrue(settled.provenance)

    def test_the_reason_is_answerable_by_yes_or_no(self):
        self.assertEqual(REASONS["llm_hypothesis"].kind, Kind.CONFIRM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
