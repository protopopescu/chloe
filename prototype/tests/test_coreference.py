"""
Tests for coreference: the proposal the model makes, who may answer it, and
what moves once somebody has.

The model is stubbed -- these check the contract around it, not the model.
Stdlib only, offline:

    python3 -m unittest tests.test_coreference
"""

import json
import os
import tempfile
import unittest

from chloe import consolidation, llm_client, questions
from chloe.dialogue import ChloeEngine
from chloe.models import Atom, AtomStatus, Person, Provenance
from chloe.storage import KnowledgeStore

CORA = "Cora"
CLAIMANT = "Cora" + Person.UNVERIFIED_SUFFIX


def reply(same):
    return json.dumps({"same": same})


class StubbedModel:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def __call__(self, messages, **kwargs):
        self.requests.append(messages)
        return self.response

    @property
    def shown(self):
        return self.requests[-1][-1]["content"]


class CoreferenceTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.store = KnowledgeStore(self.path)
        self._url = os.environ.get("VLLM_BASE_URL")
        self._real_chat = llm_client.chat

    def tearDown(self):
        llm_client.chat = self._real_chat
        if self._url is None:
            os.environ.pop("VLLM_BASE_URL", None)
        else:
            os.environ["VLLM_BASE_URL"] = self._url
        self.store.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    # ------------------------------------------------------------ fixtures
    def atom(self, subject, relation, obj, status=AtomStatus.CANDIDATE, person_id=1):
        a = self.store.save_atom(Atom(id=None, subject=subject, relation=relation,
                                      object=obj, status=status, confidence=0.6))
        if status != AtomStatus.HYPOTHESIS:
            self.store.add_provenance(a.id, Provenance(person_id=person_id, interaction_id=1,
                                                       polarity=1))
        return self.store.atom_by_id(a.id)

    def two_coras(self):
        self.atom(CLAIMANT, "is", "a computer scientist")
        self.atom(CORA, "is", "John's friend")

    def with_model(self, response):
        os.environ["VLLM_BASE_URL"] = "http://stub/v1"
        stub = StubbedModel(response)
        llm_client.chat = stub
        return stub

    def coreference_atom(self):
        return next(a for a in self.store.all_atoms()
                    if a.subject == CLAIMANT and a.object == CORA)

    # ------------------------------------------------------------ proposing
    def test_a_shared_word_makes_a_pair_worth_asking_about(self):
        self.two_coras()
        self.atom("Felix", "is", "a cat")
        pairs = consolidation._coreference_candidates(self.store)
        self.assertEqual(pairs, [(CORA, CLAIMANT)], "Felix shares no word with either")

    def test_a_pair_already_on_file_is_not_proposed_again(self):
        self.two_coras()
        self.atom(CLAIMANT, "is", CORA)
        self.assertEqual(consolidation._coreference_candidates(self.store), [])

    def test_a_proposal_becomes_a_hypothesis_and_a_question(self):
        self.two_coras()
        self.with_model(reply([{"pair": 1, "canonical": "A", "confidence": 0.8}]))
        consolidation._propose_coreference(self.store, consolidation.SleepReport())

        atom = self.coreference_atom()
        self.assertEqual(atom.status, AtomStatus.HYPOTHESIS, "nothing is believed yet")
        self.assertEqual([q["reason"] for q in self.store.pending_questions()], ["coreference"])

    def test_nothing_moves_on_the_model_s_word_alone(self):
        self.two_coras()
        self.with_model(reply([{"pair": 1, "canonical": "A", "confidence": 0.9}]))
        consolidation.sleep(self.store)
        self.assertTrue(any(a.subject == CLAIMANT for a in self.store.all_atoms()),
                        "the claimant's beliefs stay where they are until someone answers")

    def test_a_proposal_outside_the_pairs_shown_is_refused(self):
        self.two_coras()
        for item in ({"pair": 9, "canonical": "A", "confidence": 0.9},
                     {"pair": 1, "canonical": "C", "confidence": 0.9},
                     {"pair": 1, "canonical": "A", "confidence": 0.1}):
            pairs = consolidation._coreference_candidates(self.store)
            proposal, refusal = consolidation._validated_coreference(item, pairs)
            self.assertIsNone(proposal)
            self.assertTrue(refusal)

    # ---------------------------------------------------------------- gate
    def test_the_claimant_is_not_asked_under_either_name(self):
        self.two_coras()
        hyp = self.atom(CLAIMANT, "is", CORA, status=AtomStatus.HYPOTHESIS)
        questions.queue(self.store, "coreference", hyp)
        q = self.store.pending_questions()[0]

        claimant = self.store.get_or_create_person(CLAIMANT)
        real = self.store.get_or_create_person(CORA)
        third = self.store.get_or_create_person("Victor")

        self.assertFalse(questions.for_person(self.store, q, claimant))
        self.assertFalse(questions.for_person(self.store, q, real))
        self.assertTrue(questions.for_person(self.store, q, third))

    # --------------------------------------------------------------- merge
    def test_only_someone_else_can_confirm_it(self):
        self.two_coras()
        hyp = self.atom(CLAIMANT, "is", CORA, status=AtomStatus.HYPOTHESIS)
        self.store.queue_question("label", "coreference", related_atom_id=hyp.id)

        claimant = self.store.get_or_create_person(CLAIMANT)
        self.store.add_provenance(hyp.id, Provenance(person_id=claimant.id, interaction_id=1,
                                                     polarity=1))
        consolidation._apply_coreference(self.store, consolidation.SleepReport())

        self.assertTrue(any(a.subject == CLAIMANT and a.object == "a computer scientist"
                            for a in self.store.all_atoms()),
                        "their own word does not settle which self they are")

    def test_a_yes_moves_the_beliefs(self):
        self.two_coras()
        hyp = self.atom(CLAIMANT, "is", CORA, status=AtomStatus.HYPOTHESIS)
        self.store.queue_question("label", "coreference", related_atom_id=hyp.id)

        victor = ChloeEngine(self.store)
        victor.greet("Victor")
        victor._record(CLAIMANT, "is", CORA, "", +1)
        consolidation.sleep(self.store)

        self.assertEqual([a.statement() for a in self.store.find_atoms_about(CORA)
                          if a.object == "a computer scientist"],
                         ["Cora is a computer scientist"], "the belief moved across")

    def test_a_no_moves_nothing(self):
        self.two_coras()
        hyp = self.atom(CLAIMANT, "is", CORA, status=AtomStatus.HYPOTHESIS)
        self.store.queue_question("label", "coreference", related_atom_id=hyp.id)

        victor = ChloeEngine(self.store)
        victor.greet("Victor")
        victor._record(CLAIMANT, "is", CORA, "", -1)
        consolidation.sleep(self.store)

        self.assertTrue(any(a.subject == CLAIMANT and a.object == "a computer scientist"
                            for a in self.store.all_atoms()))

    # ------------------------------------- an unconfirmed name is not material
    def test_an_unconfirmed_name_is_kept_out_of_the_hypothesis_pass(self):
        self.two_coras()
        self.atom(CORA, "is", "a cook")
        stub = self.with_model(json.dumps({"hypotheses": []}))
        consolidation._generate_hypotheses_llm(self.store, consolidation.SleepReport())
        self.assertNotIn(Person.UNVERIFIED_SUFFIX.strip(), stub.shown)


if __name__ == "__main__":
    unittest.main()
