"""
Tests for the output-side linguistic interface (persona.py): the prompt
carries no label a model can copy, and a reply that breaks the interface's
licence is rejected in favour of the engine's own text.

Stdlib only, offline:

    python3 -m unittest tests.test_naturalisation
"""

import os
import tempfile
import unittest

from chloe import persona
from chloe.dialogue import ChloeEngine
from chloe.models import Stance
from chloe.storage import KnowledgeStore

MESSAGE = "Claire is an artist and George is a doctor."
GROUND_TRUTH = "Could you tell me about Claire first, one thing at a time?"


class RequestTests(unittest.TestCase):
    def test_user_turn_carries_only_what_was_said(self):
        msgs = persona.naturalise_request(MESSAGE, GROUND_TRUTH)
        self.assertEqual(msgs[-1], {"role": "user", "content": MESSAGE})

    def test_no_copyable_label_anywhere_in_the_request(self):
        blob = " ".join(m["content"] for m in persona.naturalise_request(MESSAGE, GROUND_TRUTH))
        for marker in ("ground truth", "the person just said", "reply to the person"):
            self.assertNotIn(marker, blob.lower(), marker)

    def test_the_reply_to_say_is_in_the_system_turn(self):
        msgs = persona.naturalise_request(MESSAGE, GROUND_TRUTH)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn(GROUND_TRUTH, msgs[0]["content"])

    def test_the_request_carries_no_conversation_history(self):
        # The model words one given sentence; shown its own earlier
        # naturalisations it continues them instead.
        msgs = persona.naturalise_request(MESSAGE, GROUND_TRUTH)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class RejectionTests(unittest.TestCase):
    def _reason(self, reply, message=MESSAGE, gt=GROUND_TRUTH):
        return persona.rejection_reason(reply, message, gt)

    def test_leaked_instruction_text_is_rejected(self):
        reply = "Ground truth: could you tell me about Claire first?"
        self.assertIn("leaked instruction text", self._reason(reply))

    def test_a_name_the_core_did_not_supply_is_rejected(self):
        # The model settling, from its own knowledge, a question the core
        # deliberately left open -- through a channel with no provenance.
        self.assertIn("did not supply", self._reason(
            "I've corrected you, Glasgow is in Scotland.",
            message="Glasgow is in Wales.",
            gt="Hmm, that contradicts what I was told before. I'll ask around."))

    def test_an_invented_attribution_is_rejected(self):
        self.assertIn("did not supply", self._reason(
            "Hmm, that clashes with what Dan told me before.",
            message="Claire is not an artist.",
            gt="Hmm, that contradicts what I was told before. I'll ask around."))

    def test_an_invented_number_is_rejected(self):
        self.assertIn("did not supply", self._reason(
            "I'm well -- 12 things on file just now.",
            message="How are you, Chloe?",
            gt="I'm well, thank you -- 7 things on file at the moment."))

    def test_names_the_core_supplied_are_allowed(self):
        self.assertIsNone(self._reason(
            "Got it -- Claire is an artist, noted.",
            message="Claire is an artist.",
            gt="Okay, I'll remember that Claire is an artist."))

    def test_swapping_the_speakers_is_rejected(self):
        self.assertIn("swapped the speakers", self._reason(
            "I'm Sam, what would you like to discuss?",
            message="Who am I?", gt="Only that you're Sam -- who are you?"))

    def test_keeping_the_speakers_straight_is_allowed(self):
        self.assertIsNone(self._reason(
            "You're Sam -- and who am I talking to beyond that?",
            message="Who am I?", gt="Only that you're Sam -- who are you?"))

    def test_echoing_the_person_is_rejected(self):
        reply = ("Claire is not an artist. Hmm, that contradicts what I was told "
                 "before. I'll flag it and ask around.")
        self.assertIn("echoed", self._reason(reply, message="Claire is not an artist.",
                                             gt="Hmm, that contradicts what I was told before."))

    def test_quoting_the_person_mid_reply_is_rejected(self):
        reply = ('The conversation was: "Chloe, sleep." Here is what I know now: '
                 "Sleep complete.")
        self.assertIn("echoed", self._reason(reply, message="Chloe, sleep.",
                                             gt="Sleep complete."))

    GT_DENIAL = ("Okay -- that conflicts with what I believed. "
                 "I'll double check: Claire is an artist.")

    def test_a_reply_may_not_turn_the_core_belief_around(self):
        # observed live: the model kept the core's sentence and flipped its
        # polarity, so the visitor was told Chloe would double-check the
        # opposite of the atom she holds
        self.assertIn("polarity", self._reason(
            "Okay -- that conflicts with what I believed. "
            "I'll double check: Claire is not an artist.",
            message="Claire is not an artist.", gt=self.GT_DENIAL))

    def test_quoting_the_denial_while_keeping_the_belief_is_allowed(self):
        self.assertIsNone(self._reason(
            "Got it, you say Claire is not an artist -- but I had Claire is "
            "an artist, so I'll double check.",
            message="Claire is not an artist.", gt=self.GT_DENIAL))

    def test_dropping_the_atom_from_a_denial_reply_is_rejected(self):
        # only the denied polarity survives, so what Chloe believes is gone
        self.assertIn("polarity", self._reason(
            "Got it -- you say Claire is not an artist. That conflicts with "
            "what I had; I'll double check.",
            message="Claire is not an artist.", gt=self.GT_DENIAL))

    def test_a_denial_may_name_the_atom_the_core_named(self):
        # both are the same proposition in opposite polarity, so saying it
        # plainly is phrasing, not parroting
        self.assertIsNone(self._reason(
            "You say Claire is not an artist, but I had Claire is an artist "
            "-- I'll double check.",
            message="Claire is not an artist.", gt=self.GT_DENIAL))

    def test_a_denial_the_core_never_mentioned_is_still_an_echo(self):
        self.assertIn("echoed", self._reason(
            "Claire is not an artist. Noted.",
            message="Claire is not an artist.",
            gt="Hmm, that contradicts what I was told before."))

    def test_empty_is_rejected(self):
        self.assertIsNotNone(self._reason(""))
        self.assertIsNotNone(self._reason("   \n "))

    def test_runaway_length_is_rejected(self):
        self.assertIn("far longer", self._reason("word " * 400))

    def test_a_good_naturalisation_passes(self):
        self.assertIsNone(self._reason("Could you tell me about Claire on her own first?"))

    def test_short_messages_do_not_trigger_the_echo_test(self):
        # "Noted..." begins with "no"; "Yesterday..." begins with "yes"
        self.assertIsNone(self._reason("Noted, thanks for telling me.", message="no"))
        self.assertIsNone(self._reason("Yesterday you said otherwise.", message="yes"))

    def test_echo_needs_a_word_boundary(self):
        # the reply merely starts with a longer word, not a repetition
        self.assertIsNone(self._reason("Greenhouses are warm indeed.", message="greenhouse"))

    def test_every_marker_is_caught(self):
        for marker in persona.SCAFFOLD_MARKERS:
            self.assertIsNotNone(self._reason(f"Sure. {marker} something."), marker)


class StanceTests(unittest.TestCase):
    """The core's yes/no answer must survive being worded.

    The live case: asked "Is Hector an animal?" with only "Hector is a dog"
    on file, the engine answers no. A model that knows dogs are animals
    answered yes, and every other check passed it -- "animal" is not a name
    or a number, and the polarity check looks for the person's words inside
    the core's reply, which are not there.
    """

    MESSAGE = "Is Hector an animal?"
    DENIED = "I don't think so -- I believe Hector is a dog instead (confidence 0.63)."
    UNKNOWN = "I don't know. What is Hector?"

    def test_the_observed_reversal_is_rejected(self):
        self.assertIn("answered 'affirm'", persona.rejection_reason(
            "Yes, Hector is an animal, specifically a dog.",
            self.MESSAGE, self.DENIED, stance=Stance.DENY))

    def test_the_other_checks_still_miss_it(self):
        # Without the stance, nothing else catches it -- which is why this
        # guard exists rather than a wider denylist.
        self.assertIsNone(persona.rejection_reason(
            "Yes, Hector is an animal, specifically a dog.",
            self.MESSAGE, self.DENIED))

    def test_a_faithful_denial_passes(self):
        self.assertIsNone(persona.rejection_reason(
            "I don't think so -- I have Hector down as a dog.",
            self.MESSAGE, self.DENIED, stance=Stance.DENY))

    def test_an_unknown_may_not_come_back_as_an_answer(self):
        self.assertIsNotNone(persona.rejection_reason(
            "Yes, a Hector is a kind of animal.",
            self.MESSAGE, self.UNKNOWN, stance=Stance.UNKNOWN))
        self.assertIsNone(persona.rejection_reason(
            "I don't have that yet -- what is Hector?",
            self.MESSAGE, self.UNKNOWN, stance=Stance.UNKNOWN))

    def test_a_reply_that_does_not_open_with_an_answer_is_left_alone(self):
        # The guard reads the opening only; a reply that leads with the
        # belief rather than a yes or no is judged by the other checks.
        self.assertIsNone(persona.rejection_reason(
            "Hector is a dog, as far as I have been told.",
            self.MESSAGE, self.DENIED, stance=Stance.DENY))


class EngineStanceTests(unittest.TestCase):
    """Where the stance the guard compares against comes from."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.store = KnowledgeStore(self.path)
        self.dan = ChloeEngine(self.store)
        self.dan.greet("Dan")
        self.dan.turn("no")                  # decline the secret-word offer
        self.dan.turn("Hector is a dog")

    def tearDown(self):
        self.store.close()
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_a_different_value_on_file_is_a_denial(self):
        self.dan.turn("Is Hector an animal?")
        self.assertEqual(self.dan.last_stance, Stance.DENY)

    def test_a_match_is_an_affirmation(self):
        self.dan.turn("Is Hector a dog?")
        self.assertEqual(self.dan.last_stance, Stance.AFFIRM)

    def test_nothing_on_file_is_neither(self):
        self.dan.turn("Is Rex a dog?")
        self.assertEqual(self.dan.last_stance, Stance.UNKNOWN)

    def test_a_turn_that_is_not_a_yes_no_question_carries_no_stance(self):
        self.dan.turn("Is Hector a dog?")
        self.dan.turn("Claire is an artist")
        self.assertIsNone(self.dan.last_stance, "a stale stance must not carry over")


if __name__ == "__main__":
    unittest.main(verbosity=2)
