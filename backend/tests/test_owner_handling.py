"""
Tests for: Sir / Ma'am recognition (voice pitch + words), "ek minute"
(hold-on) handling, neutral goodbyes, and the numbering-phrase filter.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_config  # noqa: E402
from local.reply_stream import stream_sentences  # noqa: E402
from local.voice_gender import GenderTracker, gender_from_text, voiced_f0s  # noqa: E402
from pooja_cli import PoojaSession  # noqa: E402


def voice(f0: float, seconds: float = 1.2, sr: int = 16000) -> np.ndarray:
    """A voiced-speech-like signal: harmonic stack with a little vibrato, RMS ~3000."""
    t = np.arange(int(sr * seconds)) / sr
    phase = 2 * np.pi * (f0 * t + 2.0 * np.sin(2 * np.pi * 5 * t) / (2 * np.pi * 5) * 1.5)
    sig = sum(np.sin(h * phase) / h for h in range(1, 9))
    sig = sig / np.sqrt(np.mean(sig ** 2)) * 3000
    return sig.astype(np.int16)


class TestVoicePitchGender(unittest.TestCase):
    def test_female_pitch(self):
        t = GenderTracker()
        t.add_audio(voice(220))
        self.assertEqual(t.guess, "female")

    def test_male_pitch(self):
        t = GenderTracker()
        t.add_audio(voice(115))
        self.assertEqual(t.guess, "male")

    def test_ambiguous_pitch_makes_no_guess(self):
        t = GenderTracker()
        t.add_audio(voice(163))
        self.assertIsNone(t.guess)

    def test_too_little_voiced_audio_makes_no_guess(self):
        t = GenderTracker()
        t.add_audio(voice(220, seconds=0.15))
        self.assertIsNone(t.guess)

    def test_silence_and_noise_are_not_voiced(self):
        rng = np.random.default_rng(1)
        self.assertEqual(voiced_f0s((rng.standard_normal(16000) * 20).astype(np.int16)), [])

    def test_evidence_accumulates_across_turns(self):
        t = GenderTracker()
        for _ in range(3):
            t.add_audio(voice(215, seconds=0.2))
        self.assertEqual(t.guess, "female")


class TestGenderFromWords(unittest.TestCase):
    def test_first_person_feminine_verbs(self):
        self.assertEqual(gender_from_text("main abhi busy hoon, kaam kar rahi hoon"), "female")

    def test_first_person_masculine_verbs(self):
        self.assertEqual(gender_from_text("main bahar ja raha hoon"), "male")

    def test_addressing_pooja_does_not_count(self):
        # "aap kaun baat kar rahi ho" is about Pooja, not the owner
        self.assertIsNone(gender_from_text("aap kaun baat kar rahi ho"))

    def test_words_override_pitch(self):
        t = GenderTracker()
        t.add_audio(voice(115))
        t.add_text("main kaam kar rahi hoon")
        self.assertEqual(t.guess, "female")


class _Getter:
    def __init__(self, items):
        self.items = list(items)
        self.last_audio = None

    def __call__(self):
        if not self.items:
            return "bye"
        text, audio = self.items.pop(0)
        self.last_audio = audio
        return text


class FakeStreamLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_messages = None

    def chat_stream(self, messages):
        self.calls += 1
        self.last_messages = [dict(m) for m in messages]
        yield self.replies.pop(0) if self.replies else "Ji."


def _session(llm, getter):
    out = []
    session = PoojaSession(llm=llm, get_user_input=getter, deliver_reply=lambda t: out.append(t), conversational=True)
    return session, out


class TestAddress(unittest.TestCase):
    def test_woman_is_addressed_as_maam_in_fixed_lines_and_llm_notes(self):
        llm = FakeStreamLLM(["Ji, batati hoon."])
        getter = _Getter([("ye kaise kaam karta hai?", voice(225))])
        session, out = _session(llm, getter)
        session.run()
        self.assertEqual(session.honorific, "Ma'am")
        self.assertIn("Ma'am", llm.last_messages[1]["content"])
        # the fixed goodbye used after "bye" uses Ma'am, not Sir
        self.assertIn("Ma'am", out[-1])
        self.assertNotIn("Sir", out[-1])

    def test_man_stays_sir(self):
        llm = FakeStreamLLM(["Ji, batati hoon."])
        getter = _Getter([("ye kaise kaam karta hai?", voice(110))])
        session, out = _session(llm, getter)
        session.run()
        self.assertEqual(session.honorific, "Sir")

    def test_opening_does_not_assume_sir(self):
        self.assertNotIn("Sir", agent_config.script_line("opening"))


class TestHoldOn(unittest.TestCase):
    def test_ek_minute_stops_the_pitch_without_calling_the_llm(self):
        llm = FakeStreamLLM([])
        getter = _Getter([("Ek minute", None)])
        session, out = _session(llm, getter)
        session.run()
        self.assertEqual(llm.calls, 0)
        self.assertIn("boliye", out[1].lower() + out[-1].lower())

    def test_variants_are_recognised(self):
        for phrase in ("ek minute", "ruko", "wait", "hold on", "ek second", "one minute", "Ek minute ek minute"):
            self.assertTrue(PoojaSession._is_hold_on(phrase), phrase)

    def test_ek_minute_with_a_real_question_goes_to_the_llm(self):
        self.assertFalse(PoojaSession._is_hold_on("ek minute aap kaun baat kar rahe ho"))
        self.assertFalse(PoojaSession._is_hold_on("wait, what is the price?"))

    def test_normal_sentences_are_not_hold_on(self):
        self.assertFalse(PoojaSession._is_hold_on("mera PG full hai"))


class TestNeutralWishesAndFilter(unittest.TestCase):
    def test_goodbyes_do_not_guess_day_or_night(self):
        for key in ("closing_interested", "closing_not_interested", "closing_busy_or_declined"):
            text = agent_config.script_line(key).lower()
            self.assertNotIn("ratri", text, key)
            self.assertNotIn("din shubh", text, key)

    def test_greeting_is_never_afternoon_at_night(self):
        from unittest.mock import patch

        for hour, expected in ((2, "namaste"), (8, "good morning"), (14, "good afternoon"), (19, "good evening"), (23, "namaste")):
            with patch("agent_config.datetime") as dt:
                dt.datetime.now.return_value = SimpleNamespace(hour=hour)
                self.assertEqual(agent_config.greeting(), expected, hour)

    def test_number_ranges_are_spoken_as_ranges(self):
        from local.speech_text import speech_text

        self.assertEqual(speech_text("sirf 10-15 owners"), "sirf 10 se 15 owners")
        self.assertEqual(speech_text("3‑4 mahine"), "3 se 4 mahine")

    def test_numbering_phrases_are_removed_from_speech(self):
        out = list(stream_sentences(["Ji Sir, dusri baat ye hai ki abhi listing free hai."], []))
        self.assertEqual(out, ["Ji Sir, abhi listing free hai."])


if __name__ == "__main__":
    unittest.main()
