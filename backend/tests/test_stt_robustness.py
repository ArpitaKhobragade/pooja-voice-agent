"""
MOCKED tests for STT/audio robustness: the suspicious/duplicate-transcript
heuristic (pooja_cli.py), the pure PCM signal analysis used to detect
silence/clipping/short clips (local/audio_io.py's analyze_pcm), device
name/index resolution, and the bilingual yes/no/maybe normalization
words added for this round of fixes. No real microphone, no network.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversation_manager import ConversationManager  # noqa: E402
from local import audio_io  # noqa: E402
from pooja_cli import _transcript_is_unreliable  # noqa: E402


class TestTranscriptReliabilityHeuristic(unittest.TestCase):
    def test_normal_short_reply_is_trusted(self):
        unreliable, reason = _transcript_is_unreliable("Haan ji bataiye", 1.4, None)
        self.assertFalse(unreliable)

    def test_genuine_one_word_short_clip_is_trusted(self):
        # "haan" said quickly (short clip, short transcript) is plausible
        # and should NOT be flagged — only a mismatch (long clip, one word)
        # is suspicious.
        unreliable, reason = _transcript_is_unreliable("haan", 0.6, None)
        self.assertFalse(unreliable)

    def test_normal_one_word_answer_with_trailing_silence_is_trusted(self):
        # Clips include ~1s of trailing silence, so "okay" records as ~1.8s.
        unreliable, _ = _transcript_is_unreliable("okay", 1.8, None)
        self.assertFalse(unreliable)

    def test_one_word_from_a_very_long_clip_is_suspicious(self):
        unreliable, reason = _transcript_is_unreliable("okay", 6.0, None)
        self.assertTrue(unreliable)
        self.assertIn("short transcript", reason)

    def test_repeated_short_answer_is_not_flagged(self):
        unreliable, _ = _transcript_is_unreliable("okay", 1.5, "okay")
        self.assertFalse(unreliable)

    def test_exact_duplicate_transcript_is_flagged(self):
        unreliable, reason = _transcript_is_unreliable("mera PG full hai", 1.2, "mera PG full hai")
        self.assertTrue(unreliable)
        self.assertIn("same transcript", reason)

    def test_duplicate_check_is_case_insensitive(self):
        unreliable, _ = _transcript_is_unreliable("Mera PG Full Hai", 1.2, "mera pg full hai")
        self.assertTrue(unreliable)

    def test_different_consecutive_transcripts_not_flagged_as_duplicate(self):
        unreliable, _ = _transcript_is_unreliable("nahi interested", 1.2, "haan ji bataiye")
        self.assertFalse(unreliable)

    def test_first_turn_has_no_previous_transcript_to_duplicate(self):
        # A short clip (not >1.0s, so it doesn't also trip the suspicious
        # single-word check) with no prior transcript to compare against.
        unreliable, _ = _transcript_is_unreliable("haan", 0.5, None)
        self.assertFalse(unreliable)


class TestPCMSignalAnalysis(unittest.TestCase):
    def test_silence_is_detected(self):
        pcm = np.zeros(16000, dtype=np.int16)  # 1s of true silence
        a = audio_io.analyze_pcm(pcm)
        self.assertTrue(a["is_silent"])
        self.assertFalse(a["is_clipped"])
        self.assertAlmostEqual(a["duration_s"], 1.0, places=2)

    def test_normal_speech_level_is_not_silent_or_clipped(self):
        rng = np.random.default_rng(0)
        pcm = (rng.standard_normal(16000) * 3000).astype(np.int16)  # moderate level
        a = audio_io.analyze_pcm(pcm)
        self.assertFalse(a["is_silent"])
        self.assertFalse(a["is_clipped"])

    def test_clipping_is_detected(self):
        pcm = np.full(16000, 32700, dtype=np.int16)  # pinned near full-scale
        a = audio_io.analyze_pcm(pcm)
        self.assertTrue(a["is_clipped"])

    def test_short_clip_flagged_but_not_silent_if_loud_enough(self):
        rng = np.random.default_rng(1)
        pcm = (rng.standard_normal(int(16000 * 0.1)) * 3000).astype(np.int16)  # 0.1s, loud
        a = audio_io.analyze_pcm(pcm)
        self.assertTrue(a["too_short"])
        self.assertFalse(a["is_silent"])

    def test_empty_clip_is_silent_and_too_short(self):
        a = audio_io.analyze_pcm(np.array([], dtype=np.int16))
        self.assertTrue(a["is_silent"])
        self.assertTrue(a["too_short"])
        self.assertEqual(a["duration_s"], 0.0)


class TestDeviceResolution(unittest.TestCase):
    def test_none_spec_resolves_to_none(self):
        self.assertIsNone(audio_io._resolve_device(None))

    def test_digit_spec_resolves_to_int_index(self):
        self.assertEqual(audio_io._resolve_device("3"), 3)

    def test_name_substring_resolves_case_insensitively(self):
        fake_devices = [
            {"name": "Microsoft Sound Mapper"},
            {"name": "USB Microphone (Realtek)"},
            {"name": "Speakers"},
        ]
        with patch("local.audio_io.sd.query_devices", return_value=fake_devices):
            idx = audio_io._resolve_device("usb microphone")
        self.assertEqual(idx, 1)

    def test_unmatched_name_raises_clear_error(self):
        fake_devices = [{"name": "Speakers"}]
        with patch("local.audio_io.sd.query_devices", return_value=fake_devices):
            with self.assertRaises(audio_io.AudioDeviceError):
                audio_io._resolve_device("nonexistent device xyz")


class TestExpandedYesNoVocabulary(unittest.TestCase):
    """The failure report specifically listed these words as real owner
    replies that should normalize cleanly: Haan, nahi, achha, busy hoon,
    baad mein call karna."""

    def _apply(self, slot: str, value: str):
        cm = ConversationManager(phone_number="local-demo")
        return cm.apply_tool_call("update_state", {"slot": slot, "value": value})

    def test_haan_normalizes_to_yes(self):
        result = self._apply("permission_granted", "Haan")
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], "yes")

    def test_nahi_normalizes_to_no(self):
        result = self._apply("interest_decision", "nahi")
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], "no")

    def test_achha_normalizes_to_yes(self):
        result = self._apply("permission_granted", "achha")
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], "yes")

    def test_busy_hoon_normalizes_to_maybe(self):
        result = self._apply("permission_granted", "busy hoon")
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], "maybe")

    def test_baad_mein_call_karna_normalizes_to_maybe(self):
        result = self._apply("callback_requested", "baad mein call karna")
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], "maybe")


if __name__ == "__main__":
    unittest.main()
