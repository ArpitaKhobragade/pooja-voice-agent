"""
MOCKED tests for the --audition-voices tool (local/voice_audition.py).
No real Cartesia calls, no real audio playback, no .env writes — every
side effect is injected and asserted on directly.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.tts_cartesia import CartesiaUnavailable  # noqa: E402
from local.voice_audition import VOICE_CANDIDATES, audition_voices, run_audition_cli  # noqa: E402

FAKE_VOICES = [
    {"name": "Siya", "id": "voice-1"},
    {"name": "Sana", "id": "voice-2"},
    {"name": "Anika", "id": "voice-3"},
]


def _fake_synthesize_ok(voice_id, text):
    return (f"pcm-for-{voice_id}", 16000)


class TestVoiceAudition(unittest.TestCase):
    def test_ten_real_candidates_are_present_with_names_and_ids(self):
        self.assertEqual(len(VOICE_CANDIDATES), 10)
        for v in VOICE_CANDIDATES:
            self.assertTrue(v["name"])
            self.assertRegex(v["id"], r"^[0-9a-f-]{36}$")

    def test_default_flow_visits_every_voice_in_order(self):
        announced = []
        played = []
        inputs = iter(["n", "n", "n"])  # advance through all 3

        failed = audition_voices(
            sample_text="test line",
            voices=FAKE_VOICES,
            synthesize_fn=_fake_synthesize_ok,
            play_fn=lambda pcm, sr: played.append(pcm),
            ask=lambda prompt: next(inputs),
            announce=announced.append,
        )

        self.assertEqual(failed, [])
        self.assertEqual(played, ["pcm-for-voice-1", "pcm-for-voice-2", "pcm-for-voice-3"])
        self.assertTrue(any("Siya" in a for a in announced))
        self.assertTrue(any("Anika" in a for a in announced))

    def test_replay_plays_the_same_voice_again_before_advancing(self):
        played = []
        # replay once on the first voice, then move on; second voice: next immediately
        inputs = iter(["r", "n", "n"])

        audition_voices(
            sample_text="test line",
            voices=FAKE_VOICES[:2],
            synthesize_fn=_fake_synthesize_ok,
            play_fn=lambda pcm, sr: played.append(pcm),
            ask=lambda prompt: next(inputs),
            announce=lambda *_: None,
        )

        # voice-1 played twice (initial + replay), voice-2 played once
        self.assertEqual(played, ["pcm-for-voice-1", "pcm-for-voice-1", "pcm-for-voice-2"])

    def test_quit_stops_immediately_without_finishing_the_list(self):
        played = []
        inputs = iter(["q"])  # quit right after the first sample

        failed = audition_voices(
            sample_text="test line",
            voices=FAKE_VOICES,
            synthesize_fn=_fake_synthesize_ok,
            play_fn=lambda pcm, sr: played.append(pcm),
            ask=lambda prompt: next(inputs),
            announce=lambda *_: None,
        )

        self.assertEqual(played, ["pcm-for-voice-1"])  # never reached voice-2 or voice-3
        self.assertEqual(failed, [])

    def test_blank_input_advances_to_next_voice(self):
        played = []
        inputs = iter(["", ""])

        audition_voices(
            sample_text="test line",
            voices=FAKE_VOICES[:2],
            synthesize_fn=_fake_synthesize_ok,
            play_fn=lambda pcm, sr: played.append(pcm),
            ask=lambda prompt: next(inputs),
            announce=lambda *_: None,
        )
        self.assertEqual(played, ["pcm-for-voice-1", "pcm-for-voice-2"])

    def test_synthesis_failure_is_reported_and_skips_to_next_voice(self):
        announced = []
        played = []
        inputs = iter(["n"])  # only needed for the voice that succeeds

        def flaky_synthesize(voice_id, text):
            if voice_id == "voice-1":
                raise CartesiaUnavailable("Cartesia rate limit hit: too many requests")
            return (f"pcm-for-{voice_id}", 16000)

        failed = audition_voices(
            sample_text="test line",
            voices=FAKE_VOICES[:2],
            synthesize_fn=flaky_synthesize,
            play_fn=lambda pcm, sr: played.append(pcm),
            ask=lambda prompt: next(inputs),
            announce=announced.append,
        )

        self.assertEqual(failed, ["voice-1"])
        self.assertEqual(played, ["pcm-for-voice-2"])  # continued past the failure
        self.assertTrue(any("Error" in a and "Siya" in a for a in announced))

    def test_no_voice_is_ever_written_to_environment(self):
        inputs = iter(["n", "n", "n"])
        before = dict(os.environ)

        audition_voices(
            sample_text="test line",
            voices=FAKE_VOICES,
            synthesize_fn=_fake_synthesize_ok,
            play_fn=lambda pcm, sr: None,
            ask=lambda prompt: next(inputs),
            announce=lambda *_: None,
        )

        self.assertEqual(dict(os.environ), before)  # CARTESIA_VOICE_ID (or anything else) untouched

    def test_run_audition_cli_requires_api_key(self):
        with patch.dict(os.environ, {"CARTESIA_API_KEY": ""}):
            with self.assertRaises(CartesiaUnavailable):
                run_audition_cli()

    def test_run_audition_cli_never_logs_the_api_key(self):
        captured = []
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-super-secret-value"}), \
             patch("local.voice_audition.audition_voices", return_value=[]), \
             patch("builtins.print", side_effect=lambda *a, **k: captured.append(" ".join(str(x) for x in a))):
            run_audition_cli(sample_text="hi")

        joined = "\n".join(captured)
        self.assertNotIn("sk-super-secret-value", joined)


if __name__ == "__main__":
    unittest.main()
