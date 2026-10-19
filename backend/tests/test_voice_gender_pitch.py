"""Pitch-based Sir / Ma'am guess: a woman's pitch must not be halved into a man's."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.voice_gender import GenderTracker, voiced_f0s  # noqa: E402


def voice_like(f0: float, seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    """A buzzy, vowel-like tone: a fundamental with strong harmonics (and a weaker first one,
    which is what makes autocorrelation pick the doubled period)."""
    t = np.arange(int(sr * seconds)) / sr
    sig = 0.5 * np.sin(2 * np.pi * f0 * t)
    for k, amp in ((2, 1.0), (3, 0.8), (4, 0.6), (5, 0.4)):
        sig += amp * 0.5 * np.sin(2 * np.pi * f0 * k * t)
    return (sig / np.abs(sig).max() * 9000).astype(np.int16)


class TestPitch(unittest.TestCase):
    def test_a_female_pitch_is_not_halved(self):
        f0s = voiced_f0s(voice_like(220))
        self.assertGreater(len(f0s), 20)
        self.assertAlmostEqual(float(np.median(f0s)), 220, delta=12)

    def test_a_male_pitch_stays_low(self):
        f0s = voiced_f0s(voice_like(120))
        self.assertAlmostEqual(float(np.median(f0s)), 120, delta=8)

    def test_tracker_guesses_from_pitch(self):
        woman, man = GenderTracker(), GenderTracker()
        woman.add_audio(voice_like(215))
        man.add_audio(voice_like(115))
        self.assertEqual(woman.guess, "female")
        self.assertEqual(man.guess, "male")


if __name__ == "__main__":
    unittest.main()


class TestWebCallEnding(unittest.TestCase):
    def test_pooja_ending_the_call_waits_for_her_last_line_to_play(self):
        import time

        from local.web_call import WebCall

        class Duplex:
            closed = False

            def __init__(self):
                self.end = time.time() + 0.6

            def buffered_seconds(self):
                return max(0.0, self.end - time.time())

        call = WebCall(lambda m: None, lambda b: None)
        call.TAIL_S = 0.0
        call.duplex = Duplex()
        t0 = time.time()
        call._let_speech_finish()
        self.assertGreaterEqual(time.time() - t0, 0.5)

    def test_a_user_stop_does_not_wait(self):
        import time

        from local.web_call import WebCall

        class Duplex:
            closed = False

            def buffered_seconds(self):
                return 5.0

        call = WebCall(lambda m: None, lambda b: None)
        call.duplex = Duplex()
        call._stop_reason = "stopped"
        t0 = time.time()
        call._let_speech_finish()
        self.assertLess(time.time() - t0, 0.2)

    def test_greeting_and_wish_follow_the_callers_timezone(self):
        import agent_config

        agent_config.set_call_timezone("Europe/London")
        self.assertIn(agent_config.greeting(), ("good morning", "good afternoon", "good evening", "namaste"))
        agent_config.set_call_timezone("not/a-zone")  # ignored, never raises
