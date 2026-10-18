"""
Tests for the interruption (barge-in) decision logic. Pure logic: feeds the
detector the per-frame numbers a real call would produce.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.duplex import BargeInDetector  # noqa: E402


def run(detector, frames):
    """frames: list of (mic_rms, is_speech, output_rms). Returns the index
    of the frame at which an interruption fired, or None."""
    detector.reset_turn()
    for i, (mic, speech, out) in enumerate(frames):
        if detector.update(mic, speech, out):
            return i
    return None


class TestBargeInDetector(unittest.TestCase):
    def test_headphones_owner_interrupts_quickly(self):
        # Output audible, mic hears ~nothing of it (headphones), then the owner speaks.
        d = BargeInDetector()
        frames = [(20, False, 3000)] * 20 + [(2500, True, 3000)] * 20
        fired = run(d, frames)
        self.assertIsNotNone(fired)
        self.assertLess(fired, 20 + 12)  # within ~12 frames (~0.4s) of speaking
        self.assertLess(d.coupling, 0.05)

    def test_speaker_echo_alone_never_triggers(self):
        # Laptop speakers: the mic hears ~25% of the output, and VAD calls the echo "speech".
        d = BargeInDetector()
        frames = [(750, True, 3000)] * 120
        self.assertIsNone(run(d, frames))
        self.assertAlmostEqual(d.coupling, 0.25, places=2)

    def test_owner_louder_than_echo_still_interrupts(self):
        d = BargeInDetector()
        echo = [(750, True, 3000)] * 20
        owner = [(6500, True, 3000)] * 20
        fired = run(d, echo + owner)
        self.assertIsNotNone(fired)
        self.assertGreaterEqual(fired, 20)

    def test_brief_noise_does_not_interrupt(self):
        d = BargeInDetector()
        frames = [(20, False, 3000)] * 20 + [(3000, True, 3000)] * 3 + [(20, False, 3000)] * 20
        self.assertIsNone(run(d, frames))

    def test_quiet_mutter_below_the_floor_is_ignored(self):
        d = BargeInDetector()
        frames = [(20, False, 3000)] * 20 + [(250, True, 3000)] * 30
        self.assertIsNone(run(d, frames))

    def test_no_false_trigger_while_echo_level_is_still_unknown(self):
        # First playing frames, before calibration finishes, must never fire.
        d = BargeInDetector()
        frames = [(900, True, 3000)] * 10
        self.assertIsNone(run(d, frames))

    def test_known_coupling_allows_interrupting_during_the_next_turns_start(self):
        d = BargeInDetector()
        run(d, [(750, True, 3000)] * 20)  # turn 1 learns coupling ~0.25
        fired = run(d, [(6500, True, 3000)] * 15)  # turn 2: owner talks from the very start
        self.assertIsNotNone(fired)

    def test_interrupting_while_nothing_is_playing(self):
        # Owner talks while Pooja is still "thinking" (no output yet).
        d = BargeInDetector()
        fired = run(d, [(3000, True, 0)] * 15)
        self.assertIsNotNone(fired)


if __name__ == "__main__":
    unittest.main()
