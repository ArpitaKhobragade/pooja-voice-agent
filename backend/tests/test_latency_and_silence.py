"""
Tests for the latency work (streamed TTS, disk cache, early first clause,
trailing-silence trim) and for how silence / unclear input is handled.
No network, no API keys, no audio hardware.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import audio_io  # noqa: E402
from local.reply_stream import stream_sentences  # noqa: E402
from local.tts_cartesia import CartesiaTTS, CartesiaUnavailable  # noqa: E402
from pooja_cli import MAX_NO_INPUT_IN_A_ROW, PoojaSession  # noqa: E402


class _StreamCtx:
    """Stands in for client.tts.with_streaming_response.generate(...)."""

    def __init__(self, payload: bytes, step: int = 3):
        self.payload = payload
        self.step = step

    def __enter__(self):
        resp = MagicMock()
        resp.iter_bytes.side_effect = lambda size: iter(
            [self.payload[i:i + self.step] for i in range(0, len(self.payload), self.step)]
        )
        return resp

    def __exit__(self, *a):
        return False


def _tts(cache_dir=None, payload=None):
    payload = payload if payload is not None else np.arange(1, 21, dtype=np.int16).tobytes()
    client = MagicMock()
    client.tts.with_streaming_response.generate.side_effect = lambda **kw: _StreamCtx(payload)
    with patch.dict(os.environ, {"CARTESIA_API_KEY": "k", "CARTESIA_VOICE_ID": "v"}):
        with patch("cartesia.Cartesia", return_value=client):
            tts = CartesiaTTS(cache_dir=cache_dir)
    return tts, client


class TestStreamedSynthesis(unittest.TestCase):
    def test_odd_sized_network_chunks_never_split_a_sample(self):
        tts, _ = _tts()
        chunks = list(tts.synthesize_stream("Hello Sir"))
        joined = np.concatenate(chunks)
        np.testing.assert_array_equal(joined, np.arange(1, 21, dtype=np.int16))

    def test_second_request_for_same_line_is_served_from_cache(self):
        tts, client = _tts()
        list(tts.synthesize_stream("Hello Sir"))
        list(tts.synthesize_stream("Hello Sir"))
        self.assertEqual(client.tts.with_streaming_response.generate.call_count, 1)

    def test_interrupted_stream_is_not_cached(self):
        tts, client = _tts()
        gen = tts.synthesize_stream("Hello Sir")
        next(gen)
        gen.close()  # owner interrupted: only a partial clip was received
        list(tts.synthesize_stream("Hello Sir"))
        self.assertEqual(client.tts.with_streaming_response.generate.call_count, 2)

    def test_disk_cache_survives_a_new_process(self):
        with tempfile.TemporaryDirectory() as d:
            tts1, _ = _tts(cache_dir=d)
            list(tts1.synthesize_stream("Hello Sir"))
            tts2, client2 = _tts(cache_dir=d)  # fresh object, same folder
            out = np.concatenate(list(tts2.synthesize_stream("Hello Sir")))
            self.assertEqual(client2.tts.with_streaming_response.generate.call_count, 0)
            self.assertEqual(len(out), 20)

    def test_prewarm_never_calls_the_network(self):
        with tempfile.TemporaryDirectory() as d:
            tts, client = _tts(cache_dir=d)
            tts.prewarm(["Hello Sir", "Dhanyavaad"])
            self.assertEqual(client.tts.with_streaming_response.generate.call_count, 0)

    def test_brand_name_is_spaced_for_speech_in_the_request(self):
        tts, client = _tts()
        list(tts.synthesize_stream("HostelPGDekho.com se"))
        kwargs = client.tts.with_streaming_response.generate.call_args.kwargs
        from local.speech_text import BRAND_SPOKEN, DOTCOM_SPOKEN

        self.assertEqual(kwargs["transcript"], f"{BRAND_SPOKEN} {DOTCOM_SPOKEN} se")


class TestQuotaMessage(unittest.TestCase):
    def test_402_gives_an_actionable_message(self):
        import cartesia

        err = cartesia.APIStatusError("quota", response=MagicMock(status_code=402, headers={}), body=None)
        err.status_code = 402
        wrapped = CartesiaTTS._wrap_error(err)
        self.assertIsInstance(wrapped, CartesiaUnavailable)
        self.assertIn("credits are used up", str(wrapped))


class TestEarlyFirstClause(unittest.TestCase):
    def test_first_segment_can_end_at_a_comma(self):
        out = list(stream_sentences(["Ji Sir, HostelPGDekho sirf listing app nahi hai, ", "smart search se inquiry milti hai."], []))
        self.assertEqual(out[0], "Ji Sir, HostelPGDekho sirf listing app nahi hai,")
        self.assertEqual(out[1], "smart search se inquiry milti hai.")

    def test_short_leading_clause_is_not_split_off(self):
        out = list(stream_sentences(["Ji Sir, bilkul sahi baat hai."], []))
        self.assertEqual(out, ["Ji Sir, bilkul sahi baat hai."])

    def test_only_the_first_segment_uses_clause_splitting(self):
        text = "Pehla vakya bahut lamba hai Sir, ismein comma hai. Doosra vakya bhi lamba hai Sir, aur comma bhi hai."
        out = list(stream_sentences([text], []))
        self.assertEqual(out[-1], "Doosra vakya bhi lamba hai Sir, aur comma bhi hai.")


class TestTrailingSilenceTrim(unittest.TestCase):
    def _clip(self, speech_s, silence_s):
        rng = np.random.default_rng(0)
        speech = (rng.standard_normal(int(16000 * speech_s)) * 3000).astype(np.int16)
        silence = (rng.standard_normal(int(16000 * silence_s)) * 10).astype(np.int16)
        return np.concatenate([speech, silence])

    def test_long_silent_tail_is_removed(self):
        clip = self._clip(1.0, 0.65)
        trimmed = audio_io.trim_trailing_silence(clip)
        self.assertLess(len(trimmed), len(clip))
        self.assertGreaterEqual(len(trimmed), int(16000 * 1.0))  # all speech kept
        self.assertLessEqual(len(trimmed) / 16000, 1.0 + 0.25 + 0.05)

    def test_all_silence_is_returned_unchanged(self):
        clip = self._clip(0.0, 1.0)
        self.assertEqual(len(audio_io.trim_trailing_silence(clip)), len(clip))

    def test_clip_without_a_silent_tail_is_unchanged(self):
        clip = self._clip(1.0, 0.0)
        self.assertEqual(len(audio_io.trim_trailing_silence(clip)), len(clip))


class _NoInput:
    """get_user_input stand-in that never produces speech."""

    def __init__(self, reason):
        self.none_reason = reason
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return None


class TestSilenceHandling(unittest.TestCase):
    def _run(self, reason):
        out = []
        getter = _NoInput(reason)
        session = PoojaSession(
            llm=MagicMock(), get_user_input=getter, deliver_reply=lambda t: out.append(t), conversational=True
        )
        cm = session.run()
        return cm, out, getter

    def test_silence_is_not_called_unclear_audio(self):
        _, out, _ = self._run("silence")
        asked = out[1]
        self.assertNotIn("clear nahi", asked)

    def test_unclear_audio_asks_to_repeat(self):
        _, out, _ = self._run("unclear")
        self.assertTrue(any("clear" in t or "bata dijiye" in t or "dobara" in t for t in out[1:]))

    def test_call_ends_politely_after_repeated_silence(self):
        cm, out, getter = self._run("silence")
        self.assertTrue(cm.get_state().call_ended)
        self.assertEqual(getter.calls, MAX_NO_INPUT_IN_A_ROW)
        self.assertIn("baad mein call karungi", out[-1])


if __name__ == "__main__":
    unittest.main()
