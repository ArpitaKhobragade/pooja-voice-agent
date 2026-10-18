"""
MOCKED unit tests for the Sarvam STT and Cartesia TTS provider wrappers.
No network calls, no API keys required — every external call is patched.
For tests that hit the REAL APIs, see test_live_apis.py (skipped
automatically unless real credentials are present in the environment).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local import stt_sarvam  # noqa: E402
from local.tts_cartesia import (  # noqa: E402
    CartesiaTTS,
    CartesiaUnavailable,
    TTSEngine,
    list_hindi_female_voices,
)


class TestSarvamSTTMocked(unittest.TestCase):
    def test_transcribe_success(self):
        fake_response = MagicMock(status_code=200)
        fake_response.json.return_value = {"transcript": "Haan ji bataiye"}
        pcm = np.zeros(1600, dtype=np.int16)

        with patch("local.stt_sarvam.requests.post", return_value=fake_response) as mock_post:
            text = stt_sarvam.transcribe(pcm, 16000, api_key="fake-key")

        self.assertEqual(text, "Haan ji bataiye")
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["api-subscription-key"], "fake-key")
        self.assertIn("file", kwargs["files"])

    def test_transcribe_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SARVAM_API_KEY", None)
            with self.assertRaises(stt_sarvam.SarvamSTTError):
                stt_sarvam.transcribe(np.zeros(100, dtype=np.int16), 16000, api_key=None)

    def test_transcribe_http_error_raises_clear_message(self):
        fake_response = MagicMock(status_code=401, text="Invalid API key")
        with patch("local.stt_sarvam.requests.post", return_value=fake_response):
            with self.assertRaises(stt_sarvam.SarvamSTTError) as ctx:
                stt_sarvam.transcribe(np.zeros(100, dtype=np.int16), 16000, api_key="bad-key")
        self.assertIn("401", str(ctx.exception))

    def test_transcribe_network_error_raises_clear_message(self):
        import requests

        with patch("local.stt_sarvam.requests.post", side_effect=requests.ConnectionError("no route")):
            with self.assertRaises(stt_sarvam.SarvamSTTError):
                stt_sarvam.transcribe(np.zeros(100, dtype=np.int16), 16000, api_key="fake-key")


class TestCartesiaTTSMocked(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(
            os.environ,
            {"CARTESIA_API_KEY": "fake-key", "CARTESIA_VOICE_ID": "fake-voice-id"},
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_missing_api_key_raises_cartesia_unavailable(self):
        with patch.dict(os.environ, {"CARTESIA_API_KEY": ""}):
            with self.assertRaises(CartesiaUnavailable):
                CartesiaTTS()

    def test_missing_voice_id_raises_cartesia_unavailable(self):
        with patch.dict(os.environ, {"CARTESIA_VOICE_ID": ""}):
            with self.assertRaises(CartesiaUnavailable):
                CartesiaTTS()

    def test_synthesize_success_returns_pcm(self):
        fake_bytes = np.array([100, -100, 200, -200], dtype=np.int16).tobytes()
        fake_response = MagicMock()
        fake_response.read.return_value = fake_bytes
        fake_client = MagicMock()
        fake_client.tts.generate.return_value = fake_response

        with patch("cartesia.Cartesia", return_value=fake_client):
            tts = CartesiaTTS()
            pcm, sr = tts.synthesize("Hello Sir")

        self.assertEqual(sr, 16000)
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(len(pcm), 4)
        # Confirm the call shape matches the installed SDK's real signature.
        _, kwargs = fake_client.tts.generate.call_args
        self.assertEqual(kwargs["transcript"], "Hello Sir")
        self.assertEqual(kwargs["voice"], "fake-voice-id")
        self.assertEqual(kwargs["output_format"]["encoding"], "pcm_s16le")

    def test_speed_out_of_range_rejected_without_calling_api(self):
        with patch("cartesia.Cartesia") as mock_cartesia_cls:
            with self.assertRaises(CartesiaUnavailable):
                CartesiaTTS(speed=2.0)  # Cartesia's documented range is [0.6, 1.5]
            with self.assertRaises(CartesiaUnavailable):
                CartesiaTTS(speed=0.1)
        mock_cartesia_cls.assert_not_called()

    def test_default_speed_omits_generation_config(self):
        fake_bytes = np.array([1, 2], dtype=np.int16).tobytes()
        fake_response = MagicMock()
        fake_response.read.return_value = fake_bytes
        fake_client = MagicMock()
        fake_client.tts.generate.return_value = fake_response

        with patch("cartesia.Cartesia", return_value=fake_client):
            tts = CartesiaTTS()  # speed defaults to 1.0
            tts.synthesize("hello")

        _, kwargs = fake_client.tts.generate.call_args
        self.assertNotIn("generation_config", kwargs)

    def test_tuned_speed_sends_generation_config_and_keeps_voice_unchanged(self):
        fake_bytes = np.array([1, 2], dtype=np.int16).tobytes()
        fake_response = MagicMock()
        fake_response.read.return_value = fake_bytes
        fake_client = MagicMock()
        fake_client.tts.generate.return_value = fake_response

        with patch("cartesia.Cartesia", return_value=fake_client):
            tts = CartesiaTTS(speed=1.2)
            tts.synthesize("hello")

        _, kwargs = fake_client.tts.generate.call_args
        self.assertEqual(kwargs["generation_config"], {"speed": 1.2})
        self.assertEqual(kwargs["voice"], "fake-voice-id")  # unchanged by speed tuning

    def test_synthesize_failure_raises_cartesia_unavailable(self):
        fake_client = MagicMock()
        fake_client.tts.generate.side_effect = RuntimeError("boom")

        with patch("cartesia.Cartesia", return_value=fake_client):
            tts = CartesiaTTS()
            with self.assertRaises(CartesiaUnavailable):
                tts.synthesize("Hello Sir")

    def test_engine_mandatory_mode_propagates_failure(self):
        fake_client = MagicMock()
        fake_client.tts.generate.side_effect = RuntimeError("boom")

        with patch("cartesia.Cartesia", return_value=fake_client):
            engine = TTSEngine(mode="cartesia", allow_fallback=False)
            with self.assertRaises(CartesiaUnavailable):
                engine.speak("Hello Sir", play_fn=lambda pcm, sr: False)

    def test_engine_fallback_mode_uses_pyttsx3_and_reports_it(self):
        fake_client = MagicMock()
        fake_client.tts.generate.side_effect = RuntimeError("boom")

        with patch("cartesia.Cartesia", return_value=fake_client), \
             patch("local.tts_cartesia.speak_pyttsx3") as mock_pyttsx3:
            engine = TTSEngine(mode="cartesia", allow_fallback=True)
            result = engine.speak("Hello Sir", play_fn=lambda pcm, sr: False)

        mock_pyttsx3.assert_called_once_with("Hello Sir")
        self.assertEqual(result.engine, "pyttsx3")
        self.assertIsNotNone(result.warning)

    def test_engine_explicit_pyttsx3_mode_never_touches_cartesia(self):
        with patch("cartesia.Cartesia") as mock_cartesia_cls, \
             patch("local.tts_cartesia.speak_pyttsx3") as mock_pyttsx3:
            engine = TTSEngine(mode="pyttsx3")
            result = engine.speak("Hello Sir")

        mock_cartesia_cls.assert_not_called()
        mock_pyttsx3.assert_called_once_with("Hello Sir")
        self.assertEqual(result.engine, "pyttsx3")

    def test_list_hindi_female_voices_filters_client_side_and_never_passes_language_kwarg(self):
        # Regression test: client.voices.list() has no `language` kwarg in
        # the installed SDK (verified against cartesia/resources/voices.py)
        # — passing one would raise TypeError. This also checks the
        # client-side language filter actually drops non-Hindi voices.
        hi_voice = MagicMock(id="v1", name="Priya", tagline="warm", language="hi")
        en_voice = MagicMock(id="v2", name="Jane", tagline="crisp", language="en")
        fake_client = MagicMock()
        fake_client.voices.list.return_value = [hi_voice, en_voice]

        with patch("cartesia.Cartesia", return_value=fake_client):
            results = list_hindi_female_voices(api_key="fake-key", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "v1")
        _, kwargs = fake_client.voices.list.call_args
        self.assertNotIn("language", kwargs)
        self.assertEqual(kwargs["gender"], "feminine")

    def test_barge_in_flag_propagated_from_play_fn(self):
        fake_bytes = np.array([1, 2], dtype=np.int16).tobytes()
        fake_response = MagicMock()
        fake_response.read.return_value = fake_bytes
        fake_client = MagicMock()
        fake_client.tts.generate.return_value = fake_response

        with patch("cartesia.Cartesia", return_value=fake_client):
            engine = TTSEngine(mode="cartesia")
            result = engine.speak("Hello Sir", play_fn=lambda pcm, sr: True)

        self.assertTrue(result.interrupted)


if __name__ == "__main__":
    unittest.main()
