"""
MOCKED tests for the local FastAPI voice-audition server
(local/audition_server.py). No real network calls, no real audio
playback, no real .env writes — every external effect is patched.
Uses FastAPI's TestClient (does not bind a real port).
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CARTESIA_API_KEY", "")  # avoid startup tagline fetch hitting real network

from fastapi.testclient import TestClient  # noqa: E402

import local.audition_server as audition_server  # noqa: E402
from local.tts_cartesia import CartesiaUnavailable  # noqa: E402

FAKE_VOICE_ID = "4459a9a5-69d6-4680-b970-e13dc51845b6"  # Siya, from VOICE_CANDIDATES


class TestAuditionServer(unittest.TestCase):
    def setUp(self):
        # Fresh cache per test so one test's synthesize count doesn't
        # leak into another's assertions.
        audition_server._audio_cache.clear()
        self.client = TestClient(audition_server.app)

    def test_get_voices_returns_all_ten_without_api_key(self):
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-super-secret"}):
            resp = self.client.get("/api/voices")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["voices"]), 10)
        self.assertIn("sample_text", data)
        self.assertTrue(data["cartesia_configured"])
        body_text = resp.text
        self.assertNotIn("sk-super-secret", body_text)
        for v in data["voices"]:
            self.assertIn("id", v)
            self.assertIn("name", v)
            self.assertNotIn("api_key", v)

    def test_get_audio_unknown_voice_is_404(self):
        resp = self.client.get("/api/audio/not-a-real-voice-id")
        self.assertEqual(resp.status_code, 404)

    def test_get_audio_success_returns_wav_and_caches(self):
        fake_pcm = np.array([1, 2, 3, 4], dtype=np.int16)

        with patch("local.audition_server.CartesiaTTS") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.synthesize.return_value = (fake_pcm, 16000)

            resp1 = self.client.get(f"/api/audio/{FAKE_VOICE_ID}")
            resp2 = self.client.get(f"/api/audio/{FAKE_VOICE_ID}")  # should hit cache

        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp1.headers["content-type"], "audio/wav")
        self.assertEqual(resp1.content, resp2.content)
        mock_instance.synthesize.assert_called_once()  # NOT called twice — cache worked

    def test_get_audio_failure_returns_502_without_leaking_key(self):
        with patch("local.audition_server.CartesiaTTS") as mock_cls, \
             patch.dict(os.environ, {"CARTESIA_API_KEY": "sk-super-secret-value"}):
            mock_cls.side_effect = CartesiaUnavailable("Cartesia rejected CARTESIA_API_KEY (authentication failed)")
            resp = self.client.get(f"/api/audio/{FAKE_VOICE_ID}")

        self.assertEqual(resp.status_code, 502)
        self.assertNotIn("sk-super-secret-value", resp.text)

    def test_save_voice_writes_only_that_key(self):
        with patch("local.audition_server.update_env_value") as mock_update:
            resp = self.client.post("/api/save", json={"voice_id": FAKE_VOICE_ID})

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertIn("Siya", data["message"])
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        self.assertEqual(args[1], "CARTESIA_VOICE_ID")
        self.assertEqual(args[2], FAKE_VOICE_ID)

    def test_save_voice_rejects_unknown_id(self):
        with patch("local.audition_server.update_env_value") as mock_update:
            resp = self.client.post("/api/save", json={"voice_id": "attacker-supplied-value"})

        self.assertEqual(resp.status_code, 400)
        mock_update.assert_not_called()

    def test_save_voice_failure_is_reported_not_500_hidden(self):
        with patch("local.audition_server.update_env_value", side_effect=OSError("disk full")):
            resp = self.client.post("/api/save", json={"voice_id": FAKE_VOICE_ID})

        self.assertEqual(resp.status_code, 500)
        self.assertIn("disk full", resp.text)

    def test_index_page_is_served(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Choose Pooja's Voice", resp.text)


if __name__ == "__main__":
    unittest.main()
