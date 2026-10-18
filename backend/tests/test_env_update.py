"""
Tests for local/env_update.py — the function that writes
CARTESIA_VOICE_ID into .env from the voice audition web UI. These are
the highest-stakes tests in this project: a bug here could corrupt a
real .env file full of real secrets.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.env_update import update_env_value  # noqa: E402


class TestEnvUpdate(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.env_path = Path(self._tmpdir.name) / ".env"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write(self, content: str) -> None:
        self.env_path.write_text(content, encoding="utf-8")

    def _read(self) -> str:
        return self.env_path.read_text(encoding="utf-8")

    def test_replaces_existing_key_preserving_every_other_line(self):
        self._write(
            "GROQ_API_KEY=sk-super-secret-groq-value\n"
            "SARVAM_API_KEY=sk-super-secret-sarvam-value\n"
            "CARTESIA_API_KEY=sk-super-secret-cartesia-value\n"
            "CARTESIA_VOICE_ID=old-voice-id\n"
            "CARTESIA_MODEL_ID=sonic-3.5\n"
            "# a comment line\n"
            "\n"
            "POOJA_MAX_TURNS=20\n"
        )
        update_env_value(self.env_path, "CARTESIA_VOICE_ID", "new-voice-id")
        result = self._read()

        self.assertIn("CARTESIA_VOICE_ID=new-voice-id\n", result)
        self.assertNotIn("old-voice-id", result)
        # Every other secret and line must be byte-for-byte untouched.
        self.assertIn("GROQ_API_KEY=sk-super-secret-groq-value\n", result)
        self.assertIn("SARVAM_API_KEY=sk-super-secret-sarvam-value\n", result)
        self.assertIn("CARTESIA_API_KEY=sk-super-secret-cartesia-value\n", result)
        self.assertIn("CARTESIA_MODEL_ID=sonic-3.5\n", result)
        self.assertIn("# a comment line\n", result)
        self.assertIn("POOJA_MAX_TURNS=20\n", result)
        # Only one line changed — same total line count.
        self.assertEqual(result.count("\n"), 8)

    def test_appends_key_when_missing(self):
        self._write("GROQ_API_KEY=abc\nSARVAM_API_KEY=def\n")
        update_env_value(self.env_path, "CARTESIA_VOICE_ID", "voice-xyz")
        result = self._read()

        self.assertIn("GROQ_API_KEY=abc\n", result)
        self.assertIn("SARVAM_API_KEY=def\n", result)
        self.assertTrue(result.rstrip("\n").endswith("CARTESIA_VOICE_ID=voice-xyz"))

    def test_creates_file_if_it_does_not_exist(self):
        update_env_value(self.env_path, "CARTESIA_VOICE_ID", "voice-xyz")
        self.assertEqual(self._read(), "CARTESIA_VOICE_ID=voice-xyz\n")

    def test_does_not_match_key_as_a_substring_of_another_key(self):
        # CARTESIA_VOICE_ID must not accidentally match
        # CARTESIA_VOICE_ID_BACKUP or similar near-miss keys.
        self._write("CARTESIA_VOICE_ID_BACKUP=keep-me\nOTHER=1\n")
        update_env_value(self.env_path, "CARTESIA_VOICE_ID", "new-voice")
        result = self._read()

        self.assertIn("CARTESIA_VOICE_ID_BACKUP=keep-me\n", result)
        self.assertIn("CARTESIA_VOICE_ID=new-voice\n", result)
        self.assertIn("OTHER=1\n", result)

    def test_blank_value_is_still_written_explicitly(self):
        self._write("CARTESIA_VOICE_ID=\nOTHER=1\n")
        update_env_value(self.env_path, "CARTESIA_VOICE_ID", "picked-id")
        result = self._read()
        self.assertIn("CARTESIA_VOICE_ID=picked-id\n", result)
        self.assertIn("OTHER=1\n", result)

    def test_rejects_unsafe_key_names(self):
        self._write("A=1\n")
        with self.assertRaises(ValueError):
            update_env_value(self.env_path, "NOT VALID; rm -rf", "x")
        # File must be untouched after a rejected call.
        self.assertEqual(self._read(), "A=1\n")

    def test_no_leftover_tmp_file_after_write(self):
        self._write("A=1\n")
        update_env_value(self.env_path, "CARTESIA_VOICE_ID", "v1")
        tmp_path = self.env_path.with_name(self.env_path.name + ".tmp")
        self.assertFalse(tmp_path.exists())


if __name__ == "__main__":
    unittest.main()
