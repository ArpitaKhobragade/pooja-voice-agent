"""
LIVE API integration tests — these make REAL network calls to Groq,
Sarvam, and Cartesia using whatever credentials are in the environment
(.env). They are NOT mocked and NOT run by default in an environment
with no credentials: each test is skipped automatically
(pytest.mark.skipif) if its required API key is missing, so `pytest` is
always safe to run without keys and these simply won't execute.

To actually run them: fill in .env with real keys, then:
    python -m pytest tests/test_live_apis.py -v -s

This project was built and tested in an environment with no API keys
available, so these tests have NOT been executed against the real APIs
as part of this work — only written and reasoned through against each
SDK's real, installed interface (see local/tts_cartesia.py and
local/stt_sarvam.py's docstrings for what was verified and how). Run
them yourself once you add your keys, and report back if any of them
fail — that's real signal about the request shape needing adjustment,
not something to paper over.
"""

import os
import sys
import unittest

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

HAS_GROQ = bool(os.environ.get("GROQ_API_KEY"))
HAS_SARVAM = bool(os.environ.get("SARVAM_API_KEY"))
HAS_CARTESIA = bool(os.environ.get("CARTESIA_API_KEY") and os.environ.get("CARTESIA_VOICE_ID"))


@pytest.mark.skipif(not HAS_GROQ, reason="GROQ_API_KEY not set")
class TestLiveGroq(unittest.TestCase):
    def test_simple_chat_completion(self):
        from local.llm_groq import GroqLLM

        llm = GroqLLM()
        message = llm.chat(
            [{"role": "user", "content": "Reply with exactly the word: PONG"}]
        )
        self.assertIsNotNone(message.content)
        print(f"\n[LIVE Groq] model={llm.model} response={message.content!r}")

    def test_tool_calling_with_update_state_schema(self):
        from local.llm_groq import GroqLLM
        from local_tools import LOCAL_TOOLS

        llm = GroqLLM()
        message = llm.chat(
            [
                {
                    "role": "system",
                    "content": "You are a test agent. The user will confirm permission to "
                    "talk. Call update_state with slot='permission_granted' and "
                    "value='yes' when they do.",
                },
                {"role": "user", "content": "Haan ji, bataiye."},
            ],
            tools=LOCAL_TOOLS,
        )
        print(f"\n[LIVE Groq tool-calling] tool_calls={message.tool_calls}")
        # Not asserting a tool call necessarily fires (model behavior can
        # vary) — this test's purpose is to confirm the call itself
        # succeeds against the real API with our schema, not to pin exact
        # model behavior.


@pytest.mark.skipif(not HAS_SARVAM, reason="SARVAM_API_KEY not set")
class TestLiveSarvam(unittest.TestCase):
    def test_transcribe_silence_does_not_error(self):
        """Sends 1s of silence — we're checking the API call itself
        succeeds (auth + request shape), not transcription accuracy,
        since we have no real recorded speech sample in this environment."""
        from local import stt_sarvam

        silence = np.zeros(16000, dtype=np.int16)  # 1s @16kHz
        text = stt_sarvam.transcribe(silence, 16000)
        print(f"\n[LIVE Sarvam] transcript of silence: {text!r}")


@pytest.mark.skipif(not HAS_CARTESIA, reason="CARTESIA_API_KEY/CARTESIA_VOICE_ID not set")
class TestLiveCartesia(unittest.TestCase):
    def test_synthesize_short_phrase(self):
        from local.tts_cartesia import CartesiaTTS

        tts = CartesiaTTS()
        pcm, sr = tts.synthesize("Hello Sir, this is a live test.")
        self.assertGreater(len(pcm), 0)
        self.assertEqual(sr, 16000)
        print(f"\n[LIVE Cartesia] model={tts.model_id} voice={tts.voice_id} "
              f"got {len(pcm)} samples ({len(pcm) / sr:.2f}s)")


if __name__ == "__main__":
    unittest.main()
