"""
Speech-to-text via Sarvam AI's REST API.

Endpoint: POST https://api.sarvam.ai/speech-to-text
Auth: api-subscription-key header
Body: multipart/form-data — model, mode (saaras:v3 only), file (wav)
Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe

The synchronous REST endpoint handles clips up to ~30s, which fits a
single turn-based utterance here. saaras:v3 auto-detects and transcribes
Hindi/English/Hinglish (code-mixed) speech, which is why it's the default
over the older saarika models.

This module has no pipecat dependency — it's a plain REST client used by
the local CLI only.
"""

from __future__ import annotations

import os

import numpy as np
import requests

from local.audio_io import pcm_int16_to_wav_bytes

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saaras:v3")
REQUEST_TIMEOUT_S = 30
# "translit" returns Roman script ("Haan ji bataiye"), which the rest of the
# pipeline (keyword matching, the Roman-script LLM prompt) expects. The
# default "transcribe" mode returns Devanagari for Hindi speech.
STT_MODE = os.environ.get("SARVAM_STT_MODE", "translit")


class SarvamSTTError(RuntimeError):
    pass


def transcribe(pcm: np.ndarray, sample_rate: int, api_key: str | None = None) -> str:
    """Sends one recorded utterance to Sarvam STT and returns the transcript
    text (may be empty string if nothing intelligible was heard)."""
    key = api_key or os.environ.get("SARVAM_API_KEY")
    if not key:
        raise SarvamSTTError("SARVAM_API_KEY is not set.")

    wav_bytes = pcm_int16_to_wav_bytes(pcm, sample_rate)

    try:
        response = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": key},
            data={"model": DEFAULT_MODEL, "mode": STT_MODE},
            files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise SarvamSTTError(f"Network error calling Sarvam STT: {e}") from e

    if response.status_code != 200:
        raise SarvamSTTError(
            f"Sarvam STT returned HTTP {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
    except ValueError as e:
        raise SarvamSTTError(f"Sarvam STT returned non-JSON response: {response.text[:500]}") from e

    transcript = data.get("transcript", "")
    return transcript.strip()
