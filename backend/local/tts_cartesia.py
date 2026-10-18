"""
Text-to-speech for Pooja, using the OFFICIAL Cartesia Python SDK
(`pip install cartesia`) as the primary, intended engine.

Verified against the installed `cartesia` SDK's actual generated types
(not guessed): `client.tts.generate(model_id=, output_format=, transcript=,
voice=, language=)` returns a `BinaryAPIResponse` (`.read()` gives raw
bytes). Confirmed by inspecting:
  cartesia/types/tts_model.py        -> valid model_id literals
  cartesia/types/supported_language.py -> valid language codes (includes "hi")
  cartesia/types/raw_output_format_param.py / wav_output_format_param.py
  cartesia/types/voice_specifier_param.py -> voice can be a plain string id
  cartesia/_exceptions.py            -> AuthenticationError, RateLimitError, etc.

On "Cartesia Sonic 3.6": the installed SDK's `TTSModel` literal is
`"sonic-3.5" | "sonic-3" | "sonic-3.5-2026-05-04" | "sonic-3-2026-01-12" |
"sonic-3-2025-10-27" | "sonic-latest"` — there is no "sonic-3.6". This is
checked against the SDK actually installed in this project, not just
documentation, so it's a hard fact for this version, not a guess.
CARTESIA_MODEL_ID therefore defaults to "sonic-3.5" (the latest
numbered/stable id in that list). If Cartesia later ships a newer model
id, update CARTESIA_MODEL_ID in .env — no code change needed.

By default, Cartesia is MANDATORY for --mode voice: if it's not
configured or a call fails, `CartesiaTTS.synthesize()` raises
CartesiaUnavailable instead of silently degrading, so a broken
integration can never be mistaken for a working one. pyttsx3 is only
used when explicitly selected (pooja_cli.py's `--tts-engine pyttsx3` or
`--allow-tts-fallback`), never as a silent default.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from local import audio_io
from local.speech_text import speech_text

DEFAULT_MODEL_ID = os.environ.get("CARTESIA_MODEL_ID", "sonic-3.5")
DEFAULT_LANGUAGE = os.environ.get("CARTESIA_LANGUAGE", "hi")
REQUEST_TIMEOUT_S = 30

# Speed control: verified against the installed SDK's real type
# (cartesia/types/generation_config_param.py) — `generation_config.speed`
# is a float in [0.6, 1.5] (1.0 = default pace), documented as applying to
# sonic-3 and newer models (which includes our sonic-3.5 default). This is
# NOT the same as guessing a `speed=1.2` kwarg on generate() itself; it's
# the actual documented field, nested under generation_config.
SPEED_MIN, SPEED_MAX = 0.6, 1.5
DEFAULT_SPEED = float(os.environ.get("CARTESIA_SPEED", "1.0"))

VALID_MODEL_IDS = {
    "sonic-3.5",
    "sonic-3",
    "sonic-3.5-2026-05-04",
    "sonic-3-2026-01-12",
    "sonic-3-2025-10-27",
    "sonic-latest",
}


class CartesiaUnavailable(RuntimeError):
    """Raised when Cartesia TTS cannot be used — missing config, auth
    failure, rate limit, network error, or any other API failure. Callers
    must not silently swallow this into a fallback unless the fallback
    was explicitly requested by the user."""


@dataclass
class SpeakResult:
    engine: str  # "cartesia" | "pyttsx3"
    model_id: str | None = None
    voice_id: str | None = None
    warning: str | None = None  # e.g. unverified model id, or a fallback explanation
    interrupted: bool = False  # True if barge-in cut this turn short
    synthesis_s: float | None = None  # Cartesia API call latency, separate from playback
    playback_s: float | None = None  # wall-clock time spent in play_fn (speaking + any blocking wait)


class CartesiaTTS:
    """Thin wrapper around the official Cartesia SDK for one-shot
    turn-based synthesis + playback (not streaming — see README for why:
    this project is turn-based, not full-duplex)."""

    def __init__(self, api_key: str | None = None, voice_id: str | None = None,
                 model_id: str | None = None, language: str | None = None,
                 speed: float | None = None, cache_dir: str | None = None):
        key = api_key or os.environ.get("CARTESIA_API_KEY")
        vid = voice_id or os.environ.get("CARTESIA_VOICE_ID")
        if not key:
            raise CartesiaUnavailable("CARTESIA_API_KEY is not set.")
        if not vid:
            raise CartesiaUnavailable(
                "CARTESIA_VOICE_ID is not set. Run "
                "'python pooja_cli.py --find-voice' to list Indian female "
                "voice ids from your Cartesia account and pick one."
            )

        self.speed = DEFAULT_SPEED if speed is None else speed
        if not (SPEED_MIN <= self.speed <= SPEED_MAX):
            raise CartesiaUnavailable(
                f"CARTESIA_SPEED={self.speed} is out of range — Cartesia's "
                f"documented generation_config.speed only accepts "
                f"[{SPEED_MIN}, {SPEED_MAX}]."
            )

        from cartesia import Cartesia  # imported lazily: not needed for --mode offline

        self._client = Cartesia(api_key=key)
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.voice_id = vid  # never changed by speed tuning — voice stays whatever was configured/passed
        self.language = language or DEFAULT_LANGUAGE
        self.model_id_verified = self.model_id in VALID_MODEL_IDS
        self.last_synthesis_s: float | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        # Optional on-disk cache (voice mode turns it on): a line that has
        # been synthesized once is never paid for again, across runs.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- cache ---------------------------------------------------------------

    def _cache_key(self, spoken: str) -> str:
        raw = f"{self.model_id}|{self.voice_id}|{self.language}|{self.speed}|{spoken}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, spoken: str) -> np.ndarray | None:
        with self._cache_lock:
            cached = self._cache.get(spoken)
        if cached is not None:
            return cached
        if self.cache_dir:
            path = self.cache_dir / f"{self._cache_key(spoken)}.pcm"
            try:
                if path.exists():
                    pcm = np.frombuffer(path.read_bytes(), dtype=np.int16)
                    with self._cache_lock:
                        self._cache[spoken] = pcm
                    return pcm
            except OSError:
                pass
        return None

    def _cache_put(self, spoken: str, pcm: np.ndarray) -> None:
        with self._cache_lock:
            self._cache[spoken] = pcm
        if self.cache_dir:
            try:
                (self.cache_dir / f"{self._cache_key(spoken)}.pcm").write_bytes(pcm.astype(np.int16).tobytes())
            except OSError:
                pass

    def prewarm(self, texts: list[str]) -> None:
        """Loads already-cached lines from disk into memory. Never calls
        the network (and so never spends credits): lines are synthesized
        the first time they're actually spoken, then cached."""
        for t in texts:
            self._cache_get(speech_text(t))

    # ---- synthesis -------------------------------------------------------------

    def _request_kwargs(self, text: str) -> dict:
        kwargs = dict(
            model_id=self.model_id,
            transcript=text,
            voice=self.voice_id,
            language=self.language,
            output_format={
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": audio_io.SAMPLE_RATE,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        # Only sent when tuned away from the default, so a stock request
        # shape is unchanged for anyone who never touches CARTESIA_SPEED.
        if self.speed != 1.0:
            kwargs["generation_config"] = {"speed": self.speed}
        return kwargs

    @staticmethod
    def _wrap_error(e: Exception) -> "CartesiaUnavailable":
        """Turns any SDK failure into a specific, actionable CartesiaUnavailable."""
        from cartesia import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
            APIStatusError,
        )

        if isinstance(e, AuthenticationError):
            return CartesiaUnavailable(f"Cartesia rejected CARTESIA_API_KEY (authentication failed): {e}")
        if isinstance(e, RateLimitError):
            return CartesiaUnavailable(f"Cartesia rate limit hit: {e}")
        if isinstance(e, APITimeoutError):
            return CartesiaUnavailable(f"Cartesia request timed out: {e}")
        if isinstance(e, APIConnectionError):
            return CartesiaUnavailable(f"Could not reach Cartesia (network error): {e}")
        if isinstance(e, APIStatusError):
            if e.status_code == 402:
                return CartesiaUnavailable(
                    "Cartesia credits are used up (HTTP 402). Top up or upgrade at "
                    "https://play.cartesia.ai/subscription, or run with --allow-tts-fallback "
                    f"for a basic offline voice: {e}"
                )
            return CartesiaUnavailable(f"Cartesia API returned an error (HTTP {e.status_code}): {e}")
        return CartesiaUnavailable(f"Cartesia TTS call failed: {e}")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Whole-clip synthesis: (pcm_int16 mono array, sample_rate). Raises
        CartesiaUnavailable on any failure. Sets self.last_synthesis_s."""
        spoken = speech_text(text)
        cached = self._cache_get(spoken)
        if cached is not None:
            self.last_synthesis_s = 0.0
            return cached, audio_io.SAMPLE_RATE

        t0 = time.time()
        try:
            response = self._client.tts.generate(**self._request_kwargs(spoken))
            raw_bytes = response.read()
            self.last_synthesis_s = time.time() - t0
        except Exception as e:  # noqa: BLE001
            raise self._wrap_error(e) from e

        pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        self._cache_put(spoken, pcm)
        return pcm, audio_io.SAMPLE_RATE

    def synthesize_stream(self, text: str):
        """Yields int16 PCM chunks as Cartesia generates them, so playback
        can start after ~0.15s instead of waiting for the whole sentence
        (measured: first chunk ~0.15s vs ~1.3s for the full clip). Cached
        lines are yielded at once. A fully received clip is cached; one cut
        short (interruption) is not. Raises CartesiaUnavailable on failure."""
        spoken = speech_text(text)
        cached = self._cache_get(spoken)
        if cached is not None:
            self.last_synthesis_s = 0.0
            yield cached
            return

        t0 = time.time()
        collected: list[np.ndarray] = []
        leftover = b""
        first = True
        try:
            with self._client.tts.with_streaming_response.generate(**self._request_kwargs(spoken)) as response:
                for chunk in response.iter_bytes(4096):
                    if first:
                        self.last_synthesis_s = time.time() - t0  # time to first audio
                        first = False
                    data = leftover + chunk
                    usable = len(data) - (len(data) % 2)  # int16 samples are 2 bytes
                    leftover = data[usable:]
                    if usable == 0:
                        continue
                    arr = np.frombuffer(data[:usable], dtype=np.int16).copy()
                    collected.append(arr)
                    yield arr
        except GeneratorExit:
            raise
        except Exception as e:  # noqa: BLE001
            raise self._wrap_error(e) from e

        if collected:
            self._cache_put(spoken, np.concatenate(collected))


def speak_pyttsx3(text: str) -> None:
    """Explicit offline fallback — Windows SAPI5 via pyttsx3. Quality and
    Hindi/Hinglish pronunciation are NOT guaranteed; this exists so a demo
    can still run without any API key, not as a stand-in for Cartesia."""
    import pyttsx3

    engine = pyttsx3.init()
    engine.say(speech_text(text))
    engine.runAndWait()
    engine.stop()


class TTSEngine:
    """Session-level TTS policy used by pooja_cli.py.

    - mode="cartesia" (the default for --mode voice): Cartesia is
      mandatory. If it's not configured or a call fails and
      allow_fallback=False, speak() raises CartesiaUnavailable — the
      session should stop rather than silently switch engines.
    - mode="cartesia" with allow_fallback=True: falls back to pyttsx3
      per-call on Cartesia failure, but every such fallback is reported
      in the returned SpeakResult.warning (never silent).
    - mode="pyttsx3": Cartesia is skipped entirely — an explicit,
      deliberate offline choice, not a masked failure.
    """

    def __init__(self, mode: str = "cartesia", allow_fallback: bool = False, cache_dir: str | None = None):
        if mode not in ("cartesia", "pyttsx3"):
            raise ValueError(f"Unknown TTS mode: {mode}")
        self.mode = mode
        self.allow_fallback = allow_fallback
        self.cartesia: CartesiaTTS | None = None
        if mode == "cartesia":
            self.cartesia = CartesiaTTS(cache_dir=cache_dir)  # raises CartesiaUnavailable if misconfigured

    def speak(self, text: str, play_fn=None) -> SpeakResult:
        """`play_fn(pcm, sample_rate) -> interrupted: bool` defaults to
        blocking playback with no barge-in. Returns a SpeakResult; raises
        CartesiaUnavailable if Cartesia fails and fallback isn't allowed."""
        play_fn = play_fn or (lambda pcm, sr: audio_io.play_pcm_int16(pcm, sr) or False)

        if self.mode == "pyttsx3":
            speak_pyttsx3(text)
            return SpeakResult(engine="pyttsx3")

        try:
            pcm, sr = self.cartesia.synthesize(text)
            t0 = time.time()
            interrupted = play_fn(pcm, sr)
            playback_s = time.time() - t0
            warning = None
            if not self.cartesia.model_id_verified:
                warning = (
                    f"CARTESIA_MODEL_ID '{self.cartesia.model_id}' is not in this SDK's "
                    f"known model list — the API accepted it, but verify it's current."
                )
            return SpeakResult(
                engine="cartesia",
                model_id=self.cartesia.model_id,
                voice_id=self.cartesia.voice_id,
                warning=warning,
                interrupted=interrupted,
                synthesis_s=self.cartesia.last_synthesis_s,
                playback_s=playback_s,
            )
        except CartesiaUnavailable as e:
            if not self.allow_fallback:
                raise
            speak_pyttsx3(text)
            return SpeakResult(engine="pyttsx3", warning=f"Cartesia failed this turn, used pyttsx3 fallback: {e}")


def list_hindi_female_voices(api_key: str | None = None, limit: int = 10) -> list[dict]:
    """Queries the Cartesia account's voice library for Hindi-capable,
    feminine-presenting voices, to help pick a CARTESIA_VOICE_ID. Used by
    `pooja_cli.py --find-voice`. Raises CartesiaUnavailable on failure.

    NOTE: `client.voices.list()` (per the installed SDK's real signature
    — `cartesia/resources/voices.py`) takes `gender`, `q`, `limit`, etc.
    but has NO `language` parameter, despite `Voice.language` existing
    on each returned object. So this fetches a larger-than-`limit` page
    filtered server-side by gender only, then filters by language
    client-side — not a guess, this was corrected after inspecting the
    SDK's actual method signature (an earlier version of this function
    incorrectly passed language= directly to list(), which would have
    raised a TypeError the first time someone ran --find-voice)."""
    key = api_key or os.environ.get("CARTESIA_API_KEY")
    if not key:
        raise CartesiaUnavailable("CARTESIA_API_KEY is not set.")

    from cartesia import Cartesia

    client = Cartesia(api_key=key)
    try:
        page = client.voices.list(gender="feminine", limit=max(limit * 5, 50))
        voices = [v for v in page if v.language == "hi"][:limit]
    except Exception as e:  # noqa: BLE001
        raise CartesiaUnavailable(f"Could not list Cartesia voices: {e}") from e

    return [
        {"id": v.id, "name": v.name, "tagline": v.tagline, "language": v.language}
        for v in voices
    ]
