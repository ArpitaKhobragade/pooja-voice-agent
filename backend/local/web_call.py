"""
One browser call: runs Pooja's conversation (PoojaSession) in a worker
thread, reading the owner's microphone frames from, and sending Pooja's
speech and transcript events to, a browser over a WebSocket.

The conversation engine is exactly the one the local microphone mode uses;
only the audio device is replaced (WebDuplex) and every turn is mirrored to
the page as a transcript event.

Events sent to the browser (JSON):
  {"type":"ready"}                              the call has started
  {"type":"status","state":"listening|thinking|speaking"}
  {"type":"transcript","speaker":"pooja|owner","turn":n,"text":"...","t":seconds}
  {"type":"interrupted","turn":n}               the owner talked over Pooja
  {"type":"clear"}                              drop queued audio now
  {"type":"notice","message":"..."}             non-fatal hiccup
  {"type":"error","message":"..."}              the call cannot continue
  {"type":"ended","reason":"...","outcome":"..."}
Binary messages to the browser are Pooja's speech: PCM int16, mono, 16 kHz.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

from loguru import logger

from local import audio_io

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tts_cache")
REQUIRED_KEYS = ("GROQ_API_KEY", "SARVAM_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID")


def missing_config() -> list[str]:
    """Names (never values) of required settings that are not set."""
    return [k for k in REQUIRED_KEYS if not os.environ.get(k, "").strip()]


class WebCall:
    def __init__(
        self,
        emit_json: Callable[[dict], None],
        emit_bytes: Callable[[bytes], None],
        *,
        max_call_s: float = 420.0,
        on_finished: Callable[[], None] | None = None,
        llm_factory=None,
        tts_factory=None,
        vad_factory=None,
        stt=None,
        timezone: str | None = None,
    ):
        self.timezone = timezone  # the caller's own timezone (from the browser): greeting and closing wish follow it
        self.emit_json = emit_json
        self.emit_bytes = emit_bytes
        self.max_call_s = max_call_s
        self.on_finished = on_finished
        self._llm_factory = llm_factory
        self._tts_factory = tts_factory
        self._vad_factory = vad_factory
        self._stt = stt
        self.duplex = None
        self.session = None
        self._thread: threading.Thread | None = None
        self._stop_reason: str | None = None
        self._t0 = time.time()
        self._turn = 0
        self._last_status: str | None = None
        self._last_transcript: str | None = None

    # ---- control (called from the WebSocket handler) -------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="web-call", daemon=True)
        self._thread.start()

    def feed_mic(self, data: bytes) -> None:
        if self.duplex is not None:
            self.duplex.feed_mic(data)

    def stop(self, reason: str = "stopped") -> None:
        self._stop_reason = self._stop_reason or reason
        if self.session is not None:
            self.session.stop_event.set()
        if self.duplex is not None:
            self.duplex.close()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # ---- the call ------------------------------------------------------------

    def _elapsed(self) -> float:
        return round(time.time() - self._t0, 1)

    def _status(self, state: str) -> None:
        if state != self._last_status:
            self._last_status = state
            self.emit_json({"type": "status", "state": state})

    def _run(self) -> None:
        reason, outcome = "completed", "none"
        try:
            from agent_config import set_call_timezone

            set_call_timezone(self.timezone)  # this thread's clock: the caller's, not the server's
            missing = missing_config()
            if missing and self._llm_factory is None:
                self.emit_json({"type": "error", "message": "The server is missing settings: " + ", ".join(missing)})
                reason = "error"
                return
            self._build_and_run()
            self._let_speech_finish()
            reason = self._stop_reason or "completed"
            outcome = self._outcome()
        except Exception as e:  # noqa: BLE001 - reported to the page, never crashes the server
            logger.exception("Web call failed")
            self.emit_json({"type": "error", "message": self._friendly(e)})
            reason = "error"
        finally:
            if self.duplex is not None:
                self.duplex.close()
            self.emit_json({"type": "ended", "reason": reason, "outcome": outcome, "seconds": self._elapsed()})
            if self.on_finished:
                self.on_finished()

    def _let_speech_finish(self) -> None:
        """When Pooja ends the call herself (her goodbye is the last thing said), keep the
        connection open until the browser has played it, so the last line is never cut off."""
        if self._stop_reason or self.duplex is None:
            return
        deadline = time.time() + 20.0
        while time.time() < deadline and not self.duplex.closed:
            wait = self.duplex.buffered_seconds()
            if wait <= 0:
                break
            time.sleep(min(wait, 0.5))
        if not self.duplex.closed:
            time.sleep(self.TAIL_S)

    TAIL_S = 0.8  # network and browser buffering allowance after the last sample is sent

    @staticmethod
    def _friendly(e: Exception) -> str:
        text = str(e)
        if "credits" in text.lower() or "402" in text:
            return "The voice service has run out of credits. Add credits and start again."
        if "Cartesia" in text or "TTS" in text:
            return "The voice service is unavailable right now: " + text[:160]
        return "Something went wrong on the server. Please start the call again."

    def _outcome(self) -> str:
        if self.session is None:
            return "none"
        state = self.session.cm.get_state()
        if state.interested_confirmed:
            return "interested"
        if state.not_interested:
            return "not_interested"
        if state.call_ended:
            return "callback_or_busy"
        return "none"

    def _build_and_run(self) -> None:
        # Imported here so importing this module stays light (and testable).
        from local import stt_sarvam
        from local.playback import play_sentences
        from local.web_audio import WebDuplex, default_web_vad
        from pooja_cli import PoojaSession, _transcript_is_unreliable

        llm = self._llm_factory() if self._llm_factory else _default_llm()
        tts = self._tts_factory() if self._tts_factory else _default_tts()
        vad = self._vad_factory() if self._vad_factory else default_web_vad()
        stt = self._stt or stt_sarvam.transcribe

        self.duplex = WebDuplex(
            vad,
            send_audio=self.emit_bytes,
            send_clear=lambda: self.emit_json({"type": "clear"}),
        )
        if self._stop_reason:  # Stop was pressed while the call was still being set up
            self.duplex.close()
            return

        def get_user_input():
            get_user_input.none_reason = "unclear"
            get_user_input.last_audio = None
            if time.time() - self._t0 > self.max_call_s:
                self.stop("time_limit")
                return None
            result = self.duplex.take_pending()
            if result is None:
                self._status("listening")
                result = self.duplex.listen()
            if result is None:
                get_user_input.none_reason = "silence"
                return None
            self._status("thinking")
            if result.is_silent:
                return None
            clip = audio_io.trim_trailing_silence(result.pcm_int16, result.sample_rate)
            get_user_input.last_audio = clip
            try:
                text = stt(clip, result.sample_rate)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"STT failed: {e}")
                self.emit_json({"type": "notice", "message": "Could not make out that last bit. Say it again."})
                return None
            if not text:
                return None
            unreliable, _why = _transcript_is_unreliable(text, result.duration_s, self._last_transcript)
            self._last_transcript = text
            if unreliable:
                return None
            self.emit_json({"type": "transcript", "speaker": "owner", "turn": self._turn + 1, "text": text,
                            "t": self._elapsed()})
            return text

        def deliver_stream(sentences):
            self._turn += 1
            turn = self._turn
            text, interrupted = play_sentences(
                sentences, tts.cartesia, self.duplex,
                allow_fallback=False,
                on_sentence=lambda s: self.emit_json(
                    {"type": "transcript", "speaker": "pooja", "turn": turn, "text": s, "t": self._elapsed()}
                ),
                on_first_audio=lambda *_: self._status("speaking"),
            )
            if interrupted:
                self.emit_json({"type": "interrupted", "turn": turn})
            return text, interrupted

        def deliver_reply(text: str) -> bool:
            return deliver_stream([text])[1]

        self.session = PoojaSession(
            llm, get_user_input, deliver_reply, status=self._status, conversational=True,
            prewarm=tts.cartesia.prewarm, deliver_stream=deliver_stream,
        )
        if self._stop_reason:
            self.session.stop_event.set()
        self.emit_json({"type": "ready"})
        self.session.run()


def _default_llm():
    from local.llm_groq import GroqLLM

    return GroqLLM()


def _default_tts():
    from local.tts_cartesia import TTSEngine

    return TTSEngine(mode="cartesia", allow_fallback=False, cache_dir=CACHE_DIR)
