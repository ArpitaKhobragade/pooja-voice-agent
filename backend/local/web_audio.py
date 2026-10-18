"""
Browser-backed audio for Pooja: the same DuplexAudio interface the local
microphone mode uses, but the microphone frames arrive over a WebSocket and
the speech audio is sent back over it. The VAD, interruption detection and
capture logic are inherited unchanged.

The browser plays the audio, so this class keeps a *virtual* playback clock
(when the audio it has sent should finish) to answer buffered_seconds() /
drained(). The browser also cancels echo (getUserMedia echoCancellation),
which makes interruption reliable without headphones.
"""

from __future__ import annotations

import collections
import os
import time

import numpy as np

from local.duplex import BargeInDetector, DuplexAudio, _rms
from local.audio_io import SAMPLE_RATE


class GatedVAD:
    """A speech detector that also requires the frame to be clearly louder than
    background noise. The small server-side detector (webrtcvad) on its own can
    flag steady hiss or room noise as speech, which would stop Pooja ever hearing
    you finish; the browser has already applied noise suppression and automatic
    gain, so real speech is comfortably above this floor."""

    def __init__(self, inner, min_rms: float | None = None):
        self._inner = inner
        self.min_rms = float(os.environ.get("POOJA_WEB_VAD_MIN_RMS", "250")) if min_rms is None else min_rms
        self.frame_samples = inner.frame_samples
        self.backend = f"{inner.backend}+gate"

    def is_speech(self, frame) -> bool:
        return _rms(frame) >= self.min_rms and self._inner.is_speech(frame)


def default_web_vad() -> "GatedVAD":
    from local import audio_io

    # aggressiveness 3 = strictest webrtcvad setting (ignored by the other backends)
    return GatedVAD(audio_io.FrameVAD(vad_aggressiveness=3))


class WebDuplex(DuplexAudio):
    requires_sounddevice = False

    def __init__(self, vad, send_audio, send_clear, detector: BargeInDetector | None = None, debug: bool = False):
        """send_audio(bytes): PCM int16 mono 16 kHz to the browser.
        send_clear(): tell the browser to drop any audio it has queued."""
        super().__init__(vad, detector, debug)
        self._send_audio = send_audio
        self._send_clear = send_clear
        self._frame_buf = np.zeros(0, dtype=np.int16)
        self._play_end = 0.0
        self._chunks: collections.deque = collections.deque()  # (start, end, rms)

    # ---- lifecycle ---------------------------------------------------------

    def open(self) -> None:  # nothing to open: the WebSocket is the device
        pass

    def close(self) -> None:
        self.closed = True

    # ---- microphone (called from the WebSocket receiver) ---------------------

    def feed_mic(self, data: bytes) -> None:
        """Accepts raw int16 PCM of any length and re-frames it for the VAD."""
        if self.closed or not data:
            return
        usable = len(data) - (len(data) % 2)
        self._frame_buf = np.concatenate([self._frame_buf, np.frombuffer(data[:usable], dtype=np.int16)])
        n = self.frame_samples
        while len(self._frame_buf) >= n:
            frame, self._frame_buf = self._frame_buf[:n].copy(), self._frame_buf[n:]
            self._push_mic(frame)

    def _push_mic(self, frame: np.ndarray) -> None:
        import queue

        try:
            self._mic_q.put_nowait(frame)
        except queue.Full:
            try:
                self._mic_q.get_nowait()
                self._mic_q.put_nowait(frame)
            except queue.Empty:
                pass

    # ---- speaker -------------------------------------------------------------

    def enqueue(self, pcm: np.ndarray) -> None:
        pcm = pcm.astype(np.int16)
        now = time.time()
        start = max(now, self._play_end)
        end = start + len(pcm) / SAMPLE_RATE
        self._play_end = end
        self._chunks.append((start, end, _rms(pcm)))
        self._send_audio(pcm.tobytes())

    def buffered_seconds(self) -> float:
        return max(0.0, self._play_end - time.time())

    def drained(self) -> bool:
        return time.time() >= self._play_end + 0.25

    def stop_output(self) -> None:
        self._play_end = 0.0
        self._chunks.clear()
        self._send_clear()

    def _output_reference(self) -> float:
        """Loudness of what the browser is playing right now (echo reference)."""
        now = time.time()
        while self._chunks and self._chunks[0][1] < now - 0.4:
            self._chunks.popleft()
        recent = [rms for (start, end, rms) in self._chunks if start <= now and end >= now - 0.35]
        return max(recent) if recent else 0.0
