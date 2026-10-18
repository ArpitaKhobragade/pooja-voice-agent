"""
Full-duplex audio for natural interruption ("barge-in").

One input stream (microphone) and one output stream (speaker) stay open for
the whole call. While Pooja is speaking, every microphone frame is checked
for the owner talking over her:

  * The owner's words are NOT lost: the frames from the moment they started
    talking are kept, and recording simply continues until they finish.
  * Her own voice leaking from the laptop speakers into the mic is not
    mistaken for the owner: the detector measures how strongly the output
    bleeds into the mic (an "echo coupling" factor) and only fires when the
    mic is clearly louder than that echo explains. With headphones the
    coupling is ~0 and interruption is very sensitive and reliable.
  * Playback stops within a few frames of the detection.

This is software echo *gating*, not true acoustic echo cancellation; the
README's honest caveat still applies (headphones are the most reliable).
"""

from __future__ import annotations

import collections
import os
import queue
import threading
import time

import numpy as np

from local import audio_io
from local.audio_io import (
    SAMPLE_RATE,
    RecordResult,
    analyze_pcm,
    AudioDeviceError,
)

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - environment dependent
    sd = None

BARGE_MIN_RMS = float(os.environ.get("POOJA_BARGE_MIN_RMS", "400"))
BARGE_FRAMES = int(os.environ.get("POOJA_BARGE_FRAMES", "9"))  # ~0.3s of sustained speech
ECHO_RATIO = float(os.environ.get("POOJA_BARGE_ECHO_RATIO", "1.8"))


def _rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


class BargeInDetector:
    """Pure decision logic (no audio I/O), so it is unit-testable.

    Call reset_turn() when playback starts, then update() once per
    microphone frame with the frame's RMS, the VAD verdict, and the recent
    output (speaker) RMS. update() returns True when the owner is talking
    over Pooja.
    """

    def __init__(self, frames_needed: int = BARGE_FRAMES, min_rms: float = BARGE_MIN_RMS,
                 echo_ratio: float = ECHO_RATIO, calib_frames: int = 14):
        self.frames_needed = frames_needed
        self.min_rms = min_rms
        self.echo_ratio = echo_ratio
        self.calib_frames = calib_frames
        self.coupling: float | None = None  # mic_rms / output_rms, learned
        self._calib: list[float] = []
        self._playing_frames = 0
        self._consecutive = 0

    def reset_turn(self) -> None:
        self._playing_frames = 0
        self._consecutive = 0
        self._calib = []

    def update(self, mic_rms: float, is_speech: bool, output_rms: float) -> bool:
        playing = output_rms > 150

        if playing:
            self._playing_frames += 1
            # Calibration window: the first moments of audible playback. The
            # owner is assumed silent then, so whatever the mic hears is
            # Pooja's own echo.
            if self._playing_frames <= self.calib_frames:
                self._calib.append(mic_rms / output_rms)
                if self._playing_frames == self.calib_frames:
                    learned = min(1.0, float(np.median(self._calib)))
                    # Smooth across turns so one noisy window can't wreck it.
                    self.coupling = learned if self.coupling is None else 0.5 * self.coupling + 0.5 * learned
                if self.coupling is None:
                    return False  # echo level still unknown: don't risk a false trigger

        echo = (self.coupling or 0.0) * output_rms
        threshold = max(self.min_rms, self.echo_ratio * echo + 150.0)
        if is_speech and mic_rms > threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= self.frames_needed


class DuplexAudio:
    """Owns the always-open microphone and speaker streams for a call."""

    requires_sounddevice = True  # the browser-backed subclass has no local audio device

    def __init__(self, vad, detector: BargeInDetector | None = None, debug: bool = False):
        if self.requires_sounddevice and sd is None:
            raise AudioDeviceError("sounddevice is not available")
        self.closed = False
        self.vad = vad
        self.frame_samples = vad.frame_samples
        self.detector = detector or BargeInDetector()
        self.debug = debug
        self.pending_result: RecordResult | None = None

        self._mic_q: queue.Queue = queue.Queue(maxsize=600)
        self._ring: collections.deque = collections.deque(maxlen=self.detector.frames_needed + 6)
        self._out_lock = threading.Lock()
        self._out_buf = np.zeros(0, dtype=np.int16)
        self._out_rms: collections.deque = collections.deque(maxlen=16)
        self._last_out_time = 0.0
        self._in_stream = None
        self._out_stream = None

    # ---- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        in_dev = audio_io._resolve_device(audio_io._INPUT_DEVICE_SPEC)
        out_dev = audio_io._resolve_device(audio_io._OUTPUT_DEVICE_SPEC)
        self._in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=self.frame_samples,
            device=in_dev, callback=self._on_mic, latency="low",
        )
        self._out_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=self.frame_samples,
            device=out_dev, callback=self._on_speaker, latency="low",
        )
        self._in_stream.start()
        self._out_stream.start()

    def close(self) -> None:
        self.closed = True
        for stream in (self._in_stream, self._out_stream):
            try:
                if stream is not None:
                    stream.stop()
                    stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._in_stream = self._out_stream = None

    # ---- stream callbacks (audio threads: keep them tiny) --------------------

    def _on_mic(self, indata, frames, time_info, status) -> None:
        frame = indata[:, 0].copy()
        try:
            self._mic_q.put_nowait(frame)
        except queue.Full:
            try:
                self._mic_q.get_nowait()
                self._mic_q.put_nowait(frame)
            except queue.Empty:
                pass

    def _on_speaker(self, outdata, frames, time_info, status) -> None:
        with self._out_lock:
            n = min(frames, len(self._out_buf))
            chunk = self._out_buf[:n]
            self._out_buf = self._out_buf[n:]
        outdata[:n, 0] = chunk
        outdata[n:, 0] = 0
        self._out_rms.append(_rms(chunk) if n else 0.0)
        if n:
            self._last_out_time = time.time()

    # ---- playback ------------------------------------------------------------

    def start_playback(self) -> None:
        """Call before the first enqueue() of a spoken turn."""
        self._drain_mic()
        self._ring.clear()
        self.detector.reset_turn()
        # Deliberately does NOT clear pending_result: an interruption captured
        # earlier must survive until get_user_input() takes it.

    def enqueue(self, pcm: np.ndarray) -> None:
        with self._out_lock:
            self._out_buf = np.concatenate([self._out_buf, pcm.astype(np.int16)])

    def buffered_seconds(self) -> float:
        with self._out_lock:
            return len(self._out_buf) / SAMPLE_RATE

    def drained(self) -> bool:
        """True once everything enqueued has actually been played."""
        return self.buffered_seconds() == 0.0 and (time.time() - self._last_out_time) > 0.2

    def stop_output(self) -> None:
        with self._out_lock:
            self._out_buf = np.zeros(0, dtype=np.int16)

    def service(self) -> bool:
        """Call every ~20ms during playback. Returns True if the owner
        interrupted: output is stopped and their utterance has been
        captured into self.pending_result."""
        prefix = self._poll_barge_in()
        if prefix is None:
            return False
        self.stop_output()
        self.pending_result = self._capture(prefix_frames=prefix)
        return True

    def _output_reference(self) -> float:
        """Recent loudness of what is being played (the echo reference)."""
        return max(self._out_rms) if self._out_rms else 0.0

    def _poll_barge_in(self):
        out_ref = self._output_reference()
        while True:
            try:
                frame = self._mic_q.get_nowait()
            except queue.Empty:
                return None
            self._ring.append(frame)
            if self.detector.update(_rms(frame), self.vad.is_speech(frame), out_ref):
                n = self.detector.frames_needed + 3
                if self.debug:
                    print(f"[DEBUG barge-in] owner interrupted (coupling={self.detector.coupling})")
                return list(self._ring)[-n:]

    # ---- listening -----------------------------------------------------------

    def take_pending(self) -> RecordResult | None:
        result, self.pending_result = self.pending_result, None
        return result

    def _drain_mic(self) -> None:
        try:
            while True:
                self._mic_q.get_nowait()
        except queue.Empty:
            pass

    def _capture(self, prefix_frames, silence_timeout_s=None, max_seconds=None) -> RecordResult:
        """Keeps recording after a barge-in until the owner stops talking."""
        silence_timeout_s = audio_io.SILENCE_TIMEOUT_S if silence_timeout_s is None else silence_timeout_s
        max_seconds = audio_io.MAX_RECORD_S if max_seconds is None else max_seconds
        frame_ms = self.frame_samples * 1000 / SAMPLE_RATE
        silence_needed = max(1, int(silence_timeout_s * 1000 / frame_ms))
        frames = list(prefix_frames)
        silence = 0
        t0 = time.time()
        timed_out = False
        while True:
            if self.closed:
                break
            if time.time() - t0 > max_seconds:
                timed_out = True
                break
            try:
                frame = self._mic_q.get(timeout=1.0)
            except queue.Empty:
                break
            frames.append(frame)
            if self.vad.is_speech(frame):
                silence = 0
            else:
                silence += 1
                if silence >= silence_needed:
                    break
        return self._result(frames, timed_out)

    def listen(self, pre_speech_timeout_s=None, silence_timeout_s=None, max_seconds=None) -> RecordResult | None:
        """Equivalent of audio_io.record_utterance() on the shared mic
        stream. Returns None if the owner says nothing in time."""
        pre_speech_timeout_s = audio_io.PRE_SPEECH_TIMEOUT_S if pre_speech_timeout_s is None else pre_speech_timeout_s
        frame_ms = self.frame_samples * 1000 / SAMPLE_RATE

        # Keep only the most recent ~0.4s that queued up while Pooja was
        # thinking, so an owner who started talking right as she finished
        # isn't cut off, but stale audio (her echo tail) is dropped.
        backlog = []
        try:
            while True:
                backlog.append(self._mic_q.get_nowait())
        except queue.Empty:
            pass
        backlog = backlog[-int(400 / frame_ms):]

        waited = 0.0
        pending = collections.deque(backlog)
        recent: collections.deque = collections.deque(maxlen=3)
        while True:
            if self.closed:
                return None
            if pending:
                frame = pending.popleft()
            else:
                try:
                    frame = self._mic_q.get(timeout=0.5)
                except queue.Empty:
                    waited += 0.5
                    if waited >= pre_speech_timeout_s:
                        return None
                    continue
            if self.vad.is_speech(frame):
                return self._capture(list(recent) + [frame], silence_timeout_s, max_seconds)
            recent.append(frame)
            waited += frame_ms / 1000
            if waited >= pre_speech_timeout_s:
                return None

    def _result(self, frames, timed_out: bool) -> RecordResult:
        pcm = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)
        a = analyze_pcm(pcm)
        return RecordResult(
            pcm_int16=pcm, sample_rate=SAMPLE_RATE, duration_s=a["duration_s"],
            vad_backend=self.vad.backend, timed_out=timed_out, rms=a["rms"], peak=a["peak"],
            is_silent=a["is_silent"], is_clipped=a["is_clipped"], too_short=a["too_short"],
        )
