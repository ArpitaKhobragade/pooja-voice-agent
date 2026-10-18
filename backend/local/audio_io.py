"""
Microphone capture and speaker playback for the local Pooja CLI.

Turn-taking model (be honest about what this is):
  This is TURN-BASED recording with silence-based endpointing, not true
  full-duplex streaming ASR/TTS. Pooja speaks, then listens until she
  detects the owner has stopped talking (VAD + a silence timeout), then
  sends the whole utterance to STT at once.

  Barge-in (best-effort, NOT guaranteed): while Pooja is speaking,
  `play_interruptible()` also runs the VAD against live mic input. If it
  sees sustained speech partway through playback, it stops playback
  immediately and returns early so the owner's turn can start. Without
  echo cancellation, a laptop's own speakers bleeding into its mic can
  trigger a false interruption — headphones make this much more
  reliable (and avoid the problem entirely). This is documented, not
  hidden: it is best-effort, not verified hands-free hardware AEC.

VAD backend, in priority order (first one importable wins, forced via
POOJA_VAD_BACKEND=silero|webrtcvad|energy):
  1. Silero VAD (`pip install silero-vad torch`) — most accurate, but
     pulls in PyTorch (a very large, slow-to-install dependency). NOT
     in requirements.txt by default for that reason — it's an opt-in
     upgrade, consistent with the project's cost/size-conscious design.
  2. webrtcvad (prebuilt Windows wheel: webrtcvad-wheels) — accurate,
     tiny, no model download. This is the default in requirements.txt.
  3. Simple RMS energy threshold — always available, least accurate.
"""

from __future__ import annotations

import io
import os
import time
import wave
from dataclasses import dataclass

import numpy as np

try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except Exception as e:  # pragma: no cover - environment dependent
    _SOUNDDEVICE_AVAILABLE = False
    _SOUNDDEVICE_IMPORT_ERROR = e

SAMPLE_RATE = 16000  # Hz — matches Sarvam STT's expected input
_FORCED_BACKEND = os.environ.get("POOJA_VAD_BACKEND", "").strip().lower() or None

# Device selection: POOJA_INPUT_DEVICE / POOJA_OUTPUT_DEVICE can be a
# sounddevice device index (e.g. "2") or a case-insensitive substring of
# the device name (e.g. "USB Microphone"). Unset (default) uses
# sounddevice's own default device, same as before this option existed.
_INPUT_DEVICE_SPEC = os.environ.get("POOJA_INPUT_DEVICE", "").strip() or None
_OUTPUT_DEVICE_SPEC = os.environ.get("POOJA_OUTPUT_DEVICE", "").strip() or None

# Listening timing, overridable per the "STT keeps cutting me off" /
# "normal pauses get cut" class of problem without editing code.
SILENCE_TIMEOUT_S = float(os.environ.get("POOJA_SILENCE_TIMEOUT_S", "1.5"))
PRE_SPEECH_TIMEOUT_S = float(os.environ.get("POOJA_PRE_SPEECH_TIMEOUT_S", "8.0"))
MAX_RECORD_S = float(os.environ.get("POOJA_MAX_RECORD_S", "20.0"))
MIN_SPEECH_S = float(os.environ.get("POOJA_MIN_SPEECH_S", "0.25"))

# Below this RMS (on a 0-32767 int16 scale), a clip is treated as
# effectively silent — not real speech, don't waste a Sarvam call on it.
SILENCE_RMS_THRESHOLD = float(os.environ.get("POOJA_SILENCE_RMS_THRESHOLD", "80"))
# Above this fraction of samples sitting within 1% of full-scale, the
# clip is flagged as likely clipped (mic gain too high).
CLIPPING_FRACTION_THRESHOLD = 0.02


def _resolve_device(spec: str | None):
    """Turns a POOJA_*_DEVICE spec (index or name substring) into a
    sounddevice device index, or None (= sounddevice's own default)."""
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    devices = sd.query_devices()
    spec_low = spec.lower()
    for i, d in enumerate(devices):
        if spec_low in d["name"].lower():
            return i
    raise AudioDeviceError(
        f"No audio device name matched '{spec}'. Run 'python pooja_cli.py "
        f"--list-devices' to see exact names/indices."
    )


def describe_devices() -> str:
    """Human-readable summary of which input/output device will actually
    be used and at what sample rate — printed at startup so a mic
    misconfiguration is visible immediately, not discovered mid-call."""
    if not _SOUNDDEVICE_AVAILABLE:
        return f"(audio unavailable: {audio_unavailable_reason()})"
    try:
        in_idx = _resolve_device(_INPUT_DEVICE_SPEC)
        out_idx = _resolve_device(_OUTPUT_DEVICE_SPEC)
        in_info = sd.query_devices(in_idx, "input") if in_idx is not None else sd.query_devices(kind="input")
        out_info = sd.query_devices(out_idx, "output") if out_idx is not None else sd.query_devices(kind="output")
        return (
            f"Input: {in_info['name']!r} (device default sample rate "
            f"{in_info['default_samplerate']:.0f}Hz, capturing at {SAMPLE_RATE}Hz) | "
            f"Output: {out_info['name']!r}"
        )
    except Exception as e:  # noqa: BLE001 - this is diagnostic text, never fatal
        return f"(could not describe audio devices: {e})"


class AudioDeviceError(RuntimeError):
    """Raised when no usable microphone/speaker is available."""


def audio_available() -> bool:
    return _SOUNDDEVICE_AVAILABLE


def audio_unavailable_reason() -> str:
    if _SOUNDDEVICE_AVAILABLE:
        return ""
    return str(_SOUNDDEVICE_IMPORT_ERROR)


def list_devices() -> str:
    if not _SOUNDDEVICE_AVAILABLE:
        raise AudioDeviceError(f"sounddevice not available: {audio_unavailable_reason()}")
    return str(sd.query_devices())


def check_devices() -> None:
    """Raises AudioDeviceError with a clear message if no input/output
    device is usable. Call this before starting the voice loop so failures
    surface immediately instead of mid-conversation."""
    if not _SOUNDDEVICE_AVAILABLE:
        raise AudioDeviceError(
            f"The 'sounddevice' package (and its PortAudio backend) is not available: "
            f"{audio_unavailable_reason()}. Install it with 'pip install sounddevice' "
            f"and make sure a microphone/speaker are connected."
        )
    try:
        devices = sd.query_devices()
    except Exception as e:
        raise AudioDeviceError(f"Could not query audio devices: {e}") from e

    has_input = any(d["max_input_channels"] > 0 for d in devices)
    has_output = any(d["max_output_channels"] > 0 for d in devices)
    if not has_input:
        raise AudioDeviceError("No microphone (input device) detected.")
    if not has_output:
        raise AudioDeviceError("No speaker (output device) detected.")


class _SileroBackend:
    """512-sample (32ms @16kHz) frames — Silero's required chunk size."""

    frame_samples = 512

    def __init__(self, threshold: float = 0.5):
        import torch  # noqa: F401 - import error here is how we detect "not installed"
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()
        self._threshold = threshold

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        audio_f32 = frame_int16.astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(audio_f32)
        with self._torch.no_grad():
            prob = self._model(tensor, SAMPLE_RATE).item()
        return prob > self._threshold


class _WebRTCBackend:
    """480-sample (30ms @16kHz) frames — a valid webrtcvad frame size."""

    frame_samples = 480

    def __init__(self, aggressiveness: int = 2):
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        return self._vad.is_speech(frame_int16.tobytes(), SAMPLE_RATE)


class _EnergyBackend:
    """480-sample (30ms @16kHz) frames — arbitrary but matches webrtcvad's
    for consistent timing math when this is the fallback."""

    frame_samples = 480

    def __init__(self, energy_threshold: float = 500.0):
        self._energy_threshold = energy_threshold

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame_int16.astype(np.float64) ** 2)))
        return rms > self._energy_threshold


_BACKEND_BUILDERS = {
    "silero": (_SileroBackend, "silero"),
    "webrtcvad": (_WebRTCBackend, "webrtcvad"),
    "energy": (_EnergyBackend, "energy-threshold"),
}


class FrameVAD:
    """Picks the best available VAD backend (or a forced one via
    POOJA_VAD_BACKEND), exposing a uniform is_speech(frame) -> bool and
    the frame_samples that backend expects."""

    def __init__(self, vad_aggressiveness: int = 2):
        order = [_FORCED_BACKEND] if _FORCED_BACKEND else ["silero", "webrtcvad", "energy"]
        order = [b for b in order if b in _BACKEND_BUILDERS] or ["energy"]

        self._impl = None
        self._backend_name = None
        errors = []
        for name in order:
            cls, label = _BACKEND_BUILDERS[name]
            try:
                self._impl = cls(vad_aggressiveness) if name == "webrtcvad" else cls()
                self._backend_name = label
                break
            except Exception as e:  # noqa: BLE001 - try the next backend
                errors.append(f"{name}: {e}")
                continue

        if self._impl is None:
            # Should never happen since "energy" has no external deps, but
            # fail loudly instead of silently if it somehow does.
            raise AudioDeviceError(f"No VAD backend could be initialized: {'; '.join(errors)}")

    @property
    def backend(self) -> str:
        return self._backend_name

    @property
    def frame_samples(self) -> int:
        return self._impl.frame_samples

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        return self._impl.is_speech(frame_int16)


@dataclass
class RecordResult:
    pcm_int16: np.ndarray  # mono, 16kHz, int16
    sample_rate: int
    duration_s: float
    vad_backend: str
    timed_out: bool  # True if max_seconds was hit without detecting silence
    rms: float  # root-mean-square amplitude of the whole clip (0-32767 scale)
    peak: int  # max absolute sample value (0-32767)
    is_silent: bool  # RMS below SILENCE_RMS_THRESHOLD — likely not real speech
    is_clipped: bool  # a notable fraction of samples near full-scale — mic gain too high
    too_short: bool  # shorter than MIN_SPEECH_S — not discarded, but worth flagging


def record_utterance(
    max_seconds: float | None = None,
    silence_timeout_s: float | None = None,
    pre_speech_timeout_s: float | None = None,
    vad_aggressiveness: int = 2,
    vad: "FrameVAD | None" = None,
) -> RecordResult | None:
    """Records one spoken turn from the microphone.

    Waits up to `pre_speech_timeout_s` for speech to start; if nothing is
    said, returns None (caller should treat this as "no input" and
    re-prompt, not as an error). Once speech starts, keeps recording until
    `silence_timeout_s` of continuous silence is detected, or
    `max_seconds` total is reached. Defaults come from
    POOJA_SILENCE_TIMEOUT_S / POOJA_PRE_SPEECH_TIMEOUT_S / POOJA_MAX_RECORD_S
    (see module top) so timing can be tuned without editing code — e.g. if
    normal speaking pauses are getting cut off, raise
    POOJA_SILENCE_TIMEOUT_S in .env rather than changing Python.

    A short clip is NOT discarded just for being short (a real "haan" or
    "nahi" is valid) — but the result's `is_silent` / `is_clipped` /
    `too_short` flags let the caller decide whether to trust the
    transcript Sarvam returns for it (see pooja_cli.py's suspicious-
    transcript handling).
    """
    if not _SOUNDDEVICE_AVAILABLE:
        raise AudioDeviceError(
            f"sounddevice not available: {audio_unavailable_reason()}"
        )

    max_seconds = MAX_RECORD_S if max_seconds is None else max_seconds
    silence_timeout_s = SILENCE_TIMEOUT_S if silence_timeout_s is None else silence_timeout_s
    pre_speech_timeout_s = PRE_SPEECH_TIMEOUT_S if pre_speech_timeout_s is None else pre_speech_timeout_s

    vad = vad or FrameVAD(vad_aggressiveness=vad_aggressiveness)
    frame_samples = vad.frame_samples
    frame_ms = frame_samples * 1000 / SAMPLE_RATE
    frames: list[np.ndarray] = []
    silence_frames_needed = max(1, int(silence_timeout_s * 1000 / frame_ms))
    pre_speech_frames_needed = max(1, int(pre_speech_timeout_s * 1000 / frame_ms))

    speech_started = False
    consecutive_silence = 0
    consecutive_pre_speech = 0
    t0 = time.time()
    timed_out = False
    device = _resolve_device(_INPUT_DEVICE_SPEC)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=frame_samples, device=device
    ) as stream:
        while True:
            if time.time() - t0 > max_seconds:
                timed_out = True
                break

            block, _overflow = stream.read(frame_samples)
            frame = block[:, 0]

            is_speech = vad.is_speech(frame)

            if not speech_started:
                if is_speech:
                    speech_started = True
                    frames.append(frame)
                    consecutive_silence = 0
                else:
                    consecutive_pre_speech += 1
                    if consecutive_pre_speech >= pre_speech_frames_needed:
                        return None
                continue

            frames.append(frame)
            if is_speech:
                consecutive_silence = 0
            else:
                consecutive_silence += 1
                if consecutive_silence >= silence_frames_needed:
                    break

    if not frames:
        return None

    pcm = np.concatenate(frames)
    analysis = analyze_pcm(pcm)

    return RecordResult(
        pcm_int16=pcm,
        sample_rate=SAMPLE_RATE,
        duration_s=analysis["duration_s"],
        vad_backend=vad.backend,
        timed_out=timed_out,
        rms=analysis["rms"],
        peak=analysis["peak"],
        is_silent=analysis["is_silent"],
        is_clipped=analysis["is_clipped"],
        too_short=analysis["too_short"],
    )


def analyze_pcm(pcm: np.ndarray) -> dict:
    """Pure signal-level analysis of a recorded clip — no mic/network
    I/O, so it's directly unit-testable with synthetic arrays. Used by
    record_utterance() on real audio and by tests on crafted audio
    (silence, clipping, short clips) to verify the detection logic
    itself, separately from whether a real microphone works."""
    duration_s = len(pcm) / SAMPLE_RATE
    if len(pcm) == 0:
        return {
            "duration_s": 0.0, "rms": 0.0, "peak": 0,
            "is_silent": True, "is_clipped": False, "too_short": True,
        }

    pcm_f = pcm.astype(np.float64)
    rms = float(np.sqrt(np.mean(pcm_f ** 2)))
    peak = int(np.max(np.abs(pcm)))
    near_fullscale = np.abs(pcm.astype(np.int64)) > int(32767 * 0.99)
    clipped_fraction = float(np.mean(near_fullscale))

    return {
        "duration_s": duration_s,
        "rms": rms,
        "peak": peak,
        "is_silent": rms < SILENCE_RMS_THRESHOLD,
        "is_clipped": clipped_fraction > CLIPPING_FRACTION_THRESHOLD,
        "too_short": duration_s < MIN_SPEECH_S,
    }


def trim_trailing_silence(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE, keep_s: float = 0.25) -> np.ndarray:
    """Drops the silent tail that endpointing leaves at the end of a clip
    (it waits for silence before stopping), keeping `keep_s` after the last
    sound. A shorter upload makes speech-to-text return sooner. Pure and
    conservative: if nothing clearly louder than the tail is found, the clip
    is returned unchanged."""
    frame = max(1, int(sample_rate * 0.02))
    n = len(pcm) // frame
    if n < 4:
        return pcm
    rms = np.sqrt(np.mean(pcm[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    loud = float(np.percentile(rms, 90))
    if loud < 150:
        return pcm
    voiced = np.nonzero(rms > max(100.0, 0.15 * loud))[0]
    if len(voiced) == 0:
        return pcm
    end = min(len(pcm), (int(voiced[-1]) + 1) * frame + int(keep_s * sample_rate))
    return pcm[:end]


def pcm_int16_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Wraps raw mono int16 PCM in a standard WAV container. Shared by
    stt_sarvam.py (building the file it uploads) and audition_server.py
    (building the file it serves to the browser's <audio> element) so
    there's one WAV-writing implementation, not two."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def play_pcm_int16(pcm: np.ndarray, sample_rate: int) -> None:
    """Blocks until playback finishes so turns don't overlap."""
    if not _SOUNDDEVICE_AVAILABLE:
        raise AudioDeviceError(
            f"sounddevice not available: {audio_unavailable_reason()}"
        )
    sd.play(pcm, samplerate=sample_rate, device=_resolve_device(_OUTPUT_DEVICE_SPEC), blocking=True)


def play_interruptible(
    pcm: np.ndarray,
    sample_rate: int,
    vad: "FrameVAD | None" = None,
    grace_period_s: float = 0.5,
    interrupt_frames_needed: int = 4,
    poll_interval_s: float = 0.05,
) -> bool:
    """Plays `pcm` while concurrently watching the microphone for speech.
    Returns True if playback was stopped early because the owner started
    talking (best-effort barge-in — see module docstring for the caveat
    about echo/feedback without headphones), False if playback finished
    normally.

    `grace_period_s` ignores the first moment of playback (avoids
    reacting to the tail of the owner's own previous turn or playback
    startup transients). `interrupt_frames_needed` requires that many
    consecutive speech-positive mic frames before declaring an
    interruption, to reduce false positives from brief noise/echo.
    """
    if not _SOUNDDEVICE_AVAILABLE:
        raise AudioDeviceError(
            f"sounddevice not available: {audio_unavailable_reason()}"
        )

    vad = vad or FrameVAD()
    frame_samples = vad.frame_samples

    sd.play(pcm, samplerate=sample_rate, device=_resolve_device(_OUTPUT_DEVICE_SPEC), blocking=False)
    t0 = time.time()
    consecutive_speech = 0
    interrupted = False

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=frame_samples,
            device=_resolve_device(_INPUT_DEVICE_SPEC),
        ) as stream:
            while sd.get_stream().active:
                if time.time() - t0 < grace_period_s:
                    stream.read(frame_samples)  # drain without evaluating
                    continue

                block, _overflow = stream.read(frame_samples)
                frame = block[:, 0]
                if vad.is_speech(frame):
                    consecutive_speech += 1
                    if consecutive_speech >= interrupt_frames_needed:
                        interrupted = True
                        break
                else:
                    consecutive_speech = 0
    finally:
        if interrupted:
            sd.stop()
        else:
            sd.wait()

    return interrupted
