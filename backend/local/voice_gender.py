"""
Estimates whether the owner sounds like a man or a woman, so Pooja can say
"Sir" or "Ma'am".

Two independent signals, text first:
  * Hindi/Hinglish first-person verb forms the owner says about THEMSELVES
    ("main bol rahi hoon" vs "main bol raha hoon").
  * Voice pitch: the median fundamental frequency (F0) of the voiced frames.
    Typical male speech is ~85-155 Hz, female ~165-255 Hz. The two ranges
    barely overlap, so there is an "uncertain" zone in between where no
    guess is made.

This is an estimate, not an identification: a low-pitched woman or a
high-pitched man can be misread, which is why the pitch needs enough
voiced audio and a clear margin before it is trusted, and why a text cue
overrides it.
"""

from __future__ import annotations

import re

import numpy as np

SAMPLE_RATE = 16000
_MIN_VOICED_FRAMES = 10       # ~0.2s of voiced speech needed before trusting pitch
_FEMALE_MIN_HZ = 178.0
_MALE_MAX_HZ = 150.0

_FEMALE_TEXT = re.compile(
    r"\b(main|mai|mein)\b[^.?!]{0,30}\b(rahi|karti|chahti|sakti|bolti|gayi|aayi|thi|hoon\s+na)\b", re.IGNORECASE
)
_FEMALE_TEXT_STRICT = re.compile(
    r"\b(main|mai|mein)\b[^.?!]{0,30}\b(rahi|karti|chahti|sakti|bolti|gayi|aayi)\b", re.IGNORECASE
)
_MALE_TEXT = re.compile(
    r"\b(main|mai|mein)\b[^.?!]{0,30}\b(raha|karta|chahta|sakta|bolta|gaya|aaya)\b", re.IGNORECASE
)


def gender_from_text(text: str) -> str | None:
    """'female' / 'male' from first-person verb endings, else None."""
    female = bool(_FEMALE_TEXT_STRICT.search(text))
    male = bool(_MALE_TEXT.search(text))
    if female and not male:
        return "female"
    if male and not female:
        return "male"
    return None


def voiced_f0s(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> list[float]:
    """Per-frame fundamental frequency (Hz) for the voiced frames of a clip,
    by autocorrelation. Unvoiced / quiet frames are skipped."""
    pcm = np.asarray(pcm)
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    lo, hi = int(sr / 400), int(sr / 70)
    window = np.hanning(frame)
    out: list[float] = []
    for start in range(0, len(pcm) - frame, hop):
        x = pcm[start:start + frame].astype(np.float64)
        x -= x.mean()
        if np.sqrt(np.mean(x ** 2)) < 220:  # too quiet to be speech on a normal mic
            continue
        x *= window
        spec = np.fft.rfft(x, 2 * frame)
        ac = np.fft.irfft(spec * np.conj(spec))[:frame]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] <= 0.5:  # not strongly periodic => not voiced
            continue
        # Autocorrelation often locks onto a multiple of the true period, which halves the pitch
        # and turns a woman's voice into a man's. Prefer the shorter period when it is nearly as strong.
        for div in (3, 2):
            cand = lag // div
            if cand >= lo:
                window_lo, window_hi = max(lo, cand - 2), min(hi, cand + 3)
                best = window_lo + int(np.argmax(ac[window_lo:window_hi]))
                if ac[best] >= 0.85 * ac[lag]:
                    lag = best
                    break
        out.append(sr / lag)
    return out


class GenderTracker:
    """Accumulates evidence across the whole call."""

    def __init__(self):
        self._f0s: list[float] = []
        self._text_vote: str | None = None

    def add_audio(self, pcm: np.ndarray, sr: int = SAMPLE_RATE) -> None:
        self._f0s.extend(voiced_f0s(pcm, sr))
        self._f0s = self._f0s[-600:]

    def add_text(self, text: str) -> None:
        vote = gender_from_text(text)
        if vote:
            self._text_vote = vote

    @property
    def median_f0(self) -> float | None:
        return float(np.median(self._f0s)) if len(self._f0s) >= _MIN_VOICED_FRAMES else None

    @property
    def guess(self) -> str | None:
        if self._text_vote:
            return self._text_vote
        f0 = self.median_f0
        if f0 is None:
            return None
        if f0 >= _FEMALE_MIN_HZ:
            return "female"
        if f0 <= _MALE_MAX_HZ:
            return "male"
        return None
