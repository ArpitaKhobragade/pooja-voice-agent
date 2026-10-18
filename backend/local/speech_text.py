"""
Rewrites display text into what the TTS engine should actually say.

Cartesia reads "HostelPGDekho.com" letter by letter (h-o-s-t-e-l-p-g...).
Spacing the brand into words fixes that without touching the on-screen
text or the transcript, which keep the original spelling.
"""

from __future__ import annotations

import os
import re

_BRAND_RE = re.compile(r"hostel\s*[-_ ]?\s*pg\s*[-_ ]?\s*dekho(\.com)?", re.IGNORECASE)
_RANGE_RE = re.compile(r"(\d+)\s*[-‐‑‒–—]\s*(\d+)")
_URL_RE =re.compile(r"https?://(?:www\.)?", re.IGNORECASE)


# How the voice should say the brand. Devanagari makes the Hindi voice read it as
# one smooth name ("hostel PG dekho") instead of "hostelpg ... dekho". Override in
# .env (POOJA_BRAND_SPOKEN / POOJA_DOTCOM_SPOKEN) to try other spellings, e.g.
# POOJA_BRAND_SPOKEN="Hostel PG Dekho".
BRAND_SPOKEN = os.environ.get("POOJA_BRAND_SPOKEN", "होस्टल पीजी देखो")
DOTCOM_SPOKEN = os.environ.get("POOJA_DOTCOM_SPOKEN", "डॉट कॉम")
SLASH_SPOKEN = os.environ.get("POOJA_SLASH_SPOKEN", "स्लैश")


def _brand(match: re.Match) -> str:
    return f"{BRAND_SPOKEN} {DOTCOM_SPOKEN}" if match.group(1) else BRAND_SPOKEN


def speech_text(text: str) -> str:
    """Return `text` with the brand name and URLs spelled for speech."""
    # "10-15" / "3-4" (any dash) -> "10 se 15": a bare hyphen can be dropped or
    # misread as "one thousand fifteen" by the voice.
    out = _RANGE_RE.sub(r"\1 se \2", text)
    out = _BRAND_RE.sub(_brand, out)
    out = _URL_RE.sub("", out)
    # "hostelpgdekho.com/app" style leftovers already handled above;
    # remaining slashes and dots in paths read better as words.
    out = out.replace("/", f" {SLASH_SPOKEN} ")
    return re.sub(r"\s{2,}", " ", out).strip()
