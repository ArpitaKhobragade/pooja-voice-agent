"""
Helpers for the streamed, conversational reply path.

- stream_sentences(): turns raw LLM text chunks into whole sentences as soon
  as each one is complete (so TTS can start on sentence 1 while the model is
  still writing sentence 2), and pulls out control markers like
  "[INTERESTED]" so they are never spoken.
- reply_effects(): the call-state changes implied by what Pooja just said
  (which pitch points she covered, whether she read out the app link) and by
  any control markers.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

_TERMINATORS = ".?!।"  # includes the Devanagari danda
_MIN_SENTENCE_CHARS = 8

# Control markers the model may append to a reply; never spoken aloud.
END_MARKERS = {
    "INTERESTED": "yes",
    "NOT_INTERESTED": "no",
    "CALLBACK": "maybe",
}


_NUMBERING = re.compile(r"\b(pehli|pahli|dusri|doosri|teesri|tisri)\s+baat(\s+(yeh|ye))?(\s+hai)?(\s+ki)?[,:]?\s*", re.IGNORECASE)


# Time-of-day wishes ("aapka din shubh ho", "shubh ratri"): Pooja can't know
# the owner's time of day, so these are replaced with a neutral wish.
_TIME_WISH = re.compile(
    r"\b(aap(ka|ki|ko)\s+)?(din|shaam|subah|raat|sham)\s+(bahut\s+)?(shubh|achha|accha|mangalmay)\s+ho(ga)?\b"
    r"|\b(aap(ko)?\s+)?shubh\s+(ratri|sandhya|prabhat)\b|\bhave\s+a\s+(good|great|nice)\s+(day|evening|night)\b|"
    r"\bgood\s+(day|evening)\b",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Strips markdown, 'dusri baat ye hai ki'-style numbering and time-of-day
    wishes, which the model sometimes produces despite instructions."""
    text = re.sub(r"[*_#`]+", "", text)
    text = _NUMBERING.sub("", text)
    text = _TIME_WISH.sub("take care", text)
    return re.sub(r"\s{2,}", " ", text)


_CLAUSE_BREAKS = ",;:"
_FIRST_CLAUSE_MIN_CHARS = 24


def stream_sentences(deltas: Iterable[str], tags: list[str], first_clause: bool = True) -> Iterator[str]:
    """Yields complete sentences from streamed text chunks. Anything inside
    square brackets is a control marker: it is appended (upper-cased) to
    `tags` and never yielded. To start speaking sooner, the FIRST segment
    of a reply may end at a comma/semicolon once it is long enough to sound
    natural on its own."""
    buf = ""
    tag_buf = ""
    in_tag = False
    yielded_any = False

    for delta in deltas:
        for ch in delta:
            if in_tag:
                if ch == "]":
                    tags.append(tag_buf.strip().upper())
                    tag_buf, in_tag = "", False
                else:
                    tag_buf += ch
                continue
            if ch == "[":
                in_tag = True
                continue
            # A terminator ends a sentence once whitespace follows it (so
            # "hostelpgdekho.com" and "3.5" are not split), or once a capital
            # letter follows it directly ("...sakti hoon.Ji Sir"), which the
            # model sometimes emits without a space.
            ends_sentence = bool(buf) and buf[-1] in _TERMINATORS and len(buf.strip()) >= _MIN_SENTENCE_CHARS
            ends_first_clause = (
                first_clause
                and not yielded_any
                and bool(buf)
                and buf[-1] in _CLAUSE_BREAKS
                and len(buf.strip()) >= _FIRST_CLAUSE_MIN_CHARS
            )
            if (ends_sentence or ends_first_clause) and (ch.isspace() or (ends_sentence and ch.isupper())):
                sentence = _clean(buf).strip()
                if sentence:
                    yield sentence
                    yielded_any = True
                buf = "" if ch.isspace() else ch
                continue
            buf += ch

    tail = _clean(buf).strip()
    if tail:
        yield tail


_SMART = re.compile(r"smart\s*search", re.IGNORECASE)
_VACANT = re.compile(r"request\s*form|vacan|khali", re.IGNORECASE)
_FREE = re.compile(r"free\s*(listing|mein|hai|me)|listing.{0,25}free|bilkul\s*free", re.IGNORECASE)


def reply_effects(text: str, app_url: str) -> dict:
    """What Pooja's reply implies about the call. Returns
    {"slots": [...benefit slots covered...], "link_read": bool}."""
    slots = []
    if _SMART.search(text):
        slots.append("benefit_smart_search_explained")
    if _VACANT.search(text):
        slots.append("benefit_vacant_form_explained")
    if _FREE.search(text):
        slots.append("benefit_free_listing_explained")
    host = re.sub(r"^https?://(www\.)?", "", app_url).lower()
    link_read = bool(host) and host in text.lower().replace(" ", "")
    return {"slots": slots, "link_read": link_read}
