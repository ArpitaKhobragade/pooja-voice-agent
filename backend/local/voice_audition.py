"""
Interactive Cartesia voice audition tool for picking Pooja's
CARTESIA_VOICE_ID — reuses the exact same CartesiaTTS wrapper and
audio_io playback code as the live --mode voice pipeline (no new SDK
calls invented here; this just drives CartesiaTTS.synthesize() and
audio_io.play_pcm_int16() per candidate voice).

Deliberately does NOT write to .env and does NOT select a voice for
you — it only plays samples and prints ids so you can choose and copy
one into CARTESIA_VOICE_ID yourself. Never touches the live-call
database (db.py) or makes any outbound call.
"""

from __future__ import annotations

import os

from local import audio_io
from local.tts_cartesia import CartesiaTTS, CartesiaUnavailable

# The owner-facing opening line, in Hindi/Hinglish — a real sample of
# what Pooja actually says, not a generic test phrase.
DEFAULT_SAMPLE_TEXT = (
    "Hello Sir, good afternoon. Main Pooja bol rahi hoon, HostelPGDekho.com se. "
    "Mujhe aapke PG ke regarding baat karni thi. Kya hum baat kar sakte hain?"
)

# Hindi, feminine-presenting voices returned by `--find-voice` against
# the user's own Cartesia account on 2026-10-17. Account-specific ids —
# re-run `--find-voice` yourself if your account's voice library changes;
# this list is just a convenient default for `--audition-voices`.
VOICE_CANDIDATES = [
    {"name": "Siya", "id": "4459a9a5-69d6-4680-b970-e13dc51845b6"},
    {"name": "Sana", "id": "e6b71342-48f1-4c70-a13a-d197b176ff24"},
    {"name": "Anika", "id": "32b0f12b-67c3-421d-8850-b46c019ced91"},
    {"name": "Riva", "id": "a631bc8b-ea1c-49bb-8dab-7a118afd11b8"},
    {"name": "Sneha", "id": "6b02ffe5-e3cb-48c0-a023-c72f85953375"},
    {"name": "Nisha", "id": "0f14d8cb-f039-41fe-a813-a9b4bee7eed8"},
    {"name": "Meera", "id": "a81fccdc-5595-4dfc-ae76-4de6a515b8a2"},
    {"name": "Neha", "id": "47f3bbb1-e98f-4e0c-92c5-5f0325e1e206"},
    {"name": "Kavya", "id": "ce45ddca-5f4b-4047-ada3-3fce2bc78acd"},
    {"name": "Kriti", "id": "5283efe8-07d1-4e3a-b615-2ae4a81c1b73"},
]


def _default_synthesize(voice_id: str, text: str) -> tuple:
    # A fresh CartesiaTTS per call is intentional and cheap: the Cartesia
    # SDK client itself does no network I/O until .synthesize() is
    # called, and this keeps each voice's config (voice_id) isolated
    # instead of mutating shared state across the loop.
    tts = CartesiaTTS(voice_id=voice_id)
    return tts.synthesize(text)


def audition_voices(
    sample_text: str = DEFAULT_SAMPLE_TEXT,
    voices: list[dict] | None = None,
    synthesize_fn=None,
    play_fn=None,
    ask=input,
    announce=print,
) -> list[str]:
    """Plays `sample_text` in each candidate voice, announcing the name
    before playback. After each sample: 'r' replays it, 'n' (or Enter)
    moves to the next voice, 'q' quits immediately without finishing the
    list. A voice that fails to synthesize is reported and skipped
    automatically — one bad id never blocks the rest of the audition.

    Returns the list of voice ids that failed (for a summary at the end).
    Never writes to .env, never sets a "chosen" voice — picking one is
    entirely the caller's decision, made outside this function.
    """
    voices = voices if voices is not None else VOICE_CANDIDATES
    synthesize_fn = synthesize_fn or _default_synthesize
    play_fn = play_fn or (lambda pcm, sr: audio_io.play_pcm_int16(pcm, sr))

    failed: list[str] = []
    for voice in voices:
        announce(f"\n=== {voice['name']}  (id={voice['id']}) ===")
        while True:
            try:
                pcm, sr = synthesize_fn(voice["id"], sample_text)
            except CartesiaUnavailable as e:
                # Never include the API key in this message — CartesiaTTS's
                # own error strings are already key-free (see
                # local/tts_cartesia.py), so this is safe to print as-is.
                announce(f"[Error] Could not synthesize '{voice['name']}': {e}")
                failed.append(voice["id"])
                break

            play_fn(pcm, sr)
            choice = (ask("  [r]eplay / [n]ext voice / [q]uit without selecting> ") or "").strip().lower()
            if choice in ("q", "quit"):
                announce("\nAudition stopped early — no voice was selected.")
                return failed
            if choice in ("r", "replay"):
                continue
            break  # anything else (including blank/Enter) advances to the next voice

    return failed


def run_audition_cli(sample_text: str | None = None) -> None:
    """Entry point used by `pooja_cli.py --audition-voices`."""
    if not os.environ.get("CARTESIA_API_KEY"):
        raise CartesiaUnavailable(
            "CARTESIA_API_KEY is not set — required to audition Cartesia voices."
        )

    text = sample_text or DEFAULT_SAMPLE_TEXT
    print(f"Auditioning {len(VOICE_CANDIDATES)} Hindi, feminine-presenting voices.")
    print(f"Sample text: {text!r}\n")

    failed = audition_voices(sample_text=text)

    print("\n--- Audition finished ---")
    if failed:
        print(f"{len(failed)} voice(s) failed to synthesize and were skipped: {failed}")
    print(
        "No voice was selected automatically and .env was not changed. "
        "Copy whichever id you liked into CARTESIA_VOICE_ID in .env yourself."
    )
