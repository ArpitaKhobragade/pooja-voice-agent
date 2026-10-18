"""
Local-only FastAPI server backing the browser-based Cartesia voice
audition page (local/audition_web/index.html).

Reuses the exact same pieces as --mode voice and the terminal
--audition-voices tool — no new Cartesia SDK calls are introduced here:
  - local/tts_cartesia.py's CartesiaTTS.synthesize() does the actual
    synthesis (same model_id/language config as live mode).
  - local/audio_io.py's pcm_int16_to_wav_bytes() wraps the result for
    the browser's <audio> element.
  - local/voice_audition.py's VOICE_CANDIDATES/DEFAULT_SAMPLE_TEXT are
    the same 10 voices and the same comparison sample text as the
    terminal tool.
  - local/env_update.py's update_env_value() does the actual .env edit,
    touching only CARTESIA_VOICE_ID.

Security/safety, by design:
  - CARTESIA_API_KEY never leaves the backend process. It is read from
    os.environ only inside CartesiaTTS, and no endpoint here echoes it
    back in any response, error message, or log line.
  - This server binds to 127.0.0.1 only (see run_server()) — not
    reachable from the network, no cloud deployment.
  - /api/save only ever writes the CARTESIA_VOICE_ID key, and only
    after validating the submitted id is one of this page's own
    VOICE_CANDIDATES (never arbitrary attacker-controlled text written
    into .env).
  - Never touches db.py / the live-call SQLite database, never starts
    a call, never imports Twilio.
"""

from __future__ import annotations

import os
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from local import audio_io
from local.env_update import update_env_value
from local.tts_cartesia import CartesiaTTS, CartesiaUnavailable
from local.voice_audition import DEFAULT_SAMPLE_TEXT, VOICE_CANDIDATES

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
WEB_DIR = Path(__file__).resolve().parent / "audition_web"

VOICE_IDS = {v["id"] for v in VOICE_CANDIDATES}

# In-memory, this-process-lifetime cache: {voice_id: wav_bytes}. Avoids
# re-billing Cartesia every time the browser replays a voice, including
# across multiple Play clicks or a page reload within the same server
# run. Cleared automatically when the server process exits — this is
# intentionally NOT written to disk.
_audio_cache: dict[str, bytes] = {}
_cache_lock = threading.Lock()

# Best-effort voice taglines, fetched once at startup if a key is
# present; left blank per-voice if the lookup fails or no key is set.
_taglines: dict[str, str] = {}


def _load_taglines() -> None:
    if not os.environ.get("CARTESIA_API_KEY"):
        return
    try:
        from local.tts_cartesia import list_hindi_female_voices

        for v in list_hindi_female_voices(limit=50):
            _taglines[v["id"]] = v.get("tagline") or ""
    except CartesiaUnavailable:
        pass  # descriptions are a nice-to-have, never block the page on this


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _load_taglines()
    yield


app = FastAPI(title="Pooja — Cartesia Voice Audition (local only)", lifespan=_lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/voices")
def get_voices() -> JSONResponse:
    """Never includes CARTESIA_API_KEY or any other secret — only
    name/id/tagline/sample text, which are safe to show in the browser."""
    voices = [
        {
            "id": v["id"],
            "name": v["name"],
            "tagline": _taglines.get(v["id"], ""),
        }
        for v in VOICE_CANDIDATES
    ]
    return JSONResponse(
        {
            "voices": voices,
            "sample_text": DEFAULT_SAMPLE_TEXT,
            "cartesia_configured": bool(os.environ.get("CARTESIA_API_KEY")),
        }
    )


@app.get("/api/audio/{voice_id}")
def get_audio(voice_id: str):
    if voice_id not in VOICE_IDS:
        raise HTTPException(status_code=404, detail="Unknown voice id")

    with _cache_lock:
        cached = _audio_cache.get(voice_id)
    if cached is not None:
        return Response(content=cached, media_type="audio/wav")

    try:
        tts = CartesiaTTS(voice_id=voice_id)
        pcm, sample_rate = tts.synthesize(DEFAULT_SAMPLE_TEXT)
    except CartesiaUnavailable as e:
        # Message is already key-free (see tts_cartesia.py) — safe to
        # return verbatim to the browser.
        raise HTTPException(status_code=502, detail=str(e)) from e

    wav_bytes = audio_io.pcm_int16_to_wav_bytes(pcm, sample_rate)
    with _cache_lock:
        _audio_cache[voice_id] = wav_bytes
    return Response(content=wav_bytes, media_type="audio/wav")


class SaveVoiceRequest(BaseModel):
    voice_id: str


@app.post("/api/save")
def save_voice(req: SaveVoiceRequest) -> JSONResponse:
    if req.voice_id not in VOICE_IDS:
        raise HTTPException(status_code=400, detail="Unknown voice id — refusing to save it.")

    try:
        update_env_value(ENV_PATH, "CARTESIA_VOICE_ID", req.voice_id)
    except Exception as e:  # noqa: BLE001 — surface any write failure clearly, don't 500 blindly
        raise HTTPException(status_code=500, detail=f"Could not update .env: {e}") from e

    name = next((v["name"] for v in VOICE_CANDIDATES if v["id"] == req.voice_id), req.voice_id)
    return JSONResponse(
        {
            "ok": True,
            "message": (
                f"Saved. CARTESIA_VOICE_ID is now {name}'s id in .env — every other line in "
                f"the file was left untouched. This takes effect the next time you start "
                f"'python pooja_cli.py --mode voice' (restart it if it's already running); "
                f"it does not change anything about the --mode text/offline paths, which "
                f"don't use Cartesia at all."
            ),
        }
    )


def run_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://{host}:{port}"
    print(f"Voice audition page: {url}")
    print("This server only listens on localhost — not reachable from your network.")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
