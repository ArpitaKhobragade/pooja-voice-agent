"""
Pooja in the browser: a small web server that serves the call page and
bridges each visitor's microphone and speaker to the conversation engine
over one WebSocket.

Run locally:      python web_server.py          (opens on http://localhost:8000)
Deploy:           see Dockerfile / render.yaml / README ("Share it with Sir")

Settings (environment variables or .env):
  GROQ_API_KEY, SARVAM_API_KEY, CARTESIA_API_KEY, CARTESIA_VOICE_ID   required
  POOJA_ACCESS_CODE      if set, visitors must enter it to start a call (recommended when public)
  POOJA_MAX_CALLS        simultaneous calls allowed (default 2; protects API limits and credits)
  POOJA_WEB_MAX_CALL_S   longest a call may run, in seconds (default 420)
  PORT                   port to listen on (default 8000)
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from loguru import logger  # noqa: E402

from local.web_call import WebCall, missing_config  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent / "web"
ACCESS_CODE = os.environ.get("POOJA_ACCESS_CODE", "").strip()
MAX_CALLS = int(os.environ.get("POOJA_MAX_CALLS", "2"))
MAX_CALL_S = float(os.environ.get("POOJA_WEB_MAX_CALL_S", "420"))

app = FastAPI(title="Pooja, HostelPGDekho voice agent")

_active_calls = 0
_active_lock = threading.Lock()


def _acquire_slot() -> bool:
    global _active_calls
    with _active_lock:
        if _active_calls >= MAX_CALLS:
            return False
        _active_calls += 1
        return True


def _release_slot() -> None:
    global _active_calls
    with _active_lock:
        _active_calls = max(0, _active_calls - 1)


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/api/config")
async def config():
    return JSONResponse({
        "requires_code": bool(ACCESS_CODE),
        "max_call_seconds": MAX_CALL_S,
        "ready": not missing_config(),
        "missing": missing_config(),  # setting names only, never values
    })


@app.websocket("/ws")
async def call_socket(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    out_q: asyncio.Queue = asyncio.Queue()

    # Called from the call's worker thread; the single sender task below keeps message order.
    def emit_json(obj: dict) -> None:
        loop.call_soon_threadsafe(out_q.put_nowait, ("json", obj))

    def emit_bytes(data: bytes) -> None:
        loop.call_soon_threadsafe(out_q.put_nowait, ("bytes", data))

    async def sender() -> None:
        while True:
            kind, item = await out_q.get()
            if kind == "close":
                return
            try:
                if kind == "json":
                    await ws.send_json(item)
                else:
                    await ws.send_bytes(item)
            except Exception:  # the browser went away
                return

    sender_task = asyncio.create_task(sender())
    call: WebCall | None = None

    try:
        if ACCESS_CODE and not secrets.compare_digest(ws.query_params.get("code", ""), ACCESS_CODE):
            emit_json({"type": "error", "message": "That access code is not right."})
            return

        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is not None:
                if call is not None:
                    call.feed_mic(data)
                continue
            text = message.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
                command = payload.get("type")
            except (ValueError, AttributeError):
                continue
            if command == "start" and call is None:
                if not _acquire_slot():
                    emit_json({"type": "error", "message": "Pooja is on other calls right now. Try again in a minute."})
                    continue
                tz = payload.get("tz")
                call = WebCall(emit_json, emit_bytes, max_call_s=MAX_CALL_S, on_finished=_release_slot,
                               timezone=tz if isinstance(tz, str) and len(tz) < 64 else None)
                call.start()
            elif command == "stop" and call is not None:
                call.stop("stopped")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Call socket closed: {e}")
    finally:
        if call is not None:
            call.stop("disconnected")
        # Queued the same way as the call's messages, so it can never overtake one of them.
        loop.call_soon_threadsafe(out_q.put_nowait, ("close", None))
        await sender_task
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    print(f"Pooja web demo: http://localhost:{port}"
          + ("  (access code required)" if ACCESS_CODE else "  (no access code set)"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
