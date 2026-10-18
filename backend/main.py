"""
FastAPI backend.

Endpoints:
  POST /call            -> trigger an outbound call to a phone number
  WS   /media-stream     -> Twilio Media Streams connects here; this is
                            where the Pipecat voice pipeline actually runs
  GET  /leads             -> list of calls + extracted lead data (dashboard)
  GET  /leads/{call_id}   -> full detail incl. transcript (dashboard)

Run with:
  uvicorn main:app --host 0.0.0.0 --port 8080

You also need an ngrok tunnel (or similar) pointing at this port, and
PUBLIC_HOSTNAME in .env set to that tunnel's hostname (no scheme),
e.g. PUBLIC_HOSTNAME=abcd1234.ngrok.io
"""

import json
import os
import sys
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, VoiceResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
from voice.bot import run_bot  # noqa: E402

load_dotenv(override=True)

app = FastAPI(title="Sia AI Voice Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
PUBLIC_HOSTNAME = os.environ.get("PUBLIC_HOSTNAME")  # e.g. abcd1234.ngrok.io, no scheme

twilio_client = (
    TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
)

# call_sid -> {"phone_number": ..., "call_id": ...}
# Populated when we trigger the outbound call, looked up when Twilio's
# Media Stream websocket connects a few seconds later.
PENDING_CALLS: dict[str, dict] = {}


@app.on_event("startup")
async def startup():
    db.init_db()
    logger.info("DB ready")


class CallRequest(BaseModel):
    phone_number: str  # E.164 format, e.g. +919876543210


@app.post("/call")
async def start_call(req: CallRequest):
    if not twilio_client:
        raise HTTPException(500, "Twilio credentials not configured (check .env)")
    if not PUBLIC_HOSTNAME:
        raise HTTPException(500, "PUBLIC_HOSTNAME not set — needed so Twilio can reach your media stream")

    call_id = str(uuid.uuid4())

    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{PUBLIC_HOSTNAME}/media-stream")
    response.append(connect)

    call = twilio_client.calls.create(
        to=req.phone_number,
        from_=TWILIO_PHONE_NUMBER,
        twiml=str(response),
        record=True,  # free bonus: call recording, just a flag away
    )

    PENDING_CALLS[call.sid] = {"phone_number": req.phone_number, "call_id": call_id}
    logger.info(f"Dialing {req.phone_number} — Twilio call_sid={call.sid}, internal call_id={call_id}")

    return {"status": "calling", "call_sid": call.sid, "call_id": call_id}


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Twilio connects here once the callee picks up. We read the initial
    'start' event to get stream_sid/call_sid, look up which phone number
    this call belongs to, then hand off to the Pipecat bot."""
    import time as _time
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

    _ws_t0 = _time.time()
    await websocket.accept()
    logger.info(f"[WS T+0ms] WebSocket accepted")

    # Twilio sends: {{"event": "connected", ...}} then {{"event": "start", "start": {{...}}}}
    started = False
    stream_sid = None
    call_sid = None
    while not started:
        raw = await websocket.receive_text()
        msg = json.loads(raw)
        logger.info(f"[WS T+{(_time.time()-_ws_t0)*1000:.0f}ms] Twilio event: {msg.get('event')}")
        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"]["callSid"]
            started = True

    logger.info(f"[WS T+{(_time.time()-_ws_t0)*1000:.0f}ms] Twilio stream started — creating serializer")

    call_info = PENDING_CALLS.pop(call_sid, None)
    phone_number = call_info["phone_number"] if call_info else "unknown"
    call_id = call_info["call_id"] if call_info else str(uuid.uuid4())

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
    )

    # ── Tuned VAD for Hindi/Hinglish phone conversations ──
    # IMPORTANT: In Pipecat 1.5.0, VAD is configured on the user
    # aggregator (LLMUserAggregatorParams), NOT on the transport.
    # The old code passed vad_enabled/vad_analyzer/vad_audio_passthrough
    # to FastAPIWebsocketParams, but these params don't exist in 1.5.0
    # (verified via inspect) — they were silently ignored.
    #
    # Tuning rationale for Hindi/Hinglish:
    # - stop_secs=0.3  (default 0.2): more generous to avoid cutting off
    #   speakers who pause mid-sentence (common in Hindi)
    # - confidence=0.6 (default 0.7): lower for phone-quality audio
    # - min_volume=0.5 (default 0.6): lower for telephony audio levels
    vad_params = VADParams(
        confidence=0.55,
        start_secs=0.12,
        stop_secs=0.20,
        min_volume=0.45,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )
    logger.info(f"[WS T+{(_time.time()-_ws_t0)*1000:.0f}ms] Transport created — calling run_bot()")

    await run_bot(transport, phone_number, call_id, vad_params=vad_params)


@app.get("/leads")
async def list_leads():
    return db.get_leads()


@app.get("/leads/{call_id}")
async def get_lead(call_id: str):
    call = db.get_call(call_id)
    if not call:
        raise HTTPException(404, "call not found")
    return call


# Static UI — mounted last so it never shadows the API routes above.
_here = os.path.dirname(os.path.abspath(__file__))
app.mount("/dashboard", StaticFiles(directory=os.path.join(_here, "..", "dashboard"), html=True), name="dashboard")
app.mount("/", StaticFiles(directory=os.path.join(_here, "..", "frontend"), html=True), name="frontend")
