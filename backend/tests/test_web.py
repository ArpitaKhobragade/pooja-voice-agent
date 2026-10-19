"""
Tests for the browser version: the WebSocket protocol (web_server.py), the
call runner (local/web_call.py) and the browser-backed audio
(local/web_audio.py). All speech, AI and voice services are fakes, so no
network, keys or audio hardware are needed.
"""

import json
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import web_server  # noqa: E402
from agent_config import SCRIPT_STEPS  # noqa: E402
from local.web_audio import WebDuplex  # noqa: E402
from local.web_call import WebCall, missing_config  # noqa: E402
from tests.test_owner_handling import voice  # noqa: E402


# ----------------------------------------------------------------------------- fakes

class FakeVAD:
    frame_samples = 480
    backend = "fake"

    def is_speech(self, frame):
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) > 500


class FakeCartesia:
    """Streams a 440 Hz tone: `chunks` chunks of 0.1 s, `delay` seconds apart."""

    last_synthesis_s = 0.01

    def __init__(self, chunks=3, delay=0.0):
        self.chunks, self.delay = chunks, delay

    def synthesize_stream(self, text):
        t = np.arange(1600) / 16000
        tone = (np.sin(2 * np.pi * 440 * t) * 2000).astype(np.int16)
        for _ in range(self.chunks):
            if self.delay:
                time.sleep(self.delay)
            yield tone

    def prewarm(self, texts):
        pass


class FakeTTS:
    def __init__(self, **kw):
        self.cartesia = FakeCartesia(**kw)


class FakeLLM:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = 0

    def chat_stream(self, messages):
        self.calls += 1
        yield self.replies.pop(0) if self.replies else "Ji."


class Recorder:
    def __init__(self):
        self.events, self.audio = [], []
        self._lock = threading.Lock()

    def json(self, obj):
        with self._lock:
            self.events.append(obj)

    def bytes(self, data):
        with self._lock:
            self.audio.append(data)

    def types(self):
        with self._lock:
            return [e["type"] for e in self.events]

    def find(self, **match):
        with self._lock:
            return [e for e in self.events if all(e.get(k) == v for k, v in match.items())]

    def wait_for(self, timeout=8.0, **match):
        end = time.time() + timeout
        while time.time() < end:
            hits = self.find(**match)
            if hits:
                return hits[0]
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {match}; saw {self.types()}")


def make_call(rec, *, tts=None, llm=None, stt=None, max_call_s=60.0):
    return WebCall(
        rec.json, rec.bytes,
        max_call_s=max_call_s,
        llm_factory=lambda: llm or FakeLLM(),
        tts_factory=lambda: tts or FakeTTS(),
        vad_factory=FakeVAD,
        stt=stt or (lambda pcm, sr: "haan ji"),
    )


def feed(call, pcm, realtime=False):
    """Push int16 samples into the call as the browser would (512-sample chunks)."""
    for i in range(0, len(pcm), 512):
        call.feed_mic(pcm[i:i + 512].tobytes())
        if realtime:
            time.sleep(0.03)


# ----------------------------------------------------------------------------- web audio

class TestWebDuplex(unittest.TestCase):
    def make(self):
        sent, clears = [], []
        d = WebDuplex(FakeVAD(), send_audio=sent.append, send_clear=lambda: clears.append(1))
        return d, sent, clears

    def test_mic_chunks_of_any_size_are_reframed_for_the_vad(self):
        d, _, _ = self.make()
        d.feed_mic(np.zeros(512, dtype=np.int16).tobytes())   # 512 -> one 480 frame, 32 left over
        d.feed_mic(np.zeros(448, dtype=np.int16).tobytes())   # 32 + 448 = 480 -> second frame
        self.assertEqual(d._mic_q.qsize(), 2)
        self.assertEqual(len(d._mic_q.get()), 480)

    def test_odd_byte_counts_do_not_crash(self):
        d, _, _ = self.make()
        d.feed_mic(b"\x00" * 961)  # 480 whole samples plus one stray byte
        self.assertEqual(d._mic_q.qsize(), 1)

    def test_audio_is_sent_and_the_playback_clock_advances(self):
        d, sent, _ = self.make()
        d.enqueue(np.zeros(8000, dtype=np.int16))            # 0.5 s
        self.assertEqual(len(sent), 1)
        self.assertEqual(len(sent[0]), 16000)                # int16 bytes
        self.assertAlmostEqual(d.buffered_seconds(), 0.5, delta=0.05)
        self.assertFalse(d.drained())

    def test_stop_output_clears_the_browser_and_the_clock(self):
        d, _, clears = self.make()
        d.enqueue(np.zeros(8000, dtype=np.int16))
        d.stop_output()
        self.assertEqual(clears, [1])
        self.assertEqual(d.buffered_seconds(), 0.0)

    def test_closed_duplex_stops_listening_immediately(self):
        d, _, _ = self.make()
        d.close()
        self.assertIsNone(d.listen())

    def test_echo_reference_follows_what_is_playing(self):
        d, _, _ = self.make()
        self.assertEqual(d._output_reference(), 0.0)
        d.enqueue((np.ones(16000) * 3000).astype(np.int16))
        self.assertGreater(d._output_reference(), 2500)


class TestGatedVAD(unittest.TestCase):
    class AlwaysSpeech:
        frame_samples = 480
        backend = "inner"

        def is_speech(self, frame):
            return True

    def test_quiet_frames_are_never_speech_even_if_the_inner_detector_says_so(self):
        from local.web_audio import GatedVAD

        vad = GatedVAD(self.AlwaysSpeech(), min_rms=250)
        self.assertFalse(vad.is_speech((np.random.default_rng(0).standard_normal(480) * 40).astype(np.int16)))
        self.assertTrue(vad.is_speech((np.random.default_rng(0).standard_normal(480) * 2000).astype(np.int16)))

    def test_it_keeps_the_inner_frame_size_and_names_itself(self):
        from local.web_audio import GatedVAD

        vad = GatedVAD(self.AlwaysSpeech(), min_rms=250)
        self.assertEqual(vad.frame_samples, 480)
        self.assertTrue(vad.backend.endswith("+gate"))


# ----------------------------------------------------------------------------- call runner

class TestWebCall(unittest.TestCase):
    def test_missing_settings_are_reported_by_name_only(self):
        rec = Recorder()
        old = {k: os.environ.pop(k, None) for k in ("GROQ_API_KEY", "SARVAM_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID")}
        try:
            call = WebCall(rec.json, rec.bytes)
            call.start()
            call.join(5)
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v
        err = rec.find(type="error")[0]["message"]
        self.assertIn("GROQ_API_KEY", err)
        self.assertEqual(rec.find(type="ended")[0]["reason"], "error")
        self.assertTrue(missing_config() == [] or isinstance(missing_config(), list))

    def test_call_opens_with_pooja_speaking_and_a_transcript_line(self):
        rec = Recorder()
        call = make_call(rec)
        call.start()
        rec.wait_for(type="ready")
        line = rec.wait_for(type="transcript", speaker="pooja")
        self.assertIn("My name is Pooja", line["text"])
        rec.wait_for(type="status", state="speaking")
        rec.wait_for(type="status", state="listening")
        self.assertGreater(len(rec.audio), 0)
        call.stop("stopped")
        call.join(5)
        ended = rec.find(type="ended")[0]
        self.assertEqual(ended["reason"], "stopped")

    def test_owner_speech_is_transcribed_and_pooja_follows_the_script(self):
        rec = Recorder()
        llm = FakeLLM()
        call = make_call(rec, llm=llm)
        call.start()
        rec.wait_for(type="status", state="listening")
        feed(call, voice(200, 1.0))                                  # the owner speaks...
        feed(call, np.zeros(16000 * 2, dtype=np.int16))              # ...then pauses
        owner = rec.wait_for(type="transcript", speaker="owner")
        self.assertEqual(owner["text"], "haan ji")
        rec.wait_for(type="transcript", speaker="pooja", turn=2)
        time.sleep(0.6)                                              # let the step's remaining sentences arrive
        said = " ".join(e["text"] for e in rec.find(type="transcript", speaker="pooja", turn=2))
        # the 200 Hz test voice reads as a woman, so the step is worded for "Ma'am"
        self.assertIn(SCRIPT_STEPS[0]["text"].replace("Sir", "Ma'am"), said)
        self.assertEqual(llm.calls, 0)                               # plain "yes" = the script, no AI call
        call.stop("stopped")
        call.join(5)

    def test_owner_can_interrupt_pooja(self):
        rec = Recorder()
        call = make_call(rec, tts=FakeTTS(chunks=40, delay=0.1))     # a 4 s sentence
        call.start()
        rec.wait_for(type="status", state="speaking")
        for _ in range(45):                                          # ~1.3 s of quiet mic while she speaks
            call.feed_mic(np.zeros(512, dtype=np.int16).tobytes())
            time.sleep(0.03)
        for _ in range(60):                                          # then the owner talks over her
            call.feed_mic(voice(200, 0.032)[:512].tobytes())
            time.sleep(0.03)
        rec.wait_for(type="clear")
        rec.wait_for(type="interrupted", turn=1)
        feed(call, np.zeros(16000 * 2, dtype=np.int16))
        rec.wait_for(type="transcript", speaker="owner")
        call.stop("stopped")
        call.join(5)

    def test_stop_before_the_call_is_ready_still_ends_cleanly(self):
        rec = Recorder()
        call = make_call(rec)
        call.stop("stopped")
        call.start()
        call.join(5)
        self.assertEqual(rec.find(type="ended")[0]["reason"], "stopped")

    def test_time_limit_ends_the_call(self):
        rec = Recorder()
        call = make_call(rec, max_call_s=0.0)
        call.start()
        ended = rec.wait_for(type="ended", timeout=10)
        self.assertEqual(ended["reason"], "time_limit")

    def test_voice_service_failure_is_reported_in_plain_words(self):
        from local.tts_cartesia import CartesiaUnavailable

        class Broken(FakeCartesia):
            def synthesize_stream(self, text):
                raise CartesiaUnavailable("Cartesia credits are used up (HTTP 402)")
                yield

        rec = Recorder()
        call = make_call(rec, tts=SimpleNamespace(cartesia=Broken()))
        call.start()
        err = rec.wait_for(type="error")
        self.assertIn("credits", err["message"])
        rec.wait_for(type="ended")


# ----------------------------------------------------------------------------- server protocol

class FakeWebCall:
    instances = []

    def __init__(self, emit_json, emit_bytes, max_call_s=0, on_finished=None, timezone=None):
        self.emit_json, self.emit_bytes, self.on_finished = emit_json, emit_bytes, on_finished
        self.mic, self.stopped = [], None
        FakeWebCall.instances.append(self)

    def start(self):
        self.emit_json({"type": "ready"})

    def feed_mic(self, data):
        self.mic.append(data)

    def stop(self, reason="stopped"):
        if self.stopped is None:
            self.stopped = reason
            self.emit_json({"type": "ended", "reason": reason, "outcome": "none", "seconds": 1})
            if self.on_finished:
                self.on_finished()


class TestServer(unittest.TestCase):
    def setUp(self):
        FakeWebCall.instances = []
        self._orig = (web_server.WebCall, web_server.ACCESS_CODE, web_server.MAX_CALLS, web_server._active_calls)
        web_server.WebCall = FakeWebCall
        web_server.ACCESS_CODE = ""
        web_server.MAX_CALLS = 2
        web_server._active_calls = 0
        self.client = TestClient(web_server.app)

    def tearDown(self):
        web_server.WebCall, web_server.ACCESS_CODE, web_server.MAX_CALLS, web_server._active_calls = self._orig

    def test_page_and_health_and_config(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Talk to Pooja", page.text)
        self.assertEqual(self.client.get("/healthz").json(), {"ok": True})
        cfg = self.client.get("/api/config").json()
        self.assertIn("requires_code", cfg)
        self.assertNotIn("sk-", json.dumps(cfg))  # never any key material, only setting names

    def test_start_mic_and_stop_flow(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            self.assertEqual(ws.receive_json()["type"], "ready")
            ws.send_bytes(b"\x01\x00" * 512)
            ws.send_text(json.dumps({"type": "stop"}))
            ended = ws.receive_json()
            self.assertEqual(ended["type"], "ended")
        self.assertEqual(len(FakeWebCall.instances), 1)
        self.assertEqual(FakeWebCall.instances[0].mic, [b"\x01\x00" * 512])
        self.assertEqual(FakeWebCall.instances[0].stopped, "stopped")

    def test_disconnect_stops_the_call_and_frees_the_slot(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.receive_json()
        deadline = time.time() + 3
        while time.time() < deadline and web_server._active_calls:
            time.sleep(0.02)
        self.assertEqual(FakeWebCall.instances[0].stopped, "disconnected")
        self.assertEqual(web_server._active_calls, 0)

    def test_access_code_is_required_when_set(self):
        web_server.ACCESS_CODE = "sesame"
        self.assertTrue(self.client.get("/api/config").json()["requires_code"])
        with self.client.websocket_connect("/ws") as ws:
            self.assertEqual(ws.receive_json()["type"], "error")
        with self.client.websocket_connect("/ws?code=wrong") as ws:
            self.assertEqual(ws.receive_json()["type"], "error")
        with self.client.websocket_connect("/ws?code=sesame") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            self.assertEqual(ws.receive_json()["type"], "ready")

    def test_call_limit_protects_the_account(self):
        web_server.MAX_CALLS = 1
        with self.client.websocket_connect("/ws") as first:
            first.send_text(json.dumps({"type": "start"}))
            self.assertEqual(first.receive_json()["type"], "ready")
            with self.client.websocket_connect("/ws") as second:
                second.send_text(json.dumps({"type": "start"}))
                err = second.receive_json()
                self.assertEqual(err["type"], "error")
                self.assertIn("other calls", err["message"])

    def test_garbage_messages_are_ignored(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.send_text("not json")
            ws.send_text(json.dumps({"type": "nonsense"}))
            ws.send_text(json.dumps({"type": "start"}))
            self.assertEqual(ws.receive_json()["type"], "ready")


if __name__ == "__main__":
    unittest.main()
