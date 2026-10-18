"""
Streams Pooja's spoken sentences into a DuplexAudio-style output while
watching for the owner talking over her. Shared by the local microphone
mode (pooja_cli.py) and the browser mode (web_server.py), so interruption
behaves identically in both.
"""

from __future__ import annotations

import queue
import threading
import time

from local.tts_cartesia import CartesiaUnavailable


def play_sentences(
    sentences,
    cartesia,
    duplex,
    *,
    allow_fallback: bool = False,
    on_sentence=None,
    on_first_audio=None,
    on_tts_error=None,
) -> tuple[str, bool]:
    """Speaks each sentence as soon as its audio starts arriving.

    A worker thread pulls sentences (which may still be streaming in from
    the LLM) and streams each one's audio from `cartesia`; this thread
    feeds it to `duplex` ~0.5s ahead of playback and polls for a barge-in.

    Callbacks (all optional):
      on_sentence(text)            a sentence is about to be spoken
      on_first_audio(elapsed_s, tts_first_chunk_s)
      on_tts_error(exc)            a CartesiaUnavailable that is about to be raised

    Returns (text of the sentences started, interrupted). If interrupted,
    the owner's utterance is already captured in duplex.pending_result.
    """
    audio_q: queue.Queue = queue.Queue(maxsize=64)
    stop = threading.Event()
    t_start = time.time()

    def worker():
        try:
            for sentence in sentences:
                if stop.is_set():
                    break
                audio_q.put(("sentence", sentence))
                try:
                    for arr in cartesia.synthesize_stream(sentence):
                        if stop.is_set():
                            break
                        audio_q.put(("pcm", arr))
                except CartesiaUnavailable:
                    if not allow_fallback:
                        raise
                    audio_q.put(("fallback", sentence))
        except BaseException as e:  # noqa: BLE001 - re-raised on the playing thread
            audio_q.put(("error", e))
        finally:
            audio_q.put(("done", None))

    threading.Thread(target=worker, name="reply-synth", daemon=True).start()

    spoken: list[str] = []
    first = True
    done = False
    duplex.start_playback()
    while True:
        if getattr(duplex, "closed", False):
            stop.set()
            return " ".join(spoken), False
        pulled = 0
        # Keep ~0.5s of audio queued ahead of the speaker.
        while not done and duplex.buffered_seconds() < 0.5 and pulled < 8:
            try:
                kind, item = audio_q.get(timeout=0.01 if pulled == 0 else 0.001)
            except queue.Empty:
                break
            pulled += 1
            if kind == "done":
                done = True
            elif kind == "error":
                stop.set()
                duplex.stop_output()
                if isinstance(item, CartesiaUnavailable) and on_tts_error:
                    on_tts_error(item)
                raise item
            elif kind == "sentence":
                if on_sentence:
                    on_sentence(item)
                spoken.append(item)
            elif kind == "pcm":
                if first:
                    first = False
                    if on_first_audio:
                        on_first_audio(time.time() - t_start, getattr(cartesia, "last_synthesis_s", None) or 0.0)
                duplex.enqueue(item)
            elif kind == "fallback":
                from local.tts_cartesia import speak_pyttsx3

                speak_pyttsx3(item)
        if pulled == 0 and not done:
            time.sleep(0.01)
        if duplex.service():
            stop.set()
            try:  # unblock the synth worker so it can exit
                while True:
                    audio_q.get_nowait()
            except queue.Empty:
                pass
            return " ".join(spoken), True
        if done and duplex.drained():
            return " ".join(spoken), False
