r"""
Prototype v5: Filler injection test — now that baseline (v4) confirms
Gen #2 works in this test harness.

Key difference from baseline: push TTSSpeakFrame BEFORE result_callback.

Run:
  .\venv\Scripts\python.exe test_filler_injection.py
"""

import asyncio
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")

load_dotenv(override=True)

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    TextFrame,
    TTSSpeakFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema


T0 = 0.0
TIMELINE = []


def mark(event: str, detail: str = ""):
    elapsed = (time.time() - T0) * 1000
    TIMELINE.append({"t_ms": round(elapsed, 1), "event": event, "detail": detail})
    logger.info(f"[T+{elapsed:7.1f}ms] {event}  {detail}")


test_tool = FunctionSchema(
    name="confirm_response",
    description="Call this to confirm the user's positive response.",
    properties={
        "confirmed": {
            "type": "string",
            "description": "Set to 'yes' if the user confirmed.",
            "enum": ["yes", "no"],
        },
    },
    required=["confirmed"],
)

TOOLS = ToolsSchema(standard_tools=[test_tool])


async def run_test():
    global T0

    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
            max_tokens=60,
        ),
    )

    tts = SarvamTTSService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice="priya",
            pace=1.12,
        ),
    )

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a test assistant. "
                    "When the user says something positive, call confirm_response with confirmed='yes'. "
                    "After the tool result, respond with a short follow-up question under 10 words. "
                    "Never include function syntax in your text output. "
                    "If the tool result contains 'filler_spoken', that word was already spoken aloud. "
                    "Do NOT repeat it. Continue directly with your question."
                ),
            },
            {"role": "user", "content": "Yeah, that sounds great."},
        ],
        tools=TOOLS,
    )

    test_complete = asyncio.Event()
    context_snapshot_at_gen2 = []
    gen_count = 0
    filler_push_t = 0.0

    # ===== FILLER INJECTION TOOL HANDLER =====
    async def tool_handler(params: FunctionCallParams):
        nonlocal filler_push_t
        mark("TOOL_HANDLER", ">>> Tool handler invoked (WITH filler injection)")
        args = params.arguments or {}

        # Step 1: Push filler BEFORE result_callback
        filler = "Great."
        filler_push_t = time.time()
        mark("FILLER_INJECT", f"Pushing TTSSpeakFrame('{filler}', append_to_context=False)")
        await params.llm.push_frame(
            TTSSpeakFrame(text=filler, append_to_context=False)
        )
        mark("FILLER_INJECT", "TTSSpeakFrame pushed")

        # Step 2: Normal result_callback (triggers Gen #2 via aggregator)
        mark("RESULT_CB", "Calling result_callback (default: run_llm=True)")
        await params.result_callback(
            {"ok": True, "confirmed": args.get("confirmed", "yes"),
             "filler_spoken": "Great."},
        )
        mark("RESULT_CB", "result_callback returned")

    llm.register_function("confirm_response", tool_handler)

    # -- LLM tracing --
    @llm.event_handler("on_before_process_frame")
    async def on_llm_before(llm_svc, data):
        nonlocal gen_count
        frame = data.frame if hasattr(data, "frame") else data
        fname = type(frame).__name__
        if fname in ("LLMContextFrame", "LLMRunFrame"):
            gen_count += 1
            mark("LLM_GEN", f"{fname} -> generation #{gen_count}")
            if gen_count >= 2:
                mark("Q3_CHECK", "=== Gen #2 TRIGGERED! Context snapshot ===")
                for i, msg in enumerate(context.messages):
                    role = msg.get("role", "?")
                    content = str(msg.get("content", ""))[:120]
                    has_filler = "Great" in content or "filler_spoken" in content
                    context_snapshot_at_gen2.append({
                        "idx": i, "role": role, "content": content, "has_filler": has_filler
                    })
                    flag = " <-- FILLER REF" if has_filler else ""
                    mark("Q3_CHECK", f"  [{i}] {role}: {content}{flag}")

    @llm.event_handler("on_after_push_frame")
    async def on_llm_after(llm_svc, data):
        frame = data.frame if hasattr(data, "frame") else data
        fname = type(frame).__name__
        if fname == "LLMFullResponseStartFrame":
            mark("LLM_FRAME", f"LLMFullResponseStartFrame (gen #{gen_count})")
        elif fname == "LLMFullResponseEndFrame":
            mark("LLM_FRAME", f"LLMFullResponseEndFrame (gen #{gen_count})")
            if gen_count >= 2:
                await asyncio.sleep(4.0)
                test_complete.set()
        elif isinstance(frame, TextFrame) and not isinstance(frame, TTSSpeakFrame):
            mark("LLM_TEXT_OUT", f"gen#{gen_count}: {frame.text[:80]!r}")
        elif isinstance(frame, TTSSpeakFrame):
            mark("LLM_TTS_SPEAK", f"TTSSpeakFrame through LLM: {frame.text!r}")

    # -- TTS tracing --
    @tts.event_handler("on_before_process_frame")
    async def on_tts_before(tts_svc, data):
        frame = data.frame if hasattr(data, "frame") else data
        fname = type(frame).__name__
        if isinstance(frame, TTSSpeakFrame):
            mark("TTS_INPUT", f"TTSSpeakFrame: {frame.text!r}")
        elif isinstance(frame, TextFrame):
            mark("TTS_INPUT", f"TextFrame: {frame.text[:60]!r}")
        elif isinstance(frame, FunctionCallResultFrame):
            mark("TTS_INPUT", f"FunctionCallResultFrame")
        elif fname in ("LLMFullResponseStartFrame", "LLMFullResponseEndFrame"):
            mark("TTS_INPUT", fname)

    first_filler_audio = True
    first_gen2_audio = True

    @tts.event_handler("on_after_push_frame")
    async def on_tts_after(tts_svc, data):
        nonlocal first_filler_audio, first_gen2_audio
        frame = data.frame if hasattr(data, "frame") else data
        if isinstance(frame, AudioRawFrame):
            nbytes = len(frame.audio) if frame.audio else 0
            mark("TTS_AUDIO_OUT", f"AudioRawFrame: {nbytes} bytes")

    # -- Aggregator tracing --
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    @assistant_agg.event_handler("on_before_process_frame")
    async def on_agg_before(agg, data):
        frame = data.frame if hasattr(data, "frame") else data
        fname = type(frame).__name__
        if "FunctionCall" in fname:
            mark("AGG_INPUT", f"{fname}")

    pipeline = Pipeline([user_agg, llm, tts, assistant_agg])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
        runner = PipelineRunner(handle_sigint=False)

    async def drive_test():
        await asyncio.sleep(1.5)
        global T0
        T0 = time.time()
        mark("TEST_START", "FILLER INJECTION TEST")
        await task.queue_frames([LLMRunFrame()])

        try:
            await asyncio.wait_for(test_complete.wait(), timeout=25.0)
        except asyncio.TimeoutError:
            mark("TIMEOUT", "Test timed out after 25s")

        mark("TEST_END", "Printing results")

        # Final context
        mark("FINAL_CONTEXT", "Context at test end:")
        for i, msg in enumerate(context.messages):
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:120]
            mark("FINAL_CONTEXT", f"  [{i}] {role}: {content}")

        print_results(context_snapshot_at_gen2)
        await task.cancel()

    asyncio.create_task(drive_test())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        await runner.run(task)


def print_results(context_snapshot):
    print("\n" + "=" * 74)
    print("  FILLER INJECTION TEST RESULTS")
    print("=" * 74)

    print("\n--- TIMELINE ---")
    for e in TIMELINE:
        print(f"  T+{e['t_ms']:8.1f}ms  {e['event']:22s}  {e['detail']}")

    # Q1: Frame ordering
    print("\n--- Q1: FRAME ORDERING AT TTS ---")
    filler_at_tts = [e for e in TIMELINE if e["event"] == "TTS_INPUT" and "TTSSpeakFrame" in e["detail"]]
    gen2_at_tts = [e for e in TIMELINE if e["event"] == "TTS_INPUT" and e["detail"].startswith("TextFrame")]

    if filler_at_tts and gen2_at_tts:
        ft = filler_at_tts[0]["t_ms"]
        gt = gen2_at_tts[0]["t_ms"]
        print(f"  Filler at TTS:     T+{ft:.1f}ms")
        print(f"  Gen #2 at TTS:     T+{gt:.1f}ms")
        print(f"  Delta: {gt-ft:.1f}ms")
        print(f"  [OK] Filler reaches TTS {gt-ft:.0f}ms before Gen #2" if gt > ft else "  [ISSUE]")
    elif filler_at_tts:
        print(f"  Filler at TTS: T+{filler_at_tts[0]['t_ms']:.1f}ms")
        print(f"  Gen #2 at TTS: {'not observed' if not gen2_at_tts else gen2_at_tts[0]['t_ms']}")

    # Q2: Audio
    print("\n--- Q2: AUDIO SEQUENCING ---")
    audio = [e for e in TIMELINE if e["event"] == "TTS_AUDIO_OUT"]
    if audio:
        print(f"  Total audio chunks: {len(audio)}")
        print(f"  First: T+{audio[0]['t_ms']:.1f}ms, Last: T+{audio[-1]['t_ms']:.1f}ms")

    # Q3: Context at Gen #2
    print("\n--- Q3: CONTEXT AT Gen #2 ---")
    if context_snapshot:
        filler_ref = any(s.get("has_filler") for s in context_snapshot)
        for s in context_snapshot:
            flag = " <-- FILLER REF" if s.get("has_filler") else ""
            print(f"    [{s['idx']}] {s['role']}: {s['content'][:80]}{flag}")
        if filler_ref:
            print(f"  [OK] Gen #2 sees filler_spoken in tool result")
        else:
            print(f"  Gen #2 does NOT see filler reference in context")
    else:
        print(f"  Gen #2 did not trigger")

    # Key timing
    print("\n--- KEY MEASUREMENTS ---")
    tool_t = [e for e in TIMELINE if "TOOL_HANDLER" in e["event"] and "invoked" in e["detail"]]
    filler_push = [e for e in TIMELINE if e["event"] == "FILLER_INJECT" and "pushed" in e["detail"]]
    gen2_start = [e for e in TIMELINE if e["event"] == "LLM_GEN" and "#2" in e["detail"]]
    first_audio = [e for e in TIMELINE if e["event"] == "TTS_AUDIO_OUT"]
    gen2_text = [e for e in TIMELINE if e["event"] == "TTS_INPUT" and e["detail"].startswith("TextFrame")]

    if tool_t: print(f"  Tool handler:       T+{tool_t[0]['t_ms']:.1f}ms")
    if filler_push: print(f"  Filler pushed:      T+{filler_push[0]['t_ms']:.1f}ms")
    if gen2_start: print(f"  Gen #2 triggered:   T+{gen2_start[0]['t_ms']:.1f}ms")
    if first_audio: print(f"  First audio:        T+{first_audio[0]['t_ms']:.1f}ms (filler)")
    if gen2_text: print(f"  Gen #2 text at TTS: T+{gen2_text[0]['t_ms']:.1f}ms")

    if filler_push and first_audio:
        print(f"  Filler TTS TTFB:    {first_audio[0]['t_ms'] - filler_push[0]['t_ms']:.0f}ms")

    if gen2_start and first_audio:
        print(f"  Gen2 start->Audio:  {first_audio[0]['t_ms'] - gen2_start[0]['t_ms']:.0f}ms (negative = audio before gen2)")

    print("\n--- VERDICT ---")
    gen2_ok = any("generation #2" in e["detail"] for e in TIMELINE if e["event"] == "LLM_GEN")
    if gen2_ok:
        print("  [OK] Filler injection + Gen #2 both work correctly!")
        if filler_push and gen2_text and first_audio:
            filler_audio_t = first_audio[0]['t_ms']
            gen2_text_t = gen2_text[0]['t_ms']
            if filler_audio_t < gen2_text_t:
                print(f"  [OK] Filler audio ({filler_audio_t:.0f}ms) plays BEFORE Gen #2 text ({gen2_text_t:.0f}ms)")
            else:
                print(f"  [NOTE] Filler audio ({filler_audio_t:.0f}ms) after Gen #2 text ({gen2_text_t:.0f}ms)")
    else:
        print("  [FAIL] Gen #2 did not trigger with filler injection")

    print("=" * 74 + "\n")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    asyncio.run(run_test())
