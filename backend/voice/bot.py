"""
The live voice pipeline for one call.

Pipeline shape:
    transport.input() -> stt -> user_aggregator -> llm -> tts -> transport.output() -> assistant_aggregator

Architecture decisions (post-refactor):
──────────────────────────────────────
1. SYSTEM INSTRUCTION UPDATED IN-PLACE
   The system prompt (context.messages[0]) is rebuilt after every tool call
   with the current state and objective.  This replaces the old approach of
   *appending* developer messages, which caused unbounded context growth
   (~100 tokens per tool call, never cleaned up).

2. TWO LLM GENERATIONS PER TOOL CALL IS EXPECTED
   When a user speaks, the pipeline is:
     User → LLM Gen #1 (produces tool call) → tool handler → result_callback
     → LLM Gen #2 (produces spoken response using tool result) → TTS
   Both generations are NECESSARY — Gen #1 extracts intent, Gen #2 produces
   the reply.  This is the standard function-calling pattern.  The old code
   did NOT have an *unnecessary* third generation; however, the developer
   messages it injected bloated every subsequent generation's prompt.

3. AUTO-SUMMARIZATION AS SAFETY NET
   Pipecat's built-in context summarization is enabled for unusually long
   conversations.  For the typical 4–8 turn sales call it never triggers
   (zero overhead).  This was chosen over a custom rolling window because:
   - It's a one-line change (minimal implementation risk)
   - It handles tool-call/result integrity automatically
   - For short calls there is literally no cost
   - A rolling window would require custom context management code with
     tricky edge cases around tool call/result pair splitting

4. SINGLE VAD CONFIGURATION
   The old code created two separate SileroVADAnalyzer instances (one in
   main.py for the transport, one here for the user aggregator) with
   default params.  Now main.py passes configured VAD params so both
   instances use the same tuned settings.

5. COMPREHENSIVE INSTRUMENTATION
   Every stage is instrumented via CallPerformanceLogger:
   - LLM generation count + TTFB (from Pipecat metrics)
   - Tool execution latency (timed directly)
   - Token usage (from Pipecat usage metrics)
   - STT / TTS latency (from Pipecat metrics)
   Structured JSON logs are saved to backend/logs/ per call.
"""

import asyncio
import os
import sys
import time

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMRunFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame, EndFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.base_transport import BaseTransport

# Optional: auto-summarization support (may not be available in all Pipecat versions)
try:
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMAssistantAggregatorParams,
    )
    _AUTO_SUMMARIZE_AVAILABLE = True
except ImportError:
    _AUTO_SUMMARIZE_AVAILABLE = False
    logger.warning(
        "LLMAssistantAggregatorParams not available in this Pipecat version — "
        "auto-summarization disabled (context may grow without bound in very long calls)"
    )

# UserTurnStrategies: container with .start and .stop strategy lists.
# In Pipecat 1.5.0, the DEFAULT already includes Smart Turn v3:
#   start: [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]
#   stop:  [TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)]
# So we just need to construct UserTurnStrategies() — no manual setup needed.
try:
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    _USER_TURN_STRATEGIES_AVAILABLE = True
except ImportError:
    _USER_TURN_STRATEGIES_AVAILABLE = False
    logger.warning(
        "UserTurnStrategies not available — falling back to VAD-only endpointing"
    )

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/
from conversation_manager import ConversationManager  # noqa: E402
from agent_config import AGENT_CONFIG  # noqa: E402
from db import save_call  # noqa: E402
from extraction import extract_lead  # noqa: E402
from db import save_lead  # noqa: E402
from objective import determine_next_objective  # noqa: E402
from perf_logger import CallPerformanceLogger, PerformanceMetricsObserver  # noqa: E402
from latency_observer import LatencyBreakdownObserver  # noqa: E402
from pre_llm_extractor import PreLLMStateExtractor  # noqa: E402
from prompts import build_dynamic_block, BASE_PERSONA_PROMPT  # noqa: E402
from tools import SIA_TOOLS  # noqa: E402


class StartupUserMuteStrategy(BaseUserMuteStrategy):
    """Mutes the user until the bot starts speaking for the first time.
    Prevents connection/line noise from prematurely triggering STT and cancelling the initial greeting.
    """
    def __init__(self):
        super().__init__()
        self._bot_has_spoken = False

    async def process_frame(self, frame: Frame) -> bool:
        await super().process_frame(frame)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_has_spoken = True

        return not self._bot_has_spoken


# ────────────────────────────────────────────────────────────────────── #
# Tool handler factory                                                   #
# ────────────────────────────────────────────────────────────────────── #

def _make_tool_handler(
    tool_name: str,
    cm: ConversationManager,
    context: LLMContext,
    perf: CallPerformanceLogger,
):
    """Build an async handler for one Pipecat tool.

    After applying the tool call via ConversationManager, the system
    instruction (context.messages[0]) is updated IN-PLACE with the
    current state and next objective.

    OLD approach (removed):
        context.add_message({"role": "developer", "content": ...})
        ↳ accumulated ~100 tokens of stale state per tool call, never cleaned

    NEW approach:
        context.messages[0]["content"] = build_system_instruction(state, obj)
        ↳ system prompt always reflects the CURRENT state, no accumulation
    """

    async def handler(params: FunctionCallParams):
        # ── Audit logging for generation verification ──
        logger.info(
            f"[GENERATION AUDIT] Tool '{tool_name}' invoked — "
            f"LLM generation produced a tool call (this is generation "
            f"#{perf.generation_count} in this call)"
        )

        # ── Execute tool with timing ──
        t0 = time.time()
        args = params.arguments or {}
        result = cm.apply_tool_call(tool_name, args)
        tool_latency_ms = (time.time() - t0) * 1000
        perf.mark_tool_execution(tool_name, tool_latency_ms)
        
        if result.get("redundant"):
            perf.redundant_tool_calls += 1

        logger.info(
            f"[GENERATION AUDIT] Tool '{tool_name}' result: {result}.  "
            f"Calling result_callback → Pipecat will trigger the next "
            f"LLM generation (post-tool response)."
        )

        # ── Update dynamic context with fresh state ──
        # This is the ONLY place where the system prompt changes during
        # a call.  It replaces (not appends) the second system message,
        # so context never grows from stale objective messages.
        objective = determine_next_objective(cm.get_state())
        context.messages[1]["content"] = build_dynamic_block(
            cm.get_state(), objective
        )

        # ── Return tool result to Pipecat → triggers next LLM generation ──
        await params.result_callback(result)

    return handler


# ────────────────────────────────────────────────────────────────────── #
# Main bot entry point                                                   #
# ────────────────────────────────────────────────────────────────────── #

async def run_bot(
    transport: BaseTransport,
    phone_number: str,
    call_id: str,
    vad_params=None,
):
    """Start the voice pipeline for a single call.

    Args:
        transport:    Pipecat transport (WebSocket, connected to Twilio)
        phone_number: Caller's phone number (for ConversationManager)
        call_id:      Unique ID for this call (for logging / DB)
        vad_params:   Optional VADParams instance shared with the transport's
                      VAD analyzer, ensuring consistent endpointing behaviour.
    """
    _t0 = time.time()  # Absolute reference for all startup timestamps
    logger.info(f"[STARTUP T+0ms] Starting Sia bot for call {call_id} to {phone_number}")

    cm = ConversationManager(phone_number=phone_number)
    cm.state.call_id = call_id
    perf = CallPerformanceLogger(call_id)
    logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] ConversationManager + PerfLogger created")

    # ── Services ──

    _ts = time.time()
    stt = SarvamSTTService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamSTTService.Settings(model="saaras:v3"),
    )
    logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] SarvamSTT created ({(time.time()-_ts)*1000:.0f}ms)")

    _ts = time.time()
    tts_mode_str = AGENT_CONFIG.get("tts_aggregation_mode", "sentence").lower()
    tts_agg_mode = TextAggregationMode.TOKEN if tts_mode_str == "token" else TextAggregationMode.SENTENCE
    
    tts = SarvamTTSService(
        api_key=os.environ["SARVAM_API_KEY"],
        text_aggregation_mode=tts_agg_mode,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice="priya",
            pace=1.12,
        ),
    )
    logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] SarvamTTS created ({(time.time()-_ts)*1000:.0f}ms)")

    model_name = os.environ.get(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    _ts = time.time()
    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model=model_name,
            temperature=0.2,
            max_tokens=80,
        ),

    )
    logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] GroqLLM created ({(time.time()-_ts)*1000:.0f}ms) — model={model_name}")

    # ── Context (split into static persona and dynamic state) ──

    initial_objective = determine_next_objective(cm.get_state())

    context = LLMContext(
        messages=[
            {"role": "system", "content": BASE_PERSONA_PROMPT},
            {"role": "system", "content": build_dynamic_block(cm.get_state(), initial_objective)}
        ],
        tools=SIA_TOOLS,
    )
    logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] Context + tools registered")

    # ── Register tool handlers ──

    for tool_name in (
        "update_state",
    ):
        llm.register_function(
            tool_name,
            _make_tool_handler(tool_name, cm, context, perf),
        )

    # ── Aggregators ──
    # VAD with tuned params (passed from main.py).  In Pipecat 1.5.0,
    # the ONLY place VAD is configured is on LLMUserAggregatorParams.
    # The old code also passed vad_analyzer to FastAPIWebsocketParams,
    # but that param doesn't exist in 1.5.0 and was silently ignored.
    _ts = time.time()
    vad = (
        SileroVADAnalyzer(params=vad_params) if vad_params
        else SileroVADAnalyzer()
    )
    logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] SileroVAD created ({(time.time()-_ts)*1000:.0f}ms)")

    # Build user aggregator params with VAD + Smart Turn strategies.
    #
    # CRITICAL: user_turn_strategies expects a UserTurnStrategies dataclass
    # (which has .start and .stop list attributes), NOT a plain list.
    # Passing a list causes:
    #   'list' object has no attribute 'start'
    # in pipecat.turns.user_turn_controller.py
    #
    # In Pipecat 1.5.0, UserTurnStrategies() with no args defaults to:
    #   start: [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]
    #   stop:  [TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)]
    # This gives us Smart Turn v3 ML-based endpointing out of the box.
    user_agg_kwargs: dict = {
        "vad_analyzer": vad,
        "user_mute_strategies": [StartupUserMuteStrategy()],
    }
    if _USER_TURN_STRATEGIES_AVAILABLE:
        _ts = time.time()
        # Configure local ONNX model with 4 CPU threads (instead of default 1) to reduce endpoint latency
        analyzer = LocalSmartTurnAnalyzerV3(cpu_count=4)
        stop_strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)
        user_agg_kwargs["user_turn_strategies"] = UserTurnStrategies(stop=[stop_strategy])
        logger.info(
            f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] UserTurnStrategies created "
            f"({(time.time()-_ts)*1000:.0f}ms) — Smart Turn v3 ONNX loaded (cpu_count=4)"
        )

    aggregator_kwargs: dict = {
        "user_params": LLMUserAggregatorParams(**user_agg_kwargs),
    }

    # Enable auto-summarization as a safety net for unusually long calls.
    # For the typical 4–8 turn sales call this never triggers (zero cost).
    if _AUTO_SUMMARIZE_AVAILABLE:
        aggregator_kwargs["assistant_params"] = LLMAssistantAggregatorParams(
            enable_auto_context_summarization=True,
        )
        logger.info(
            "Auto context summarization enabled "
            "(safety net for long conversations)"
        )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, **aggregator_kwargs
    )

    pre_llm_extractor = PreLLMStateExtractor(cm=cm, perf=perf)

    from pipecat.frames.frames import InterruptionFrame

    class CallLifecycleProcessor(FrameProcessor):
        def __init__(self, cm: ConversationManager, **kwargs):
            super().__init__(**kwargs)
            self.cm = cm

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            
            if isinstance(frame, InterruptionFrame):
                logger.info("[CallLifecycleProcessor] User interrupted the bot.")
                self.cm.state.was_interrupted_last_turn = True

            elif isinstance(frame, BotStartedSpeakingFrame):
                # Once the bot starts speaking its next thought, clear the interrupted flag
                if self.cm.state.was_interrupted_last_turn:
                    logger.debug("[CallLifecycleProcessor] Bot started speaking, clearing interrupted flag.")
                    self.cm.state.was_interrupted_last_turn = False
                    
            elif isinstance(frame, BotStoppedSpeakingFrame):
                if self.cm.state.call_ended:
                    logger.info("[CallLifecycleProcessor] Call is ended deterministically. Injecting EndFrame.")
                    await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    call_lifecycle = CallLifecycleProcessor(cm=cm)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            pre_llm_extractor,
            llm,
            tts,
            call_lifecycle,
            transport.output(),
            assistant_aggregator,
        ]
    )
    logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] Pipeline created")

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] PipelineTask created")

    # Attach performance metrics observers
    task.add_observer(PerformanceMetricsObserver(perf))
    task.add_observer(LatencyBreakdownObserver(call_id))

    # ── Transport event handlers ──

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] on_client_connected fired")

        await task.queue_frames([
            LLMRunFrame(),
        ])
        logger.info(f"[STARTUP T+{(time.time()-_t0)*1000:.0f}ms] LLMRunFrame queued")

    # ── Turn-lifecycle diagnostics ──
    # These event handlers trace the full interruption/turn lifecycle.
    # They are pure observers — they log but never alter pipeline state.

    @user_aggregator.event_handler("on_user_turn_started")
    async def _diag_turn_started(agg, strategy):
        logger.info(
            f"[TURN] >>> User turn STARTED (strategy={type(strategy).__name__})"
        )

    @user_aggregator.event_handler("on_user_turn_inference_triggered")
    async def _diag_inference_triggered(agg, strategy):
        logger.info(
            f"[TURN] === Inference TRIGGERED (strategy={type(strategy).__name__}). "
            f"Context will be pushed to LLM."
        )

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def _diag_turn_stopped(agg, strategy, message):
        content = message.content if message else "<none>"
        # Truncate to 200 chars for readability
        if content and len(content) > 200:
            content = content[:200] + "..."
        logger.info(
            f"[TURN] <<< User turn STOPPED "
            f"(strategy={type(strategy).__name__ if strategy else 'None'}, "
            f"content={content!r})"
        )

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def _diag_msg_added(agg, message):
        content = message.content if message else "<none>"
        if content and len(content) > 200:
            content = content[:200] + "..."
        logger.info(
            f"[TURN] +++ User message ADDED to context: {content!r}"
        )

    @user_aggregator.event_handler("on_user_turn_idle")
    async def _diag_idle(agg):
        logger.info("[TURN] ~~~ User turn IDLE (no speech for idle timeout)")

    # LLM lifecycle — track when context arrives and generation completes
    # Uses sync frame hooks since on_completion_started/finished don't exist.
    from pipecat.frames.frames import LLMFullResponseStartFrame, LLMFullResponseEndFrame

    @llm.event_handler("on_before_process_frame")
    async def _diag_llm_frame(llm_svc, data):
        frame = data.frame if hasattr(data, 'frame') else data
        fname = type(frame).__name__
        if fname == "LLMContextFrame":
            logger.info("[LLM] >>> LLMContextFrame received — generation starting")
        elif fname == "InterruptionFrame":
            logger.warning("[LLM] !!! InterruptionFrame — generation will be CANCELLED")

    @llm.event_handler("on_after_push_frame")
    async def _diag_llm_push(llm_svc, data):
        frame = data.frame if hasattr(data, 'frame') else data
        fname = type(frame).__name__
        if fname == "LLMFullResponseEndFrame":
            logger.info("[LLM] <<< LLMFullResponseEndFrame — generation FINISHED")
        elif fname == "LLMFullResponseStartFrame":
            logger.info("[LLM] --- LLMFullResponseStartFrame pushed")

    @llm.event_handler("on_completion_timeout")
    async def _diag_llm_timeout(llm_svc):
        logger.warning("[LLM] !!! Generation TIMEOUT")

    # STT transcript logging — see exactly what the STT produced
    @stt.event_handler("on_after_push_frame")
    async def _diag_stt_push(stt_svc, data):
        frame = data.frame if hasattr(data, 'frame') else data
        fname = type(frame).__name__
        if fname == "TranscriptionFrame":
            text = getattr(frame, 'text', '?')
            finalized = getattr(frame, 'finalized', '?')
            logger.info(f"[STT] Transcript: {text!r} (finalized={finalized})")
        elif fname == "InterimTranscriptionFrame":
            text = getattr(frame, 'text', '?')
            logger.debug(f"[STT] Interim: {text!r}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Caller disconnected — saving call + running extraction")
        cm.get_state().ended_at = cm.get_state().ended_at or time.time()

        # ── Reconstruct transcript from context ──
        # Now that we no longer inject developer messages after every tool
        # call, context.messages is much cleaner:
        #   [system, developer(greeting), user/assistant/tool messages]
        # We extract only user and assistant messages for the transcript.
        try:
            for msg in context.messages:
                role = msg.get("role")
                content = str(msg.get("content", ""))
                if role == "user" and content:
                    cm.get_state().add_turn("user", content)
                elif role == "assistant" and content:
                    cm.get_state().add_turn("assistant", content)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not reconstruct transcript: {e}")

        save_call(call_id, phone_number, cm.get_state().to_dict())

        # ── Post-call extraction (bonus feature) ──
        # extract_lead() is SYNCHRONOUS (google-genai's generate_content
        # blocks for 1-3s). Running it inline would freeze the event loop,
        # potentially delaying other incoming WebSocket connections.
        # We offload it to a thread and fire-and-forget — call data is
        # already persisted by save_call() above, so losing extraction
        # on a rare failure is acceptable.
        def _extract_and_save():
            try:
                extracted = extract_lead(cm.get_state())
                save_lead(call_id, extracted)
                logger.info(f"Lead extraction complete for call {call_id}")
            except Exception as e:
                logger.warning(f"Lead extraction failed (non-fatal): {e}")

        asyncio.get_event_loop().run_in_executor(None, _extract_and_save)

        # ── Performance summary ──
        perf.print_summary()
        perf.save_to_file()

        await task.cancel()

    # ── Run ──

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)