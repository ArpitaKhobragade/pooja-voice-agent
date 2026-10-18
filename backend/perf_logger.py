"""
Performance instrumentation for the Sia voice pipeline.

Captures these metrics per call:
  - LLM generation count (and count per user turn)
  - LLM TTFB and total processing time
  - STT processing latency
  - TTS latency
  - Tool execution latency (timed directly in handlers)
  - Prompt / completion token counts
  - Estimated end-to-end turn latency

Data sources:
  - Pipecat's built-in metrics events  (enable_metrics=True on PipelineTask)
  - Explicit timing in tool handlers   (time.time() around apply_tool_call)
  - Usage metrics from the LLM service (enable_usage_metrics=True)

Outputs:
  - Real-time [PERF] log lines via loguru (visible in console)
  - Structured JSON file in  backend/logs/call_<call_id>.json
  - Human-readable summary printed at call end
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from pipecat.frames.frames import MetricsFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

LOGS_DIR = Path(__file__).parent / "logs"


class CallPerformanceLogger:
    """Per-call performance tracker.  Create one instance per call in run_bot."""

    def __init__(self, call_id: str):
        self.call_id = call_id
        self.call_start = time.time()

        # ---- Explicit metrics (instrumented in our code) ----
        self.tool_executions: list[dict] = []       # {name, latency_ms, ts}
        self.token_usages: list[dict] = []          # {prompt_tokens, completion_tokens, ts}

        # ---- Passive metrics (from Pipecat events) ----
        self.pipecat_metrics_raw: list[dict] = []   # everything Pipecat emits
        self.llm_ttfb_ms: list[float] = []          # one entry per LLM generation
        self.llm_processing_ms: list[float] = []
        self.stt_processing_ms: list[float] = []
        self.tts_ttfb_ms: list[float] = []

        # ---- Turn tracking ----
        # Logic: every LLM TTFB that is NOT a post-tool-call continuation
        # represents a new user turn (or the initial greeting).
        self.generation_count: int = 0
        self.user_turn_count: int = 0
        self._expecting_post_tool_gen: bool = False
        self._greeting_done: bool = False  # first generation is the greeting

        # ---- Deterministic Extraction Metrics ----
        self.deterministic_extractions_successful: int = 0
        self.llm_fallbacks: int = 0
        self.latency_deterministic_path: list[float] = []
        self.latency_llm_path: list[float] = []
        self.redundant_tool_calls: int = 0

        LOGS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # Explicit metrics  (called from tool handlers / bot.py)              #
    # ------------------------------------------------------------------ #

    def mark_tool_execution(self, tool_name: str, latency_ms: float) -> None:
        self.tool_executions.append({
            "name": tool_name,
            "latency_ms": round(latency_ms, 2),
            "ts": time.time(),
        })
        # Next LLM generation will be a post-tool continuation, not a new turn
        self._expecting_post_tool_gen = True
        logger.info(f"[PERF] Tool '{tool_name}': {latency_ms:.1f}ms")

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.token_usages.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ts": time.time(),
        })
        logger.info(f"[PERF] Tokens: {prompt_tokens} prompt / {completion_tokens} completion")

    # ------------------------------------------------------------------ #
    # Passive metrics  (called from @task / @llm event handlers)          #
    # ------------------------------------------------------------------ #

    def record_pipecat_metric(self, metric: Any) -> None:
        """Process a single metric object from Pipecat's on_metrics event."""
        try:
            processor = getattr(metric, "processor", "")
            value = getattr(metric, "value", None)
            metric_type = type(metric).__name__

            if value is None:
                return

            value_ms = round(value * 1000, 1) if isinstance(value, (int, float)) else 0

            self.pipecat_metrics_raw.append({
                "type": metric_type,
                "processor": processor,
                "value_ms": value_ms,
                "ts": time.time(),
            })

            # Classify the metric by processor name (defensive matching)
            p = (processor or "").lower()
            mt = metric_type.lower()

            if "ttfb" in mt:
                if "groq" in p or "llm" in p or "google" in p or "gemini" in p:
                    self._on_llm_ttfb(value_ms)
                elif "tts" in p or ("sarvam" in p and "stt" not in p):
                    self.tts_ttfb_ms.append(value_ms)
                    logger.info(f"[PERF] TTS TTFB: {value_ms:.0f}ms")
                elif "stt" in p:
                    self.stt_processing_ms.append(value_ms)
                    logger.info(f"[PERF] STT TTFB: {value_ms:.0f}ms")

            elif "processing" in mt:
                if "groq" in p or "llm" in p or "google" in p or "gemini" in p:
                    self.llm_processing_ms.append(value_ms)
                    logger.info(f"[PERF] LLM total: {value_ms:.0f}ms")
                elif "stt" in p:
                    self.stt_processing_ms.append(value_ms)
                    logger.info(f"[PERF] STT processing: {value_ms:.0f}ms")
                elif "tts" in p:
                    self.tts_ttfb_ms.append(value_ms)
                    logger.info(f"[PERF] TTS processing: {value_ms:.0f}ms")

        except Exception as e:
            logger.debug(f"[PERF] Could not classify metric: {e}")

    def _on_llm_ttfb(self, value_ms: float) -> None:
        """Called for every LLM TTFB event.  Handles generation counting
        and user-turn detection."""
        self.generation_count += 1
        self.llm_ttfb_ms.append(value_ms)

        if not self._greeting_done:
            # First ever generation = the opening greeting, not a user turn
            self._greeting_done = True
            logger.info(
                f"[PERF] LLM TTFB: {value_ms:.0f}ms  "
                f"(generation #{self.generation_count} — greeting)"
            )
        elif self._expecting_post_tool_gen:
            # This generation was triggered by a tool result_callback,
            # same user turn as the previous generation.
            self._expecting_post_tool_gen = False
            logger.info(
                f"[PERF] LLM TTFB: {value_ms:.0f}ms  "
                f"(generation #{self.generation_count} — post-tool response)"
            )
        else:
            # New user turn detected (LLM received a new user message)
            self.user_turn_count += 1
            logger.info(
                f"[PERF] LLM TTFB: {value_ms:.0f}ms  "
                f"(generation #{self.generation_count} — user turn #{self.user_turn_count})"
            )

    # ------------------------------------------------------------------ #
    # Summary + persistence                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stat(values: list[float]) -> str:
        if not values:
            return "n/a"
        if len(values) == 1:
            return f"{values[0]:.0f}ms"
        avg = sum(values) / len(values)
        return f"avg={avg:.0f}  min={min(values):.0f}  max={max(values):.0f}ms  (n={len(values)})"

    def print_summary(self) -> None:
        """Print a human-readable summary.  Call once at call end."""
        duration = time.time() - self.call_start
        gens = self.generation_count
        turns = self.user_turn_count or 1
        gpt = gens / turns if turns else 0

        logger.info("")
        logger.info("=" * 74)
        logger.info(f"  PERFORMANCE SUMMARY — call {self.call_id[:12]}…")
        logger.info("=" * 74)
        logger.info(f"  Call duration          : {duration:.1f}s")
        logger.info(f"  User turns             : {self.user_turn_count}")
        logger.info(f"  LLM generations        : {gens}")
        logger.info(f"  Generations / turn     : {gpt:.1f}")
        logger.info(f"  Tool executions        : {len(self.tool_executions)}")
        logger.info(f"  Redundant tool calls   : {self.redundant_tool_calls}")
        logger.info(f"  Deterministic turns    : {self.deterministic_extractions_successful}")
        logger.info(f"  LLM fallback turns     : {self.llm_fallbacks}")
        logger.info("-" * 74)
        logger.info(f"  STT latency            : {self._stat(self.stt_processing_ms)}")
        logger.info(f"  LLM TTFB               : {self._stat(self.llm_ttfb_ms)}")
        logger.info(f"  LLM total              : {self._stat(self.llm_processing_ms)}")
        logger.info(f"  TTS latency            : {self._stat(self.tts_ttfb_ms)}")
        logger.info("-" * 74)

        if self.tool_executions:
            logger.info("  Tool breakdown:")
            for t in self.tool_executions:
                logger.info(f"    {t['name']:25s} {t['latency_ms']:>8.1f}ms")

        if self.token_usages:
            total_pt = sum(u["prompt_tokens"] for u in self.token_usages)
            total_ct = sum(u["completion_tokens"] for u in self.token_usages)
            logger.info("-" * 74)
            logger.info(f"  Total tokens           : {total_pt} prompt / {total_ct} completion")
            logger.info("  Per-generation token growth:")
            for i, u in enumerate(self.token_usages, 1):
                logger.info(f"    gen {i:>2d}: {u['prompt_tokens']:>5d} prompt  {u['completion_tokens']:>5d} completion")

        logger.info("=" * 74)
        logger.info("")

    def save_to_file(self) -> None:
        """Persist all metrics to a JSON file for later analysis / diffing."""
        filepath = LOGS_DIR / f"call_{self.call_id}.json"
        turns = self.user_turn_count or 1
        data = {
            "call_id": self.call_id,
            "call_start_utc": self.call_start,
            "call_duration_s": round(time.time() - self.call_start, 2),
            "user_turn_count": self.user_turn_count,
            "llm_generation_count": self.generation_count,
            "generations_per_turn": round(self.generation_count / turns, 2),
            "redundant_tool_calls": self.redundant_tool_calls,
            "deterministic_extractions_successful": self.deterministic_extractions_successful,
            "llm_fallbacks": self.llm_fallbacks,
            "summary_ms": {
                "stt": self.stt_processing_ms,
                "llm_ttfb": self.llm_ttfb_ms,
                "llm_total": self.llm_processing_ms,
                "tts": self.tts_ttfb_ms,
            },
            "tool_executions": self.tool_executions,
            "token_usages": self.token_usages,
            "pipecat_metrics_raw": self.pipecat_metrics_raw,
        }
        filepath.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"[PERF] Metrics saved → {filepath}")


class PerformanceMetricsObserver(BaseObserver):
    """Observer to intercept Pipecat 1.5.0 MetricsFrames and feed them to the logger."""

    def __init__(self, perf_logger: CallPerformanceLogger):
        super().__init__()
        self.perf = perf_logger
        self._frames_seen = set()

    async def on_push_frame(self, data: FramePushed):
        """Intercept pushed frames and extract metrics."""
        frame = data.frame
        if not isinstance(frame, MetricsFrame):
            return

        # Avoid double-counting if a frame passes through multiple points
        if id(frame) in self._frames_seen:
            return
        self._frames_seen.add(id(frame))

        for metric in frame.data:
            self.perf.record_pipecat_metric(metric)
