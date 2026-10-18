"""
Thin wrapper around the Groq chat-completions API for the local Pooja CLI.

Model ids — verified 2026-10-17, not guessed:
  - `openai/gpt-oss-120b` (default primary): listed as a Production text
    model on console.groq.com/docs/models, with tool-use/function-calling
    support documented at console.groq.com/docs/model/openai/gpt-oss-120b.
  - `llama-3.3-70b-versatile` was the PREVIOUS default fallback here, but
    a real run against this project hit `404 model_not_found` on it —
    Groq's deprecations page (console.groq.com/docs/deprecations)
    confirms it (and llama-3.1-8b-instant) were retired/scheduled for
    shutdown, with `openai/gpt-oss-20b` named as the recommended
    replacement. `openai/gpt-oss-20b` is confirmed Production and
    explicitly documents "tool use, and function calling" support
    (console.groq.com/docs/model/openai/gpt-oss-20b) — it's now the
    default fallback. NOT verified against a live API call (no credentials
    were available while writing this) — verify yourself once you have a
    real GROQ_API_KEY; see README's Test Results for exactly what was and
    wasn't exercised.

Error handling policy:
  - ONE fallback attempt per chat() call, always against the
    freshly-reconfigurable primary (never "sticky" to a broken fallback
    for the rest of the session — that was the bug behind "every
    subsequent turn crashed" after the 404).
  - If both the primary and fallback calls fail, chat() raises
    GroqAllModelsFailedError instead of retrying forever or propagating
    a raw SDK exception — callers (pooja_cli.py) catch this once and
    give a safe, deterministic spoken response instead of crashing.
  - Errors are categorized (auth / not-found / rate-limit / timeout /
    connection / other) using each exception's `status_code` where the
    SDK sets one, without requiring an exact-version import of every
    Groq exception subclass.
"""

from __future__ import annotations

import os
import re
import time

from loguru import logger

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.environ.get("GROQ_MODEL_FALLBACK", "openai/gpt-oss-20b")
# Reserved per request and counted against Groq's tokens-per-minute limit, so keep it
# just above a script step (~50 words) plus low-effort reasoning.
MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "260"))
REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low")
TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0.3"))
REQUEST_TIMEOUT_S = float(os.environ.get("GROQ_REQUEST_TIMEOUT_S", "20"))


class GroqLLMError(RuntimeError):
    pass


class GroqAllModelsFailedError(GroqLLMError):
    """Both the primary and fallback model calls failed for one request.
    Callers should treat this as "the LLM is unavailable right now", not
    as a bug to crash on — see pooja_cli.py's LLMUnavailable handling."""


_RETRY_IN = re.compile(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)(ms|s)", re.IGNORECASE)


def retry_after_seconds(message: str) -> float | None:
    """Seconds Groq says to wait ("Please try again in 10.3s" / "150ms" / "2m1.4s"),
    or None. Only meaningful for per-minute limits; a daily limit's wait is too long to use."""
    m = _RETRY_IN.search(message)
    if not m:
        return None
    minutes = float(m.group(1) or 0)
    value = float(m.group(2))
    return minutes * 60 + (value / 1000 if m.group(3).lower() == "ms" else value)


def _describe_error(e: Exception) -> str:
    status = getattr(e, "status_code", None)
    name = type(e).__name__
    if status == 401:
        return f"authentication failed — check GROQ_API_KEY ({e})"
    if status == 404:
        return f"model not found (404) — this model id may be retired or misspelled ({e})"
    if status == 429:
        return f"rate limited (429) ({e})"
    if isinstance(status, int) and status >= 500:
        return f"Groq server error ({status}) ({e})"
    if "Timeout" in name:
        return f"request timed out after {REQUEST_TIMEOUT_S}s ({e})"
    if "Connection" in name:
        return f"network/connection error ({e})"
    return f"{name}: {e}"


class GroqLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None, fallback_model: str | None = None):
        from groq import Groq  # imported lazily so --mode offline needs no groq install

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise GroqLLMError("GROQ_API_KEY is not set.")
        # max_retries=0: on a rate limit (429) the SDK would otherwise sleep
        # for the retry-after window, which is a multi-second dead pause on a
        # live call. Failing fast lets chat()/chat_stream() move straight to
        # the fallback model, which has its own separate rate-limit bucket.
        self._client = Groq(api_key=key, max_retries=0)
        self.primary_model = model or DEFAULT_MODEL
        self.fallback_model = fallback_model or FALLBACK_MODEL
        self.last_used_model = self.primary_model
        self.last_latency_s: float | None = None

    def _call(self, model: str, messages: list[dict], tools: list[dict] | None):
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=REQUEST_TIMEOUT_S,
        )
        if "gpt-oss" in model:
            # gpt-oss are reasoning models: hidden reasoning tokens count
            # against max_tokens, so a small budget with default effort
            # yields empty or truncated replies. Keep reasoning minimal.
            kwargs["reasoning_effort"] = REASONING_EFFORT
        if tools:
            # "auto" lets the model decide whether to call a tool or just
            # answer conversationally — never forced, per the requirement
            # not to force tool calls when a plain reply is appropriate.
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        completion = self._client.chat.completions.create(**kwargs)
        return completion.choices[0].message

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """Runs one chat-completion call, with one bounded fallback
        attempt on failure. Returns the raw Groq message object (has
        .content and .tool_calls). Raises GroqAllModelsFailedError if
        both the primary and fallback calls fail — never retries beyond
        that single fallback attempt."""
        t0 = time.time()
        try:
            message = self._call(self.primary_model, messages, tools)
            self.last_used_model = self.primary_model
            self.last_latency_s = time.time() - t0
            return message
        except Exception as primary_err:
            primary_reason = _describe_error(primary_err)

            if not self.fallback_model or self.fallback_model == self.primary_model:
                raise GroqAllModelsFailedError(
                    f"Groq model '{self.primary_model}' failed and no distinct fallback is "
                    f"configured: {primary_reason}"
                ) from primary_err

            logger.warning(
                f"Groq primary model '{self.primary_model}' failed ({primary_reason}); "
                f"trying fallback '{self.fallback_model}' once (not retried further)."
            )
            t1 = time.time()
            try:
                message = self._call(self.fallback_model, messages, tools)
            except Exception as fallback_err:
                fallback_reason = _describe_error(fallback_err)
                raise GroqAllModelsFailedError(
                    f"Both Groq models failed for this request — "
                    f"primary '{self.primary_model}': {primary_reason}; "
                    f"fallback '{self.fallback_model}': {fallback_reason}"
                ) from fallback_err

            self.last_used_model = self.fallback_model
            self.last_latency_s = time.time() - t1
            logger.info(f"Fallback model '{self.fallback_model}' succeeded.")
            return message

    def chat_stream(self, messages: list[dict]):
        """Streams a plain-text reply (no tools) as it is generated, so the
        caller can start speaking the first sentence immediately. Yields
        text chunks. Tries the primary model, then the fallback, only while
        nothing has been yielded yet; a failure after output has started
        simply ends the stream (the partial reply is still used). If every
        model was rate limited (429), waits 3s and tries once more, since a
        short pause beats ending the call. Raises GroqAllModelsFailedError
        if no model produced anything."""
        models = [self.primary_model]
        if self.fallback_model and self.fallback_model != self.primary_model:
            models.append(self.fallback_model)

        reasons: list[str] = []
        for round_no in range(2):
            reasons = []
            all_rate_limited = True
            waits: list[float] = []
            for model in models:
                t0 = time.time()
                started = False
                try:
                    kwargs = dict(
                        model=model,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        timeout=REQUEST_TIMEOUT_S,
                        stream=True,
                    )
                    if "gpt-oss" in model:
                        kwargs["reasoning_effort"] = REASONING_EFFORT
                    for chunk in self._client.chat.completions.create(**kwargs):
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta.content
                        if delta:
                            if not started:
                                self.last_latency_s = time.time() - t0  # time to first token
                            started = True
                            yield delta
                    self.last_used_model = model
                    return
                except Exception as e:  # noqa: BLE001
                    reason = _describe_error(e)
                    if started:
                        logger.warning(f"Groq stream from '{model}' broke mid-reply ({reason}); using what was received.")
                        return
                    reasons.append(f"'{model}': {reason}")
                    logger.warning(f"Groq stream from '{model}' failed ({reason}).")
                    if getattr(e, "status_code", None) != 429:
                        all_rate_limited = False
                    elif "per minute" in str(e):
                        hint = retry_after_seconds(str(e))
                        if hint is not None:
                            waits.append(hint)
            if round_no == 0 and all_rate_limited:
                # Wait as long as Groq says the per-minute window needs (capped), else a short default.
                pause = min(max(waits) + 0.3, 9.0) if waits else 3.0
                logger.warning(f"All Groq models are rate limited — waiting {pause:.1f}s and trying once more.")
                time.sleep(pause)
                continue
            break

        raise GroqAllModelsFailedError("All Groq models failed for this request — " + "; ".join(reasons))

    @property
    def model(self) -> str:
        """The model actually used by the most recent chat() call (starts
        as the configured primary before any call has been made)."""
        return self.last_used_model
