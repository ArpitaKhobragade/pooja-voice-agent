#!/usr/bin/env python
"""
Pooja — standalone local AI voice agent for HostelPGDekho.com.

Launch with:
    python pooja_cli.py                 # auto-picks the best available mode
    python pooja_cli.py --mode voice    # mic + speakers (Sarvam STT, Groq LLM, Cartesia TTS)
    python pooja_cli.py --mode text     # keyboard + Groq LLM (no STT/TTS needed)
    python pooja_cli.py --mode offline  # keyboard only, no LLM, no API keys, no network
    python pooja_cli.py --find-voice    # list Hindi female Cartesia voices from your account

See README.md for full setup and for exactly what is and isn't verified.
This is turn-based (Pooja speaks, then listens, then speaks again), with
best-effort barge-in (see local/audio_io.py's docstring for the honest
caveat about echo without headphones) — not a guaranteed full-duplex
phone call.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import threading
import sys
import time
import uuid
from types import SimpleNamespace

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from agent_config import AGENT_CONFIG, FINAL_STEP_TEXT, SCRIPT_STEPS, script_line  # noqa: E402
from local.reply_stream import END_MARKERS, reply_effects, stream_sentences  # noqa: E402
from local.voice_gender import GenderTracker  # noqa: E402
from conversation_manager import ConversationManager  # noqa: E402
from local_tools import LOCAL_TOOLS, VALID_SLOTS  # noqa: E402
from objective import determine_next_objective  # noqa: E402
from prompts import BASE_PERSONA_PROMPT, build_dynamic_block  # noqa: E402

MAX_TURNS = int(os.environ.get("POOJA_MAX_TURNS", "20"))
STOP_PHRASES = (
    "bye", "band karo", "stop", "exit", "quit", "rakhta hoon", "rakhti hoon",
    "goodbye", "stop calling", "mat karo", "dobara mat karna", "call mat karo",
    "don't call", "no more calls",
)
REPEAT_PHRASES = (
    "repeat", "say that again", "say again", "come again",
    "dobara", "phir se", "ek baar fir", "dubara",
)


GOODNIGHT_RE = re.compile(r"good\s*night|shubh\s*ratri|\u0936\u0941\u092d\s*\u0930\u093e\u0924\u094d\u0930\u093f", re.IGNORECASE)
# Appended (not stored) after the conversation on every turn: the last thing
# the model reads, so style rules hold even deep into a call.
TURN_REMINDER = (
    "Reply as Pooja: warm and conversational. If you are saying the NEXT SCRIPT STEP use its wording almost exactly "
    "(up to ~50 words); otherwise 1-2 short but complete sentences (under 30 words), "
    "Roman script only (no Devanagari), at most one question asked as a full polite sentence, "
    "nothing already said repeated, and no dhanyavaad/goodbye unless the call is ending now."
)
CANNED_REPROMPT = "Sorry Sir, main theek se samajh nahi payi. Kya aap ek baar phir bata sakte hain?"
UNCLEAR_PHRASES = [
    "Sir, mujhe aapki awaaz clear nahi aayi, ek baar phir bolenge?",
    "Maaf kijiye Sir, ek baar phir bata dijiye?",
    "Sir, thoda clear nahi aaya, kya aap dobara bol sakte hain?",
]
SILENCE_PHRASES = [
    "Hello Sir, kya aap line par hain?",
    "Ji Sir, main sun rahi hoon, aap boliye.",
    "Sir, kya aap meri awaaz sun pa rahe hain?",
]
MAX_NO_INPUT_IN_A_ROW = 3
HISTORY_MESSAGES = 10  # recent messages sent to the LLM (older ones just cost tokens)

# "Ek minute" / "ruko" / "wait" / "hold on": the owner wants a moment or wants to speak.
# The right response is to stop and invite them to talk, never to carry on with the pitch.
HOLD_ON_RE = re.compile(
    r"\b(ek|1|one)\s*(minute|min|second|sec|moment)\b|\bwait\b|\bhold\s*on\b|\bruk(o|iye|na)\b|"
    r"\bthehr(o|iye)\b|\bjust\s+a\s+(minute|moment|sec|second)\b",
    re.IGNORECASE,
)
HOLD_ON_REPLIES = ["Ji, boliye.", "Haan ji Sir, boliye.", "Ji, main sun rahi hoon."]


class LLMUnavailable(Exception):
    """Raised when Groq failed on both the primary and fallback model for
    one request. Caught once in PoojaSession.run() to end the call with a
    safe, deterministic message instead of crashing the whole process."""


def _parse_and_validate_tool_call(tc) -> tuple[bool, dict, str]:
    """Parses one tool call's JSON arguments and validates them against
    the update_state schema BEFORE anything is applied to conversation
    state. Returns (is_valid, args, error_message). A malformed or
    invalid call never reaches ConversationManager — its rejection is
    reported back to the model as a normal tool result instead, so the
    model can self-correct next turn rather than the process crashing."""
    if tc.function.name != "update_state":
        return False, {}, f"unknown tool '{tc.function.name}'"

    raw = tc.function.arguments or ""
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, {}, f"malformed JSON in tool arguments: {e}"

    if not isinstance(args, dict):
        return False, {}, "tool arguments were not a JSON object"

    slot = args.get("slot")
    value = args.get("value")
    if not isinstance(slot, str) or slot not in VALID_SLOTS:
        return False, {}, f"'slot' is missing or not a recognized value (got {slot!r})"
    if not isinstance(value, str) or not value.strip():
        return False, {}, "'value' is missing or empty"

    return True, args, ""


def _detect_default_mode() -> str:
    if not os.environ.get("GROQ_API_KEY"):
        return "offline"
    try:
        from local import audio_io

        audio_io.check_devices()
    except Exception:
        return "text"
    if not os.environ.get("SARVAM_API_KEY"):
        return "text"
    return "voice"


def _tool_call_to_dict(tc) -> dict:
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
    low = text.strip().lower()
    return any(phrase in low for phrase in phrases)


_YES_TOKENS = {
    "haan", "han", "haa", "ha", "ji", "yes", "yeah", "yep", "ok", "okay", "theek", "thik",
    "sure", "bilkul", "achha", "accha", "acha", "bataiye", "batao", "boliye", "bolo",
    "samajh", "samjha", "sahi", "hmm", "hm", "interested",
    "achchha", "achcha", "acchha", "achhaa", "mm", "mmm", "mhm", "aage", "continue",
    # Devanagari, in case STT returns it despite translit mode.
    "हाँ", "हां", "हा", "जी", "ठीक", "बताइए", "बताओ", "बोलिए", "बोलो", "ओके", "अच्छा", "बिल्कुल",
    "समझ", "सही", "यस", "इंटरेस्टेड",
}
_NO_TOKENS = {"nahi", "nahin", "nahi.", "no", "nope", "mat", "नहीं", "नही", "मत", "नो"}
# "dobara batao" / "phir se samjhao": a request to explain again, not a "go ahead".
_REEXPLAIN_TOKENS = {
    "dobara", "dubara", "phir", "fir", "repeat", "again", "simple", "simply", "samjhao", "samjhana",
    "detail", "details", "detailed", "explain", "matlab",
}
_BUSY_PHRASES = ("busy", "time nahi", "baad me", "baad mein", "later", "abhi nahi", "meeting", "driving")
_LINK_TOKENS = {"link", "whatsapp", "url"}
_QUESTION_TOKENS = {
    "kya", "kaise", "kyun", "kyu", "kitna", "kitne", "kaun", "who", "what", "how", "why",
    "which", "where", "kab", "charge", "charges", "price", "cost", "paisa", "paise",
    "fees", "fee", "commission",
}


def classify_reply(text: str) -> str:
    """Cheap, deterministic read of a short owner reply: 'yes' | 'no' |
    'busy' | 'link' | 'other'. 'other' (questions, objections, anything
    long or ambiguous) is always handed to the LLM, so this only ever
    shortcuts plain acknowledgements — never guesses at real content."""
    low = text.lower()
    toks = re.findall(r"[a-z'ऀ-ॣ०-ॿ]+", low)
    if any(t in _LINK_TOKENS for t in toks):
        return "link"
    if "?" in text or any(t in _QUESTION_TOKENS for t in toks) or any(t in _REEXPLAIN_TOKENS for t in toks):
        return "other"
    if any(p in low for p in _BUSY_PHRASES):
        return "busy"
    has_yes = any(t in _YES_TOKENS for t in toks)
    has_no = any(t in _NO_TOKENS for t in toks)
    if has_no and not has_yes and len(toks) <= 4:
        return "no"
    if has_yes and not has_no and len(toks) <= 6:
        return "yes"
    return "other"


_KNOWN_SHORT = (
    _YES_TOKENS | _NO_TOKENS | _QUESTION_TOKENS | _LINK_TOKENS
    | {"hello", "hi", "hey", "namaste", "namaskar", "pranam", "bye", "sir", "madam", "mam", "busy", "free",
       "app", "stop", "band", "kaun", "sunao", "suniye", "nahin", "thanks", "thank", "sorry", "please"}
)
_GREETING_TOKENS = {"hello", "hi", "hey", "namaste", "namaskar", "pranam", "hallo", "helo", "ji"}


_END_CUES = re.compile(
    r"\b(nahi|nahin|nhi|mat|band|baad|busy|zaroorat|interest|manager|owner|human|galat|wrong|disturb|rakh|"
    r"bakwas|faltu|bekar|chup|pagal|stupid|idiot|shut|stop|later|sochunga|sochungi|dekhte)\b",
    re.IGNORECASE,
)


def owner_signalled_end(text: str) -> bool:
    """True when the owner's own words suggest ending / postponing the call (a no, busy, stop, a
    request for someone else, rudeness). An end marker the LLM attaches to anything else is a mistake."""
    return classify_reply(text) in ("no", "busy") or bool(_END_CUES.search(text))


def looks_like_noise(text: str) -> bool:
    """A one- or two-word transcript made of words we don't recognise
    ("of", "Include the"): almost always speech-recognition noise, not an
    answer, so it is better to ask again than to let the LLM react to it."""
    toks = re.findall(r"[a-z'\u0900-\u0963\u0966-\u097f]+", text.lower())
    if "?" in text or len(toks) == 0 or len(toks) > 2:
        return False
    return not any(t in _KNOWN_SHORT for t in toks)


def is_greeting_only(text: str) -> bool:
    toks = re.findall(r"[a-z'\u0900-\u0963\u0966-\u097f]+", text.lower())
    return bool(toks) and all(t in _GREETING_TOKENS for t in toks) and any(t != "ji" for t in toks)


# Short natural reactions placed before a script step now and then, so it doesn't sound read out.
LEAD_INS = ["Ji.", "Achha.", "Haan ji."]


def _transcript_is_unreliable(text: str, duration_s: float, last_text: str | None) -> tuple[bool, str]:
    """Pure heuristic (no I/O) deciding whether an STT transcript should
    be trusted. Sarvam's synchronous REST endpoint doesn't return a
    confidence score, so this uses two practical proxies instead:
      - a one-word transcript from a clip well over a second long doesn't
        add up — more likely a truncated/garbled recognition than someone
        genuinely saying one word for that long;
      - the exact same transcript twice in a row is often a sign of a
        stuck/looping recognition rather than the owner deliberately
        repeating themselves verbatim.
    Returns (is_unreliable, human_readable_reason)."""
    # duration_s includes ~1s of trailing silence, so a normal one-word
    # answer ("okay", "nahi") records as 1.4-2s. Only flag a single word
    # from a clip long enough that it can't be explained by that padding.
    suspicious = len(text.split()) <= 1 and duration_s > 4.0
    # A repeated transcript is only suspicious for longer text; "haan" or
    # "okay" twice in a row is a perfectly normal conversation.
    duplicate = (
        len(text.split()) >= 3
        and text.strip().lower() == (last_text or "").strip().lower()
    )
    if suspicious:
        return True, "that's a very short transcript for how long you spoke"
    if duplicate:
        return True, "I got the exact same transcript as last time"
    return False, ""


class PoojaSession:
    """Shared conversation loop for both voice and text-LLM modes. The
    only difference between the two modes is how user input is captured
    and how Pooja's replies are delivered — both are injected as
    callables so this class has no I/O-specific code.

    `deliver_reply(text) -> bool` must return True if the owner
    barged in / interrupted that reply (voice mode only; text mode's
    deliver_reply always returns False)."""

    def __init__(self, llm, get_user_input, deliver_reply, call_id: str | None = None, status=None,
                 conversational: bool = False, prewarm=None, deliver_stream=None):
        """conversational=True runs the natural, streamed conversation: the
        opening is a fixed line, then every owner reply goes to the LLM, whose
        reply is streamed sentence by sentence (so speech starts at the first
        sentence). `deliver_stream(sentences) -> (text, interrupted)` plays
        those sentences as they arrive (voice mode); without it the sentences
        are joined and delivered once. `prewarm(texts)` pre-synthesizes fixed
        lines in the background."""
        self.conversational = conversational
        self._prewarm = prewarm
        self.deliver_stream = deliver_stream
        self._awaiting_interest = False
        self._unclear_count = 0
        self._no_input_streak = 0
        self._hold_on_count = 0
        self.stop_event = threading.Event()  # set by the web UI's Stop button
        self._covered_steps: set[str] = set()
        self._lead_in_due = False
        self._last_owner_text = ""
        self.gender = GenderTracker()
        self.honorific = "Sir"  # becomes "Ma'am" if the owner sounds like a woman
        self.llm = llm
        self.get_user_input = get_user_input
        self.deliver_reply = deliver_reply
        self.status = status or (lambda *a, **k: None)
        self.call_id = call_id or str(uuid.uuid4())
        self.cm = ConversationManager(phone_number="local-demo")
        self.cm.state.call_id = self.call_id
        self.messages: list[dict] = [{"role": "system", "content": BASE_PERSONA_PROMPT}]
        self._last_spoken = None

    def _refresh_dynamic_block(self) -> None:
        # messages[0] is always the static persona; messages[1] is always
        # this dynamic state/objective block, rebuilt (not appended) every
        # time so context never grows from stale state snapshots.
        objective = determine_next_objective(self.cm.get_state())
        note = f"Address the owner as \"{self.honorific}\" (use feminine-neutral wording if Ma'am).\n\n"
        if self.conversational and not self.cm.get_state().get_slot("permission_granted").is_filled():
            note += (
                "You already greeted the owner and asked if they can talk; their next message is the "
                "answer to that. Do not greet or introduce yourself again.\n\n"
            )
        block = {"role": "system", "content": build_dynamic_block(
            self.cm.get_state(), objective, note, answer_only=self.conversational
        )}
        if len(self.messages) >= 2 and self.messages[1]["role"] == "system":
            self.messages[1] = block
        else:
            self.messages.insert(1, block)

    def _speak(self, text: str, force: bool = False) -> None:
        text = self._addr(text)
        if text == self._last_spoken and not force:
            logger.warning("Duplicate assistant response suppressed (not re-spoken).")
            return
        self._last_spoken = text
        self.cm.get_state().add_turn("assistant", text)
        self.status("speaking")
        interrupted = bool(self.deliver_reply(text))
        self.cm.state.was_interrupted_last_turn = interrupted
        if interrupted:
            logger.info("[Barge-in] Owner started speaking — Pooja's turn was cut short.")

    def _safe_chat(self):
        """Wraps llm.chat() so a Groq failure (primary + fallback both
        exhausted) becomes LLMUnavailable instead of an unhandled SDK
        exception — callers higher up end the call gracefully."""
        from local.llm_groq import GroqLLMError

        try:
            return self.llm.chat(self.messages, tools=LOCAL_TOOLS)
        except GroqLLMError as e:
            raise LLMUnavailable(str(e)) from e

    def _run_tool_loop(self) -> str:
        """Calls the LLM, applying any VALID tool calls via
        ConversationManager, until it produces a final spoken response.
        Malformed/invalid tool calls are rejected before they ever reach
        ConversationManager — see _parse_and_validate_tool_call. Bounded
        to 4 tool-call hops so a model that keeps calling tools instead
        of answering can't loop forever."""
        self.status("thinking")
        message = self._safe_chat()

        hops = 0
        while getattr(message, "tool_calls", None) and hops < 4:
            hops += 1
            self.messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [_tool_call_to_dict(tc) for tc in message.tool_calls],
                }
            )
            for tc in message.tool_calls:
                valid, args, err = _parse_and_validate_tool_call(tc)
                if valid:
                    result = self.cm.apply_tool_call(tc.function.name, args)
                else:
                    result = {"ok": False, "error": err}
                    logger.warning(
                        f"[Tool] Rejected invalid tool call from model: {err} "
                        f"(raw arguments={tc.function.arguments!r})"
                    )
                logger.info(f"[Tool] {tc.function.name}({args}) -> {result}")
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
                )
            self._refresh_dynamic_block()
            self.status("thinking")
            message = self._safe_chat()

        return (message.content or "").strip()

    def _end_due_to_llm_failure(self, reason: str) -> ConversationManager:
        logger.error(f"Ending call — Groq is unavailable: {reason}")
        self.cm.state.call_ended = True
        self._speak(
            "Sorry Sir, abhi thoda technical issue ho raha hai mere system mein. "
            "Main thodi der baad phir try karungi. Dhanyavaad!",
            force=True,
        )
        return self.cm

    # ---- conversational path (streamed, no tool round-trip) ------------------

    def _next_script_step(self) -> str | None:
        """Text of the first approved script step not yet said, in the right
        form of address. After a yes to the interest question, the closing
        step (the app link) instead."""
        state = self.cm.get_state()
        if state.interested_confirmed:
            return self._addr(FINAL_STEP_TEXT.format(app_url=AGENT_CONFIG["app_url"]))
        if state.call_ended:
            return None
        for step in SCRIPT_STEPS:
            if step["id"] not in self._covered_steps:
                return self._addr(step["text"])
        return None

    def _mark_steps_covered(self, text: str) -> None:
        for step in SCRIPT_STEPS:
            if step["id"] not in self._covered_steps and re.search(step["covered"], text, re.IGNORECASE):
                self._covered_steps.add(step["id"])

    def _addr(self, text: str) -> str:
        """Uses the right form of address ("Sir" / "Ma'am") in fixed lines."""
        if self.honorific == "Ma'am":
            return re.sub(r"\b(Sir|Madam|Mam|Ma'am)\b", "Ma'am", text, flags=re.IGNORECASE)
        return text

    def _learn_owner(self, user_text: str) -> None:
        """Updates the Sir / Ma'am guess from the owner's words and voice."""
        self.gender.add_text(user_text)
        audio = getattr(self.get_user_input, "last_audio", None)
        if audio is not None:
            self.gender.add_audio(audio)
            self.get_user_input.last_audio = None
        guess = self.gender.guess
        new = "Ma'am" if guess == "female" else "Sir" if guess == "male" else self.honorific
        if new != self.honorific:
            logger.info(f"Address changed to {new!r} (voice pitch ~{self.gender.median_f0}, text cue considered).")
            self.honorific = new

    def _line(self, key: str) -> str:
        return self._addr(script_line(key))

    def _fixed_texts_to_prewarm(self) -> list[str]:
        script = [self._addr(step["text"]) for step in SCRIPT_STEPS]
        script += [self._addr(FINAL_STEP_TEXT.format(app_url=AGENT_CONFIG["app_url"]) + " Dhanyavaad Sir, take care!")]
        script += [self._line(k) for k in ("reintro", "closing_not_interested", "closing_busy_or_declined")]
        script += LEAD_INS
        return script + [self._line("closing_goodnight"), self._line("closing_no_response"), CANNED_REPROMPT] + UNCLEAR_PHRASES + SILENCE_PHRASES + [self._addr(r) for r in HOLD_ON_REPLIES]

    def _turn_reminder(self) -> str:
        """Style reminder placed last, after the conversation. If Pooja's
        previous reply ended in a question, this one must not: back-to-back
        questions are what makes a caller sound like a questionnaire."""
        reminder = TURN_REMINDER
        if self._last_spoken and self._last_spoken.rstrip().endswith("?"):
            reminder += (
                " Your previous message already ended with a question, so this reply must NOT end "
                "with a question: answer or react, then stop and let them talk."
            )
        return reminder

    @staticmethod
    def _is_hold_on(text: str) -> bool:
        """A short 'wait / ek minute / ruko' with no real question in it."""
        if not HOLD_ON_RE.search(text) or "?" in text:
            return False
        tokens = re.findall(r"[a-z\u0900-\u097f']+", text.lower())
        return len(tokens) <= 4 and not any(t in _QUESTION_TOKENS for t in tokens)

    def _stream_deltas(self):
        from local.llm_groq import GroqLLMError

        try:
            history = self.messages[:2] + self.messages[2:][-HISTORY_MESSAGES:]
            yield from self.llm.chat_stream(history + [{"role": "system", "content": self._turn_reminder()}])
        except GroqLLMError as e:
            raise LLMUnavailable(str(e)) from e

    def _deliver(self, sentences) -> tuple[str, bool]:
        if self.deliver_stream:
            return self.deliver_stream(sentences)
        text = " ".join(sentences).strip()
        if not text:
            return "", False
        self.status("speaking")
        return text, bool(self.deliver_reply(text))

    def _pitch_complete(self) -> bool:
        state = self.cm.get_state()
        return all(state.get_slot(slot).is_filled() for slot in (
            "benefit_smart_search_explained", "benefit_vacant_form_explained", "benefit_free_listing_explained",
        ))

    def _pre_llm_updates(self, user_text: str) -> None:
        """Cheap, certain state changes made before the LLM sees the reply,
        so the LLM's notes (and the end-of-call logic) are already right."""
        kind = classify_reply(user_text)
        state = self.cm.get_state()
        if kind == "link":
            self.cm.apply_tool_call("update_state", {"slot": "link_requested", "value": "yes"})
        if not state.get_slot("permission_granted").is_filled():
            if kind == "yes":
                self.cm.apply_tool_call("update_state", {"slot": "permission_granted", "value": "yes"})
            elif kind == "no":
                self.cm.apply_tool_call("update_state", {"slot": "permission_granted", "value": "no"})
        elif self._awaiting_interest and not state.get_slot("interest_decision").is_filled():
            decision = {"yes": "yes", "no": "no", "busy": "maybe"}.get(kind)
            if decision == "yes" and not self._pitch_complete():
                decision = None
            if decision:
                self.cm.apply_tool_call("update_state", {"slot": "interest_decision", "value": decision})

    def _apply_reply(self, text: str, tags: list[str]) -> None:
        """Updates call state from what Pooja just said and any end markers."""
        state = self.cm.get_state()
        effects = reply_effects(text, AGENT_CONFIG["app_url"])
        self._mark_steps_covered(text)
        for slot in effects["slots"]:
            self.cm.apply_tool_call("update_state", {"slot": slot, "value": "yes"})
        if effects["slots"] and not state.get_slot("permission_granted").is_filled():
            self.cm.apply_tool_call("update_state", {"slot": "permission_granted", "value": "yes"})
        if effects["link_read"]:
            state.link_sent = True
        if text.rstrip().endswith("?") and tags:
            # A goodbye never ends in a question; the model attached a marker to a
            # normal reply, so the conversation is continuing.
            logger.warning(f"Ignoring end marker(s) {tags}: the reply ended with a question.")
            tags = []
        if tags and not owner_signalled_end(self._last_owner_text) and not self._awaiting_interest:
            logger.warning(f"Ignoring end marker(s) {tags}: nothing the owner said asked to end the call.")
            tags = []
        for tag in tags:
            decision = END_MARKERS.get(tag)
            if decision == "yes" and not self._pitch_complete():
                # A plain "okay" mid-pitch is not interest; don't let a stray marker end the call.
                logger.warning("Ignoring [INTERESTED]: the pitch points have not all been covered yet.")
                continue
            if decision and not state.get_slot("interest_decision").is_filled():
                self.cm.apply_tool_call("update_state", {"slot": "interest_decision", "value": decision})
        self._awaiting_interest = (
            "interested" in text.lower() and not state.get_slot("interest_decision").is_filled()
        )

    def _scripted_reply(self, user_text: str) -> str | None:
        """The approved script, said by the code (not the LLM) whenever the
        owner just acknowledges: exact wording, in order, instant, and no
        tokens used. Returns None when the owner said something that needs a
        real answer, which goes to the LLM."""
        state = self.cm.get_state()

        # The interest question has just been answered (pre-LLM updates set the decision).
        if state.call_ended:
            if state.interested_confirmed:
                state.link_sent = True
                return self._addr(
                    FINAL_STEP_TEXT.format(app_url=AGENT_CONFIG["app_url"]) + " Dhanyavaad Sir, take care!"
                )
            if state.not_interested:
                return self._line("closing_not_interested")
            return self._line("closing_busy_or_declined")

        permission = state.get_slot("permission_granted").is_filled()
        if not permission and is_greeting_only(user_text):
            return self._line("reintro")  # they just said "Hello?": repeat who is calling

        if classify_reply(user_text) != "yes" or "interested" in user_text.lower():
            return None
        step = self._next_script_step()
        if step is None or self._awaiting_interest:
            return None
        if self._lead_in_due:
            self._lead_in_due = False
        elif random.random() < 0.3:
            self._lead_in_due = True  # next step goes without one, so it never becomes a tic
            step = f"{random.choice(LEAD_INS)} {step}"
        return step

    def _converse(self, user_text: str, retry: bool = False) -> None:
        """One owner turn. Plain acknowledgements get the next script step
        verbatim; anything else is answered by the LLM, streamed sentence by
        sentence. retry=True re-runs a turn whose LLM call failed, without
        recording the owner's message twice."""
        forced = None
        self._last_owner_text = user_text
        if not retry:
            self.messages.append({"role": "user", "content": user_text})
            self._pre_llm_updates(user_text)
            forced = self._scripted_reply(user_text)
        self._refresh_dynamic_block()

        tags: list[str] = []
        if forced:
            spoken = (self._addr(s) for s in stream_sentences([forced], tags, first_clause=False))
        else:
            self.status("thinking")
            spoken = (self._addr(s) for s in stream_sentences(self._stream_deltas(), tags))
        text, interrupted = self._deliver(spoken)
        text = text.strip()

        if not text and interrupted:
            # The owner spoke before Pooja had said anything. Say nothing: their words are
            # already captured and are handled as the next turn.
            self.cm.state.was_interrupted_last_turn = True
            return

        if not text:
            # A reply that is empty or only a marker. Apply the marker first: if it
            # ended the call, say the matching goodbye; otherwise re-prompt. Never silence.
            self._apply_reply("", tags)
            state = self.cm.get_state()
            if state.call_ended:
                key = ("closing_interested" if state.interested_confirmed
                       else "closing_not_interested" if state.not_interested
                       else "closing_busy_or_declined")
                self._speak(self._line(key), force=True)
            else:
                logger.warning("LLM returned an empty reply — using a short canned re-prompt.")
                self._speak(CANNED_REPROMPT, force=True)
            return

        if text:
            self.messages.append({"role": "assistant", "content": text})
            self._last_spoken = text
            self.cm.get_state().add_turn("assistant", text)
            self.cm.state.was_interrupted_last_turn = interrupted
            if interrupted:
                logger.info("[Barge-in] Owner started speaking — Pooja's turn was cut short.")
        self._apply_reply(text, tags)

    def run(self) -> ConversationManager:
        self._refresh_dynamic_block()

        if self.conversational:
            opening = self._line("opening")
            if self._prewarm:
                self._prewarm(self._fixed_texts_to_prewarm())
        else:
            try:
                opening = self._run_tool_loop()
            except LLMUnavailable as e:
                return self._end_due_to_llm_failure(str(e))
        if not opening:
            opening = script_line("opening")
        self.messages.append({"role": "assistant", "content": opening})
        self._speak(opening)

        turns = 0
        while not self.cm.get_state().call_ended and turns < MAX_TURNS and not self.stop_event.is_set():
            turns += 1
            self.status("listening")
            user_text = self.get_user_input()
            if self.stop_event.is_set():
                break

            if user_text is None:
                # force=True: this prompt can legitimately repeat; the
                # duplicate filter must never turn it into silence.
                self._no_input_streak += 1
                if self._no_input_streak >= MAX_NO_INPUT_IN_A_ROW:
                    self.cm.state.call_ended = True
                    self._speak(self._line("closing_no_response"), force=True)
                    break
                silent = getattr(self.get_user_input, "none_reason", "unclear") == "silence"
                phrases = SILENCE_PHRASES if silent else UNCLEAR_PHRASES
                self._speak(phrases[self._unclear_count % len(phrases)], force=True)
                self._unclear_count += 1
                continue
            self._no_input_streak = 0
            user_text = user_text.strip()
            if not user_text:
                continue

            self.cm.get_state().add_turn("user", user_text)
            self._learn_owner(user_text)

            if GOODNIGHT_RE.search(user_text):
                # Mirror a "good night" instead of a generic "have a good day".
                self.cm.state.call_ended = True
                self._speak(self._line("closing_goodnight"), force=True)
                break

            if _matches_any(user_text, STOP_PHRASES):
                # Deterministic opt-out — never routed through the LLM,
                # so it can't be reinterpreted or overridden, and it keeps
                # working even if Groq itself is down.
                self.cm.state.call_ended = True
                self._speak(self._line("closing_busy_or_declined"))
                break

            if not self.conversational and _matches_any(user_text, REPEAT_PHRASES) and self._last_spoken:
                # Deterministic repeat — re-speak the exact last line rather
                # than asking the LLM to regenerate it (avoids drifting
                # wording and avoids an unnecessary API call).
                self._speak(self._last_spoken, force=True)
                continue

            if self.conversational and self._is_hold_on(user_text):
                # "Ek minute" / "ruko": stop and invite them to talk. No LLM call, instant.
                reply = HOLD_ON_REPLIES[self._hold_on_count % len(HOLD_ON_REPLIES)]
                self._hold_on_count += 1
                self.messages.append({"role": "user", "content": user_text})
                self.messages.append({"role": "assistant", "content": self._addr(reply)})
                self._speak(reply, force=True)
                continue

            if self.conversational and looks_like_noise(user_text):
                # Speech-recognition noise ("of", "Include the"): ask again rather than react to it.
                logger.info(f"Ignoring likely noise transcript {user_text!r}.")
                self._speak(UNCLEAR_PHRASES[self._unclear_count % len(UNCLEAR_PHRASES)], force=True)
                self._unclear_count += 1
                continue

            if self.conversational:
                try:
                    self._converse(user_text)
                except LLMUnavailable:
                    # Often a brief rate limit: hold on for a moment and try once more
                    # instead of hanging up on the owner.
                    logger.warning("LLM unavailable for this turn; asking for a second and retrying once.")
                    self._speak("Ek second Sir, line thodi slow hai.", force=True)
                    time.sleep(4.0)
                    try:
                        self._converse(user_text, retry=True)
                    except LLMUnavailable as e2:
                        return self._end_due_to_llm_failure(str(e2))
                continue

            self.messages.append({"role": "user", "content": user_text})
            self._refresh_dynamic_block()
            try:
                reply = self._run_tool_loop()
            except LLMUnavailable as e:
                return self._end_due_to_llm_failure(str(e))
            if not reply and not self.cm.get_state().call_ended:
                # Never leave the owner in silence on an empty model reply.
                logger.warning("LLM returned an empty reply — using a short canned re-prompt.")
                reply = "Sorry Sir, main theek se samajh nahi payi. Kya aap ek baar phir bata sakte hain?"
            if reply:
                self.messages.append({"role": "assistant", "content": reply})
                self._speak(reply)

        if turns >= MAX_TURNS and not self.cm.get_state().call_ended:
            logger.warning("Max turn limit reached — ending call as a safety net.")
            self.cm.state.call_ended = True
            self._speak("Sir, aapka time ho gaya hai, main ab call wrap up karti hoon. Dhanyavaad!")

        return self.cm


class CallUI:
    """Minimal call-style terminal presentation: session timer and a
    LISTENING / THINKING / SPEAKING status line, so a screen recording
    reads like a live call rather than a bare REPL."""

    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> str:
        s = int(time.time() - self.t0)
        return f"{s // 60:02d}:{s % 60:02d}"

    def status(self, state: str) -> None:
        labels = {
            "listening": "LISTENING",
            "thinking": "THINKING",
            "speaking": "SPEAKING",
        }
        print(f"[{self.elapsed()}] >>> {labels.get(state, state.upper())}")

    def banner(self, lines: list[str]) -> None:
        width = max(len(l) for l in lines) + 4
        print("=" * width)
        for l in lines:
            print(f"  {l}")
        print("=" * width)


def run_voice_mode(args) -> ConversationManager:
    from local import audio_io, stt_sarvam
    from local.llm_groq import GroqLLM
    from local.tts_cartesia import TTSEngine, CartesiaUnavailable

    # ---- Preflight: fail fast with clear, specific errors instead of
    # discovering a missing/broken provider mid-conversation. ----
    audio_io.check_devices()

    if not os.environ.get("SARVAM_API_KEY"):
        raise RuntimeError(
            "SARVAM_API_KEY is not set — required for --mode voice (speech-to-text). "
            "Set it in .env, or use --mode text (keyboard input, same LLM conversation)."
        )

    llm = GroqLLM()  # raises a clear error if GROQ_API_KEY is missing

    tts_mode = "pyttsx3" if args.tts_engine == "pyttsx3" else "cartesia"
    try:
        tts = TTSEngine(
            mode=tts_mode, allow_fallback=args.allow_tts_fallback,
            cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tts_cache"),
        )
    except CartesiaUnavailable as e:
        raise RuntimeError(
            f"Cartesia TTS is not usable ({e}). Pooja's voice is Cartesia by default — "
            f"this is not masked as a silent fallback. Fix your CARTESIA_API_KEY / "
            f"CARTESIA_VOICE_ID (run 'python pooja_cli.py --find-voice'), or explicitly "
            f"choose an offline voice with '--tts-engine pyttsx3', or allow a same-turn "
            f"fallback with '--allow-tts-fallback'."
        ) from e

    vad = audio_io.FrameVAD()

    # Full-duplex audio (always-open mic + speaker) is what makes real
    # interruption possible. If it can't be opened, fall back to the old
    # one-way audio with barge-in off, and say so.
    duplex = None
    if not args.no_barge_in and tts.mode == "cartesia":
        try:
            from local.duplex import DuplexAudio

            duplex = DuplexAudio(vad, debug=args.debug)
            duplex.open()
        except Exception as e:  # noqa: BLE001
            print(f"[Audio] Could not open full-duplex audio ({e}); barge-in is off for this run.")
            duplex = None

    ui = CallUI()
    tts_desc = (
        f"Cartesia (model={tts.cartesia.model_id}, voice={tts.cartesia.voice_id}, "
        f"lang={tts.cartesia.language}, speed={tts.cartesia.speed})"
        if tts.mode == "cartesia" else "pyttsx3 (offline, explicitly selected)"
    )
    ui.banner([
        "Pooja -- HostelPGDekho.com  [LOCAL VOICE CALL]",
        f"Audio: {audio_io.describe_devices()}",
        f"STT: Sarvam ({stt_sarvam.DEFAULT_MODEL})   VAD: {vad.backend}",
        f"LLM: Groq ({llm.model})",
        f"TTS: {tts_desc}",
        f"Barge-in: {'ON (echo-gated; headphones are best)' if duplex else 'disabled'}",
        f"Listening timing: silence_timeout={audio_io.SILENCE_TIMEOUT_S}s "
        f"(POOJA_SILENCE_TIMEOUT_S to tune)",
    ])

    last_transcript = {"text": None}

    def get_user_input():
        get_user_input.none_reason = "unclear"
        result = duplex.take_pending() if duplex else None
        if result is not None:
            print(f"[{ui.elapsed()}] [Barge-in] You spoke over Pooja, she stopped to listen.")
        else:
            print(
                f"[{ui.elapsed()}] Speak now (pause ~{audio_io.SILENCE_TIMEOUT_S:.1f}s "
                f"when you're done)..."
            )
            result = duplex.listen() if duplex else audio_io.record_utterance(vad=vad)
        if result is None:
            get_user_input.none_reason = "silence"  # nobody spoke within the wait
            return None
        ui.status("thinking")

        if args.debug:
            print(
                f"[DEBUG audio] duration={result.duration_s:.2f}s rms={result.rms:.0f} "
                f"peak={result.peak} clipped={result.is_clipped} vad={result.vad_backend}"
            )

        if result.is_silent:
            print(
                f"[STT] That recording was near-silent (RMS={result.rms:.0f}, "
                f"threshold={audio_io.SILENCE_RMS_THRESHOLD:.0f}) — not sending it to Sarvam. "
                f"Check your mic input level, or run 'python pooja_cli.py --diagnose' to test it."
            )
            return None
        if result.is_clipped:
            print(
                "[STT] Warning: this recording looks clipped (mic gain may be too high) — "
                "transcription may be less accurate."
            )

        t0 = time.time()
        try:
            # Drop the silent tail endpointing leaves: shorter upload, faster transcript.
            clip = audio_io.trim_trailing_silence(result.pcm_int16, result.sample_rate)
            get_user_input.last_audio = clip
            text = stt_sarvam.transcribe(clip, result.sample_rate)
        except Exception as e:
            logger.error(f"STT failed: {e}")
            print("[STT] Could not transcribe that. Please type your response instead:")
            return input("You (typed)> ")
        stt_latency = time.time() - t0

        if args.debug:
            print(f"[DEBUG stt] transcript={text!r} latency={stt_latency:.2f}s")

        if not text:
            return None

        unreliable, reason = _transcript_is_unreliable(text, result.duration_s, last_transcript["text"])
        last_transcript["text"] = text
        if unreliable:
            print(f"[STT] Transcript {text!r} looked unreliable ({reason}) — asking you to repeat.")
            return None

        print(f"[You said] {text}")
        return text

    def duplex_play(sentences) -> tuple[str, bool]:
        """Streams each sentence's audio into the always-open speaker while
        it is still being generated, and watches the mic for the owner talking
        over her (shared with the browser mode in local/playback.py). If
        interrupted, the owner's utterance is already in duplex.pending_result."""
        from local.playback import play_sentences

        def first_audio(elapsed, first_chunk):
            ui.status("speaking")
            if args.debug:
                print(
                    f"[DEBUG latency] first audio {elapsed:.2f}s after the reply began "
                    f"(tts first chunk {first_chunk:.2f}s)"
                )

        def tts_error(e):
            print(f"[TTS ERROR] {e} (re-run with --allow-tts-fallback to use pyttsx3 instead)")

        return play_sentences(
            sentences, tts.cartesia, duplex,
            allow_fallback=args.allow_tts_fallback,
            on_sentence=lambda sentence: print(f"[Pooja] {sentence}"),
            on_first_audio=first_audio,
            on_tts_error=tts_error,
        )

    def deliver_reply(text: str) -> bool:
        if duplex is not None:
            return duplex_play([text])[1]

        print(f"[Pooja] {text}")

        def play_fn(pcm, sr):
            if args.no_barge_in:
                audio_io.play_pcm_int16(pcm, sr)
                return False
            return audio_io.play_interruptible(pcm, sr, vad=vad)

        try:
            result = tts.speak(text, play_fn=play_fn)
        except CartesiaUnavailable as e:
            # Mandatory mode, no fallback allowed: surface the failure
            # loudly rather than pretending the turn was spoken.
            logger.error(f"Cartesia TTS failed and fallback is not enabled: {e}")
            print(f"[TTS ERROR] {e} (re-run with --allow-tts-fallback to use pyttsx3 instead)")
            raise
        tag = f"[TTS ENGINE: {result.engine.upper()}]"
        if result.warning:
            tag += f" ({result.warning})"
        print(tag)
        if args.debug and result.synthesis_s is not None:
            print(
                f"[DEBUG tts] synthesis={result.synthesis_s:.2f}s "
                f"playback={result.playback_s:.2f}s model={result.model_id}"
            )
        return result.interrupted

    def deliver_stream(sentences):
        """Speaks sentences as the LLM produces them: a worker thread pulls
        sentences (waiting on the LLM) and synthesizes each, while this
        thread plays the previous one, so audio starts at sentence 1 and
        the gaps between sentences are hidden."""
        if duplex is not None:
            return duplex_play(sentences)
        if tts.mode != "cartesia":
            text = " ".join(sentences).strip()
            return (text, bool(deliver_reply(text))) if text else ("", False)

        audio_q: queue.Queue = queue.Queue(maxsize=3)
        stop = threading.Event()
        t_start = time.time()

        def worker():
            try:
                for sentence in sentences:
                    if stop.is_set():
                        break
                    try:
                        pcm, sr = tts.cartesia.synthesize(sentence)
                    except CartesiaUnavailable:
                        if not args.allow_tts_fallback:
                            raise
                        audio_q.put(("fallback", sentence, None, None))
                        continue
                    audio_q.put(("audio", sentence, pcm, sr))
            except BaseException as e:  # noqa: BLE001 - re-raised on the playing thread
                audio_q.put(("error", e, None, None))
            finally:
                audio_q.put(("done", None, None, None))

        threading.Thread(target=worker, name="reply-synth", daemon=True).start()

        spoken: list[str] = []
        interrupted = False
        first = True
        while True:
            kind, a, pcm, sr = audio_q.get()
            if kind == "done":
                break
            if kind == "error":
                stop.set()
                if isinstance(a, CartesiaUnavailable):
                    print(f"[TTS ERROR] {a} (re-run with --allow-tts-fallback to use pyttsx3 instead)")
                raise a
            if interrupted:
                continue  # drain: the owner already cut Pooja off
            if first:
                first = False
                ui.status("speaking")
                if args.debug:
                    print(f"[DEBUG latency] first audio ready {time.time() - t_start:.2f}s after the owner finished")
            print(f"[Pooja] {a}")
            spoken.append(a)
            if kind == "fallback":
                from local.tts_cartesia import speak_pyttsx3

                speak_pyttsx3(a)
                continue
            if args.no_barge_in:
                audio_io.play_pcm_int16(pcm, sr)
            elif audio_io.play_interruptible(pcm, sr, vad=vad):
                interrupted = True
                stop.set()
        return " ".join(spoken), interrupted

    prewarm = tts.cartesia.prewarm if tts.mode == "cartesia" else None
    session = PoojaSession(
        llm, get_user_input, deliver_reply, status=ui.status, conversational=True,
        prewarm=prewarm, deliver_stream=deliver_stream,
    )
    try:
        return session.run()
    finally:
        if duplex is not None:
            duplex.close()


def run_text_mode() -> ConversationManager:
    from local.llm_groq import GroqLLM

    llm = GroqLLM()
    ui = CallUI()
    ui.banner([
        "Pooja -- HostelPGDekho.com  [TEXT MODE]",
        f"LLM: Groq ({llm.model})",
        "Type your replies as the PG owner.",
    ])

    def get_user_input():
        return input(f"[{ui.elapsed()}] You (owner)> ")

    def deliver_reply(text: str) -> bool:
        print(f"[Pooja] {text}")
        return False

    session = PoojaSession(llm, get_user_input, deliver_reply, status=ui.status, conversational=True)
    return session.run()


def run_find_voice():
    from local.tts_cartesia import list_hindi_female_voices, CartesiaUnavailable

    try:
        voices = list_hindi_female_voices()
    except CartesiaUnavailable as e:
        print(f"[Error] {e}")
        sys.exit(1)

    if not voices:
        print("No Hindi/feminine voices found on this Cartesia account. "
              "Browse https://play.cartesia.ai to pick one manually, or widen "
              "the search in local/tts_cartesia.py's list_hindi_female_voices().")
        return

    print("Hindi, feminine-presenting voices available on your Cartesia account:\n")
    for v in voices:
        print(f"  id={v['id']}")
        print(f"    name={v['name']!r}  tagline={v['tagline']!r}  language={v['language']}")
    print("\nSet CARTESIA_VOICE_ID in .env to one of the ids above.")


def run_audition_voices(sample_text: str | None):
    from local.tts_cartesia import CartesiaUnavailable
    from local.voice_audition import run_audition_cli

    try:
        run_audition_cli(sample_text=sample_text)
    except CartesiaUnavailable as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupted] Audition stopped early — no voice was selected.")


def _save_diagnostic_wav(pcm, sample_rate: int, prefix: str) -> str:
    """Only ever called after an explicit y/N confirmation from the
    diagnostic menu — never automatic. Saved locally under
    backend/diagnostics/; never uploaded or persisted anywhere else."""
    import datetime

    from local import audio_io

    os.makedirs("diagnostics", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join("diagnostics", f"{prefix}_{ts}.wav")
    with open(path, "wb") as f:
        f.write(audio_io.pcm_int16_to_wav_bytes(pcm, sample_rate))
    return path


def _diagnose_mic_playback():
    from local import audio_io

    audio_io.check_devices()
    print(f"Audio devices: {audio_io.describe_devices()}")
    vad = audio_io.FrameVAD()
    print(f"VAD backend: {vad.backend}")
    print("Speak now — this records with the exact same timing as --mode voice...")
    result = audio_io.record_utterance(vad=vad)
    if result is None:
        print("No speech detected (timed out waiting for you to start talking).")
        return
    print(
        f"Recorded {result.duration_s:.2f}s | RMS={result.rms:.0f} | peak={result.peak} | "
        f"silent={result.is_silent} | clipped={result.is_clipped}"
    )
    print("Playing it back now — listen for yourself whether this is what you actually said:")
    audio_io.play_pcm_int16(result.pcm_int16, result.sample_rate)
    if input("Save this clip to a local WAV file for inspection? [y/N] ").strip().lower() == "y":
        print(f"Saved: {_save_diagnostic_wav(result.pcm_int16, result.sample_rate, 'mic_test')}")


def _diagnose_stt():
    from local import audio_io, stt_sarvam

    if not os.environ.get("SARVAM_API_KEY"):
        print("[Error] SARVAM_API_KEY is not set.")
        return
    audio_io.check_devices()
    vad = audio_io.FrameVAD()
    print(f"Audio devices: {audio_io.describe_devices()}")
    print("Speak now...")
    result = audio_io.record_utterance(vad=vad)
    if result is None:
        print("No speech detected.")
        return
    print(
        f"Recorded {result.duration_s:.2f}s | RMS={result.rms:.0f} | "
        f"silent={result.is_silent} | clipped={result.is_clipped}"
    )
    print("Playing back exactly what will be sent to Sarvam...")
    audio_io.play_pcm_int16(result.pcm_int16, result.sample_rate)
    t0 = time.time()
    try:
        text = stt_sarvam.transcribe(result.pcm_int16, result.sample_rate)
    except Exception as e:
        print(f"[Error] Sarvam STT failed: {e}")
        return
    print(f"Sarvam transcript ({time.time() - t0:.2f}s): {text!r}")
    if input("Save this clip to a local WAV file? [y/N] ").strip().lower() == "y":
        print(f"Saved: {_save_diagnostic_wav(result.pcm_int16, result.sample_rate, 'stt_test')}")


def _diagnose_groq_plain():
    from local.llm_groq import GroqLLM, GroqLLMError

    try:
        llm = GroqLLM()
    except GroqLLMError as e:
        print(f"[Error] {e}")
        return
    print(f"Primary model: {llm.primary_model} | Fallback: {llm.fallback_model}")
    msg = input("Type a message to send (plain chat, no tools)> ")
    t0 = time.time()
    try:
        response = llm.chat([{"role": "user", "content": msg}])
    except GroqLLMError as e:
        print(f"[Error] Both models failed: {e}")
        return
    print(f"[{time.time() - t0:.2f}s, model used={llm.model}] {response.content}")


def _diagnose_groq_tools():
    from local.llm_groq import GroqLLM, GroqLLMError
    from local_tools import LOCAL_TOOLS

    try:
        llm = GroqLLM()
    except GroqLLMError as e:
        print(f"[Error] {e}")
        return

    print("Testing a prompt designed to trigger a VALID update_state tool call...")
    try:
        msg = llm.chat(
            [
                {
                    "role": "system",
                    "content": "Call update_state with slot='permission_granted' value='yes' "
                    "when the user agrees to talk.",
                },
                {"role": "user", "content": "Haan ji, bataiye."},
            ],
            tools=LOCAL_TOOLS,
        )
    except GroqLLMError as e:
        print(f"[Error] {e}")
        return
    print(f"model used={llm.model} | tool_calls={msg.tool_calls} | content={msg.content!r}")

    print("\nNow testing the argument validator against an INTENTIONALLY INVALID tool call...")
    bad_call = SimpleNamespace(id="x", function=SimpleNamespace(name="update_state", arguments="{bad json"))
    valid, args, err = _parse_and_validate_tool_call(bad_call)
    print(f"Rejected as expected: valid={valid} error={err!r} (never reached ConversationManager)")


def _diagnose_tts():
    from local.tts_cartesia import TTSEngine, CartesiaUnavailable

    try:
        tts = TTSEngine(mode="cartesia")
    except CartesiaUnavailable as e:
        print(f"[Error] {e}")
        return
    print(f"Voice: {tts.cartesia.voice_id} | model: {tts.cartesia.model_id} | speed: {tts.cartesia.speed}")
    text = input("Text to speak (Enter for the default opening line)> ").strip()
    if not text:
        from local.voice_audition import DEFAULT_SAMPLE_TEXT

        text = DEFAULT_SAMPLE_TEXT
    try:
        result = tts.speak(text)
    except CartesiaUnavailable as e:
        print(f"[Error] {e}")
        return
    print(
        f"engine={result.engine} synthesis={result.synthesis_s:.2f}s "
        f"playback={result.playback_s:.2f}s"
    )


def run_diagnose():
    """Interactive menu testing each piece of the pipeline in isolation.
    Never places a real phone call or contacts a PG owner — this project
    has no telephony integration to do that with in the first place."""
    print("=== Pooja Diagnostics ===")
    print("This never places a real call or contacts PG owners.")
    menu = (
        "\n  1) Microphone capture + local playback (hear exactly what was recorded)\n"
        "  2) Sarvam STT on a fresh recording\n"
        "  3) Groq: plain chat (no tools)\n"
        "  4) Groq: tool-calling (valid call + an intentionally invalid one)\n"
        "  5) Cartesia TTS with your configured voice\n"
        "  6) Full local conversation (same as --mode voice)\n"
        "  q) Quit\n"
    )
    actions = {
        "1": _diagnose_mic_playback,
        "2": _diagnose_stt,
        "3": _diagnose_groq_plain,
        "4": _diagnose_groq_tools,
    }
    while True:
        print(menu)
        choice = input("Choose> ").strip().lstrip("﻿").lower()
        if choice in ("q", "quit", "exit"):
            return
        if choice == "5":
            _diagnose_tts()
        elif choice == "6":
            default_args = argparse.Namespace(
                tts_engine="cartesia", allow_tts_fallback=False, no_barge_in=False, debug=False
            )
            run_voice_mode(default_args)
            return
        elif choice in actions:
            try:
                actions[choice]()
            except KeyboardInterrupt:
                print("\n[Interrupted]")
        else:
            print(f"Unrecognized choice: {choice!r}")


def run_audition_voices_web(host: str, port: int, no_browser: bool):
    from local.audition_server import run_server

    try:
        run_server(host=host, port=port, open_browser=not no_browser)
    except KeyboardInterrupt:
        print("\n[Stopped] Voice audition server shut down.")


def main():
    # Flush every line and use UTF-8 so output is live and readable even
    # when piped (e.g. into Tee-Object) instead of arriving only at exit.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    parser = argparse.ArgumentParser(description="Pooja — HostelPGDekho.com local voice agent")
    parser.add_argument(
        "--mode", choices=["voice", "text", "offline"], default=None,
        help="voice: mic+speakers. text: keyboard+LLM. offline: keyboard only, no API keys, no network.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument(
        "--find-voice", action="store_true",
        help="List Hindi female Cartesia voice ids from your account (needs CARTESIA_API_KEY) and exit.",
    )
    parser.add_argument(
        "--audition-voices", action="store_true",
        help="Play a sample line in each candidate Cartesia voice (needs CARTESIA_API_KEY) so you "
             "can pick a CARTESIA_VOICE_ID by ear; r=replay, n=next, q=quit. Never writes to .env "
             "or selects a voice automatically.",
    )
    parser.add_argument(
        "--audition-text", default=None,
        help="Override the sample line used by --audition-voices / --audition-voices-web.",
    )
    parser.add_argument(
        "--audition-voices-web", action="store_true",
        help="Launch a local browser page (http://127.0.0.1:8765 by default) with Play/Pause/"
             "Restart/Select controls for each candidate Cartesia voice (needs CARTESIA_API_KEY). "
             "Only listens on localhost. Never writes to .env until you click Save.",
    )
    parser.add_argument(
        "--audition-port", type=int, default=8765,
        help="Port for --audition-voices-web (default 8765).",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="With --audition-voices-web, don't auto-open a browser tab.",
    )
    parser.add_argument(
        "--tts-engine", choices=["cartesia", "pyttsx3"], default="cartesia",
        help="Voice mode only. Default 'cartesia' (mandatory unless --allow-tts-fallback). "
             "'pyttsx3' explicitly runs fully offline instead.",
    )
    parser.add_argument(
        "--allow-tts-fallback", action="store_true",
        help="Voice mode only. If Cartesia is unavailable or a call fails, fall back to "
             "pyttsx3 for that turn instead of stopping the session. Every fallback is "
             "reported, never silent.",
    )
    parser.add_argument(
        "--no-barge-in", action="store_true",
        help="Voice mode only. Disable best-effort interruption detection while Pooja speaks.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print transcript/intent/action/model/latency debug lines each turn. Never prints "
             "API keys or other secrets.",
    )
    parser.add_argument(
        "--input-device", default=None,
        help="Microphone to use: a sounddevice index or a substring of its name "
             "(see --list-devices). Overrides POOJA_INPUT_DEVICE for this run.",
    )
    parser.add_argument(
        "--output-device", default=None,
        help="Speaker to use: a sounddevice index or a substring of its name. "
             "Overrides POOJA_OUTPUT_DEVICE for this run.",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Open an interactive menu to test the microphone, Sarvam STT, Groq (with and "
             "without tools), and Cartesia TTS separately. Never places a real call.",
    )
    args = parser.parse_args()

    # Set before any `local.*` module is imported, since audio_io reads
    # these env vars once at import time.
    if args.input_device:
        os.environ["POOJA_INPUT_DEVICE"] = args.input_device
    if args.output_device:
        os.environ["POOJA_OUTPUT_DEVICE"] = args.output_device

    if args.list_devices:
        from local import audio_io

        print(audio_io.list_devices())
        return

    if args.find_voice:
        run_find_voice()
        return

    if args.audition_voices:
        run_audition_voices(args.audition_text)
        return

    if args.audition_voices_web:
        run_audition_voices_web(host="127.0.0.1", port=args.audition_port, no_browser=args.no_browser)
        return

    if args.diagnose:
        run_diagnose()
        return

    mode = args.mode or _detect_default_mode()
    print(f"=== Pooja (HostelPGDekho.com) — mode: {mode} ===")

    cm = None
    try:
        if mode == "voice":
            cm = run_voice_mode(args)
        elif mode == "text":
            cm = run_text_mode()
        else:
            from local.offline_script import run_offline_demo

            cm = run_offline_demo()
    except KeyboardInterrupt:
        print("\n[Interrupted] Ending call cleanly.")
    except Exception as e:
        logger.exception(f"Pooja encountered an unrecoverable error: {e}")
        print(f"\n[Error] {e}")
        sys.exit(1)
    finally:
        if cm is not None:
            try:
                import db

                db.init_db()
                state = cm.get_state()
                state.ended_at = state.ended_at or time.time()
                db.save_call(state.call_id, "local-demo", state.to_dict())
                print(f"\n[Saved] Transcript stored in backend/sia.db under call_id={state.call_id}")
            except Exception as e:  # pragma: no cover - persistence is best-effort
                logger.warning(f"Could not save call transcript: {e}")


if __name__ == "__main__":
    main()
