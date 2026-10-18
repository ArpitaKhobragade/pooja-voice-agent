"""
Post-call structured extraction (bonus feature).

Reuses the state already built live during the call (via update_state
tool calls) instead of re-deriving everything from scratch — the only
thing this adds is a written `summary` and a final confidence-aware
JSON shape suitable for the dashboard / a database row.

Uses Gemini's structured output mode (response_schema), not a "please
return JSON" instruction, so the shape is actually enforced.
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types

from state_store import CallState

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_name": {"type": "string"},
        "project": {"type": "string"},
        "timeline": {"type": "string"},
        "budget": {"type": "string"},
        "meeting_booked": {"type": "boolean"},
        "meeting_time": {"type": "string"},
        "language": {"type": "string", "description": "Primary language(s) used, e.g. 'Hindi', 'English', 'Hinglish'"},
        "not_interested": {"type": "boolean"},
        "summary": {"type": "string", "description": "2-3 sentence summary of the call"},
    },
    "required": ["meeting_booked", "not_interested", "summary"],
}


def _transcript_text(state: CallState) -> str:
    lines = []
    for turn in state.turns:
        speaker = "Caller" if turn["role"] == "user" else "Aisha"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def extract_lead(state: CallState) -> dict:
    """Runs one extra Gemini call after the call ends. Reuses live-tracked
    slots as a starting point and asks the model to fill in the rest
    (summary, language) from the full transcript."""

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    known_slots = state.filled_slots()

    prompt = f"""Here is a phone call transcript between an AI agent (Aisha, calling from \
Sunrise Interiors) and a caller who inquired about interior design work.

Information already tracked during the call: {json.dumps(known_slots)}
Meeting booked: {state.meeting_booked} (time: {state.meeting_time})
Not interested: {state.not_interested}

Transcript:
{_transcript_text(state)}

Fill in the structured lead record. Use the already-tracked information where available \
rather than re-guessing it, but fill customer_name, budget, language, and summary from the \
transcript. If something was never mentioned, leave it as an empty string."""

    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EXTRACTION_SCHEMA,
        ),
    )

    extracted = json.loads(response.text)
    # Fill in fields we're already certain about from tracked state,
    # in case the model's re-derivation disagrees.
    extracted["meeting_booked"] = state.meeting_booked
    extracted["not_interested"] = state.not_interested
    if state.meeting_time:
        extracted["meeting_time"] = state.meeting_time
    return extracted
