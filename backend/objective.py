"""
determine_next_objective(state) -> Objective

This stands in for a future generic "Workflow Engine". It answers
"what should happen next?" based purely on which required slots are
still empty in `state` right now — recomputed fresh every turn.

Deliberately NOT a linear state graph (no `next:` pointers). A caller
who volunteers their timeline before being asked shouldn't confuse
this — we just look at what's still missing and pick the
highest-priority gap.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_config import AGENT_CONFIG
from state_store import CallState


@dataclass
class Objective:
    kind: str  # "collect_slot" | "wrap_up" | "end_call"
    slot: str | None
    instruction: str
    missing_slots: list[str]


def _closing_instruction(state: CallState) -> str:
    """Pick a closing instruction that matches how the call actually ended,
    instead of a generic one-size-fits-all goodbye."""
    app_url = AGENT_CONFIG["app_url"]

    if state.interested_confirmed:
        link_note = (
            f"Offer to share the app link. This local demo cannot send WhatsApp "
            f"messages — say so plainly, then read out the app URL: {app_url}. "
            "Never claim a message was sent."
            if not state.link_sent
            else "The app URL has already been read out — do not repeat it."
        )
        return (
            "The owner agreed to use HostelPGDekho. Before ending:\n"
            f"- Thank them for their time.\n"
            f"- {link_note}\n"
            "- Wish them a good day and end warmly, in under three short sentences."
        )

    if state.not_interested:
        return (
            "The owner is not interested. Before ending:\n"
            "- Respect their decision without pushing further.\n"
            "- Thank them for their time.\n"
            "- Wish them a good day and end warmly, in under three short sentences."
        )

    return (
        "The owner could not talk right now or asked to end the call. Before ending:\n"
        "- Acknowledge that politely (no pressure, no renewed pitch).\n"
        "- Thank them for their time.\n"
        "- Wish them a good day and end warmly, in under three short sentences."
    )


def determine_next_objective(state: CallState) -> Objective:
    required = AGENT_CONFIG["required_slots"]
    descriptions = AGENT_CONFIG["slot_descriptions"]

    if state.is_terminal(AGENT_CONFIG["terminal_conditions"]):
        return Objective(
            kind="end_call",
            slot=None,
            instruction=_closing_instruction(state),
            missing_slots=[],
        )

    missing = state.missing_slots(required)

    if not missing:
        # All required slots filled but no terminal flag set yet —
        # nudge toward wrapping up (e.g. confirming the meeting decision).
        return Objective(
            kind="wrap_up",
            slot=None,
            instruction="All key information has been collected. Confirm next steps and close the call politely.",
            missing_slots=[],
        )

    next_slot = missing[0]
    return Objective(
        kind="collect_slot",
        slot=next_slot,
        instruction=descriptions.get(next_slot, f"Find out: {next_slot}"),
        missing_slots=missing,
    )
