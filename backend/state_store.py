"""
State store for a single call.

Design notes (per architecture discussion):
- Slots are editable, not write-once. Updating a slot keeps the old
  value + timestamp in `history` instead of silently discarding it.
- Slots carry a confidence score (as reported by the LLM tool call)
  so the dashboard can flag uncertain extractions for human review.
- This module knows NOTHING about HostelPGDekho, PGs, or any
  specific business. It's a generic bag of slots + terminal flags.
  Business meaning (which slots are required, what "done" means)
  lives in agent_config.py / objective.py, not here.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SlotChange:
    value: Any
    confidence: Optional[float]
    timestamp: float


@dataclass
class Slot:
    value: Any = None
    confidence: Optional[float] = None
    updated_at: Optional[float] = None
    history: list[SlotChange] = field(default_factory=list)

    def set(self, value: Any, confidence: Optional[float] = None) -> None:
        # Keep the previous value in history instead of discarding it —
        # lets a human reviewer see "timeline was X, corrected to Y".
        if self.value is not None:
            self.history.append(
                SlotChange(value=self.value, confidence=self.confidence, timestamp=self.updated_at or time.time())
            )
        self.value = value
        self.confidence = confidence
        self.updated_at = time.time()

    def is_filled(self) -> bool:
        return self.value is not None and self.value != ""

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "updated_at": self.updated_at,
            "history": [
                {"value": h.value, "confidence": h.confidence, "timestamp": h.timestamp} for h in self.history
            ],
        }


@dataclass
class CallState:
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phone_number: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    slots: dict[str, Slot] = field(default_factory=dict)

    # Terminal / control flags — set by tool calls, read by objective.py
    meeting_booked: bool = False
    meeting_time: Optional[str] = None
    callback_requested: bool = False
    not_interested: bool = False
    call_ended: bool = False
    was_interrupted_last_turn: bool = False

    # Pooja / HostelPGDekho specific outcome flags
    interested_confirmed: bool = False
    link_sent: bool = False  # "sent" meaning: URL was read out loud (no real WhatsApp send)

    # Turn history for prompt building (kept short — see build_prompt)
    turns: list[dict] = field(default_factory=list)  # [{"role": "user"/"assistant", "text": "..."}]

    def get_slot(self, name: str) -> Slot:
        if name not in self.slots:
            self.slots[name] = Slot()
        return self.slots[name]

    def update_slot(self, name: str, value: Any, confidence: Optional[float] = None) -> None:
        self.get_slot(name).set(value, confidence)

    def filled_slots(self) -> dict[str, Any]:
        return {name: slot.value for name, slot in self.slots.items() if slot.is_filled()}

    def missing_slots(self, required: list[str]) -> list[str]:
        return [name for name in required if not self.get_slot(name).is_filled()]

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text, "ts": time.time()})

    def recent_turns(self, n: int = 4) -> list[dict]:
        return self.turns[-n:]

    def is_terminal(self, terminal_conditions: list[str]) -> bool:
        flags = {
            "meeting_booked": self.meeting_booked,
            "not_interested": self.not_interested,
            "call_ended": self.call_ended,
            "interested_confirmed": self.interested_confirmed,
        }
        return any(flags.get(cond, False) for cond in terminal_conditions)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "phone_number": self.phone_number,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "slots": {name: slot.to_dict() for name, slot in self.slots.items()},
            "meeting_booked": self.meeting_booked,
            "meeting_time": self.meeting_time,
            "callback_requested": self.callback_requested,
            "not_interested": self.not_interested,
            "call_ended": self.call_ended,
            "interested_confirmed": self.interested_confirmed,
            "link_sent": self.link_sent,
            "turns": self.turns,
        }
