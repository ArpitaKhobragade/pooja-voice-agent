"""
ConversationManager

Owns exactly four responsibilities (per architecture discussion):
  1. Maintain conversation state (delegates to CallState)
  2. Validate tool calls Gemini proposes
  3. Update slots
  4. Track conversation progress (terminal flags)

It does NOT know it's talking about PGs, HostelPGDekho, or vacant beds —
that knowledge lives in agent_config.py and prompts.py. This class would
be reused unchanged for a different outbound sales agent.

Gemini proposes actions via tool calls; this class decides what
actually happens. That separation is intentional — a misheard word
should never be able to single-handedly "confirm" a meeting; it has
to pass through here first.
"""

from __future__ import annotations

from typing import Any

from state_store import CallState
from objective import determine_next_objective, Objective

YES_NO_SLOTS = {
    "permission_granted",
    "interest_decision",
    "link_requested",
    "callback_requested",
}

BENEFIT_EXPLAINED_SLOTS = {
    "benefit_smart_search_explained",
    "benefit_vacant_form_explained",
    "benefit_free_listing_explained",
}

# Loose but bilingual (Hindi/English/Hinglish) yes/no/maybe vocabulary, so a
# plausible normalized answer from the LLM is accepted without forcing it
# into a brittle exact-match enum.
_YES_WORDS = {"yes", "haan", "ha", "sure", "ok", "okay", "theek hai", "bilkul", "achha", "accha"}
_NO_WORDS = {"no", "nahi", "nahin", "not interested", "never"}
_MAYBE_WORDS = {"maybe", "later", "maybe_later", "shayad", "baad me", "soch kar batata hoon", "busy"}


def is_valid_yes_no_maybe(value: str) -> bool:
    """Reject anything that isn't recognizably yes/no/maybe in English,
    Hindi, or Hinglish — prevents the LLM from writing a paragraph into a
    slot meant to hold a single decision."""
    val = value.strip().lower()
    if not val:
        return False
    vocab = _YES_WORDS | _NO_WORDS | _MAYBE_WORDS
    if val in vocab:
        return True
    # Allow short phrases built from the vocabulary (e.g. "haan bilkul")
    # but reject long free-text answers that likely aren't a clean decision.
    if len(val.split()) <= 4 and any(word in val for word in vocab):
        return True
    return False


def normalize_yes_no_maybe(value: str) -> str:
    val = value.strip().lower()
    if any(word in val for word in _NO_WORDS):
        return "no"
    if any(word in val for word in _MAYBE_WORDS):
        return "maybe"
    if any(word in val for word in _YES_WORDS):
        return "yes"
    return val


class ConversationManager:
    def __init__(self, phone_number: str):
        self.state = CallState(phone_number=phone_number)

    # ---- tool call entry point -------------------------------------------------

    def apply_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Validate + apply a tool call Gemini proposed. Returns a small result
        dict that gets fed back to the LLM as the tool result."""
        handler = getattr(self, f"_handle_{name}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown tool '{name}'"}
        return handler(args)

    # ---- individual tool handlers -----------------------------------------------

    def _handle_update_state(self, args: dict[str, Any]) -> dict[str, Any]:
        slot = args.get("slot")
        value = args.get("value")
        if isinstance(value, str) and value.strip().lower() in {
            "unknown",
            "unsure",
            "none",
            "null",
            "",
        }:
            return {
                "ok": False,
                "error": "Only record information explicitly provided by the caller."
            }
            
        if slot in YES_NO_SLOTS and isinstance(value, str):
            if not is_valid_yes_no_maybe(value):
                return {
                    "ok": False,
                    "error": f"'{value}' is not a clear yes/no/maybe answer for '{slot}'. Ask a clarifying question instead of guessing."
                }
            value = normalize_yes_no_maybe(value)

        if slot in BENEFIT_EXPLAINED_SLOTS and isinstance(value, str):
            # These slots only ever mean "I just explained this benefit" —
            # normalize anything truthy to "yes" so the state is clean.
            value = "yes"

        existing = self.state.get_slot(slot)
        if existing.is_filled():
            # Idempotent: ignore updates to slots that are already filled.
            return {"ok": True, "redundant": True, "slot": slot, "value": value}

        self.state.update_slot(slot, value)

        # Determine terminal transitions automatically
        self.evaluate_state()

        return {"ok": True, "slot": slot, "value": value}

    def evaluate_state(self):
        """Evaluate if the current state transitions the call to a terminal state."""
        permission = self.state.get_slot("permission_granted").value
        interest = self.state.get_slot("interest_decision").value
        link_requested = self.state.get_slot("link_requested").value

        # 1. Owner confirmed interest in using the platform.
        if interest == "yes":
            if not self.state.interested_confirmed:
                self.state.interested_confirmed = True
                self.state.call_ended = True

        # 2. Owner explicitly not interested.
        elif interest == "no":
            if not self.state.not_interested:
                self.state.not_interested = True
                self.state.call_ended = True

        # 3. Owner said "maybe later" — non-committal, still a clean close
        #    (not a positive or negative outcome) so the call doesn't stall.
        elif interest == "maybe" and not self.state.call_ended:
            self.state.call_ended = True

        # 4. Owner couldn't talk right now / declined to continue before
        #    the pitch even started.
        elif permission == "no" and not self.state.call_ended:
            self.state.call_ended = True

        if link_requested == "yes":
            # Marks that the owner asked — the actual "link_sent" flag is
            # set once Pooja has read the URL out loud (tracked separately
            # so she doesn't repeat it every turn).
            pass

    def get_state(self) -> CallState:
        return self.state
        
    def get_next_objective(self) -> Objective:
        return determine_next_objective(self.state)
