"""
Fully offline, no-API-key text demo of Pooja's call flow.

No LLM, no STT, no TTS — this is a deterministic, keyword-driven
walkthrough of the approved HostelPGDekho script using the literal script
lines in agent_config.py, with the SAME ConversationManager/state-machine
core used by the voice and text-LLM modes. It exists so the project can be
demonstrated and graded with zero API credits spent and zero network
calls, per the cost-optimization requirement.

Because there's no LLM to interpret free-form speech, objection handling
here is keyword-based (English/Hindi/Hinglish) rather than true language
understanding — documented as a known limitation, not claimed to be as
flexible as the LLM-backed modes.
"""

from __future__ import annotations

from agent_config import AGENT_CONFIG, script_line
from conversation_manager import ConversationManager

_NO_KEYWORDS = ("no", "nahi", "nahin", "not interested", "band karo", "stop", "bye", "exit", "quit")
_BUSY_KEYWORDS = ("busy", "time nahi", "abhi nahi", "later", "baad me")
_YES_KEYWORDS = ("yes", "haan", "ha ji", "ji haan", "sure", "theek hai", "ok", "okay")
_LINK_KEYWORDS = ("link", "whatsapp", "url", "bhej do", "bhejo")
_STOP_KEYWORDS = ("bye", "band karo", "stop", "exit", "quit", "rakhta hoon", "rakhti hoon")


def _matches(text: str, keywords: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(kw in low for kw in keywords)


def _say(line_key: str) -> None:
    print(f"\nPooja: {script_line(line_key)}")


def _ask(prompt: str = "You (owner)> ") -> str:
    try:
        return input(prompt)
    except EOFError:
        return "bye"


def run_offline_demo(max_turns: int = 20) -> ConversationManager:
    """Runs the keyword-driven offline demo on the keyboard. Returns the
    ConversationManager at the end so callers (or tests) can inspect the
    final state."""
    cm = ConversationManager(phone_number="offline-demo")
    state = cm.get_state()

    print("=" * 60)
    print("Pooja — HostelPGDekho.com (OFFLINE TEXT DEMO, no API keys used)")
    print("Type your reply as the PG owner. Type 'bye' to end anytime.")
    print("=" * 60)

    _say("opening")
    reply = _ask()
    state.add_turn("assistant", script_line("opening"))
    state.add_turn("user", reply)

    # Check "busy" before "no": a phrase like "abhi time nahi hai" contains
    # "nahi" (matches _NO_KEYWORDS) but means "busy right now", not an
    # outright refusal — it deserves the reschedule-ask, not an immediate
    # decline.
    if _matches(reply, _BUSY_KEYWORDS):
        _say("ask_reschedule")
        reply2 = _ask()
        state.add_turn("assistant", script_line("ask_reschedule"))
        state.add_turn("user", reply2)
        if not _matches(reply2, _YES_KEYWORDS):
            cm.apply_tool_call("update_state", {"slot": "permission_granted", "value": "no"})
            _say("closing_busy_or_declined")
            return cm
    elif _matches(reply, _STOP_KEYWORDS) or _matches(reply, _NO_KEYWORDS):
        cm.apply_tool_call("update_state", {"slot": "permission_granted", "value": "no"})
        _say("closing_busy_or_declined")
        return cm

    cm.apply_tool_call("update_state", {"slot": "permission_granted", "value": "yes"})

    # Each entry is one back-and-forth: the beats Pooja says, then the
    # slots those beats cover, then the owner replies.
    benefit_order = [
        (["beat_1", "beat_2"], ["benefit_smart_search_explained"]),
        (["beat_3", "beat_4"], ["benefit_vacant_form_explained", "benefit_free_listing_explained"]),
    ]
    benefit_index = 0
    turns = 0

    while benefit_index < len(benefit_order) and turns < max_turns:
        turns += 1
        line_keys, slots = benefit_order[benefit_index]
        for line_key in line_keys:
            _say(line_key)
            state.add_turn("assistant", script_line(line_key))
        for slot in slots:
            cm.apply_tool_call("update_state", {"slot": slot, "value": "yes"})

        reply = _ask()
        state.add_turn("user", reply)

        if _matches(reply, _STOP_KEYWORDS) or _matches(reply, _NO_KEYWORDS):
            cm.apply_tool_call("update_state", {"slot": "interest_decision", "value": "no"})
            break
        if _matches(reply, _LINK_KEYWORDS):
            cm.apply_tool_call("update_state", {"slot": "link_requested", "value": "yes"})
            _say("link_offer_demo_limitation")
            state.add_turn("assistant", script_line("link_offer_demo_limitation"))
            state.link_sent = True

        benefit_index += 1

    if not state.call_ended:
        _say("ask_interest")
        state.add_turn("assistant", script_line("ask_interest"))
        reply = _ask()
        state.add_turn("user", reply)

        if _matches(reply, _YES_KEYWORDS):
            cm.apply_tool_call("update_state", {"slot": "interest_decision", "value": "yes"})
        elif _matches(reply, _NO_KEYWORDS):
            cm.apply_tool_call("update_state", {"slot": "interest_decision", "value": "no"})
        else:
            cm.apply_tool_call("update_state", {"slot": "interest_decision", "value": "maybe"})

        if _matches(reply, _LINK_KEYWORDS) and not state.link_sent:
            cm.apply_tool_call("update_state", {"slot": "link_requested", "value": "yes"})
            _say("link_offer_demo_limitation")
            state.link_sent = True

    if state.interested_confirmed:
        _say("closing_interested")
    elif state.not_interested:
        _say("closing_not_interested")
    else:
        _say("closing_busy_or_declined")

    return cm


if __name__ == "__main__":
    run_offline_demo()
