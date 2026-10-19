"""
Tests for PoojaSession's LLM tool-calling loop (pooja_cli.py), using a
fake LLM so no network call or API key is needed. This exercises the
exact message-shape plumbing (tool_calls -> ConversationManager ->
tool-result messages -> next completion) that the real Groq-backed voice
and text modes depend on.
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pooja_cli import PoojaSession  # noqa: E402


def _tool_call(call_id: str, slot: str, value: str):
    import json

    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name="update_state", arguments=json.dumps({"slot": slot, "value": value})),
    )


def _message(content: str | None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class ScriptedLLM:
    """Replays a fixed queue of responses, ignoring the actual prompt
    content (we're testing the plumbing, not real model behaviour)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._responses.pop(0)


class TestPoojaSessionToolLoop(unittest.TestCase):
    def test_tool_call_updates_state_and_reaches_terminal(self):
        llm = ScriptedLLM(
            [
                _message("Hello ji, good afternoon..."),  # opening
                _message(None, [_tool_call("c1", "permission_granted", "yes")]),
                _message("Great, let me tell you about our features..."),
                _message(None, [_tool_call("c2", "interest_decision", "haan")]),
                _message("Thank you, have a great day!"),
            ]
        )

        inputs = iter(["Haan ji bataiye", "Haan interested hoon"])
        outputs = []

        session = PoojaSession(
            llm=llm,
            get_user_input=lambda: next(inputs, "bye"),
            deliver_reply=lambda text: outputs.append(text),
        )
        cm = session.run()

        self.assertTrue(cm.get_state().interested_confirmed)
        self.assertTrue(cm.get_state().call_ended)
        self.assertEqual(outputs[0], "Hello ji, good afternoon...")
        self.assertIn("Thank you, have a great day!", outputs)
        # Tool-result messages must be well-formed for the Groq API shape.
        tool_messages = [m for m in session.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertEqual(tool_messages[0]["tool_call_id"], "c1")

    def test_stop_phrase_ends_call_without_llm_tool_loop_hanging(self):
        llm = ScriptedLLM(
            [
                _message("Hello ji..."),
                _message("Okay, thank you, goodbye!"),
            ]
        )
        session = PoojaSession(
            llm=llm,
            get_user_input=lambda: "band karo",
            deliver_reply=lambda text: None,
        )
        cm = session.run()
        self.assertTrue(cm.get_state().call_ended)

    def test_duplicate_reply_not_spoken_twice(self):
        llm = ScriptedLLM(
            [
                _message("Hello ji..."),
                _message("Hello ji..."),  # exact duplicate of opening
            ]
        )
        inputs = iter(["bye"])
        outputs = []
        session = PoojaSession(
            llm=llm,
            get_user_input=lambda: next(inputs, "bye"),
            deliver_reply=lambda text: outputs.append(text),
        )
        session.run()
        # Only spoken once even though the model produced it twice.
        self.assertEqual(outputs.count("Hello ji..."), 1)

    def test_repeat_phrase_resays_last_line_without_extra_llm_call(self):
        llm = ScriptedLLM(
            [
                _message("Hello ji, good afternoon..."),  # opening
            ]
        )
        inputs = iter(["aap repeat kar sakte ho", "bye"])
        outputs = []
        session = PoojaSession(
            llm=llm,
            get_user_input=lambda: next(inputs, "bye"),
            deliver_reply=lambda text: outputs.append(text) or False,
        )
        session.run()
        # "repeat" costs no LLM call, and "bye" is handled deterministically
        # (never routed through the LLM — see STOP_PHRASES handling) so
        # only the opening ever calls the model.
        self.assertEqual(llm.calls, 1)
        self.assertEqual(outputs[0], "Hello ji, good afternoon...")
        self.assertEqual(outputs[1], "Hello ji, good afternoon...")  # repeated verbatim
        self.assertTrue(session.cm.state.call_ended)

    def test_barge_in_sets_interrupted_flag_for_next_turn(self):
        llm = ScriptedLLM(
            [
                _message("Hello ji, good afternoon..."),
                _message("Continuing after interruption..."),
            ]
        )
        inputs = iter(["bye"])
        session = PoojaSession(
            llm=llm,
            get_user_input=lambda: next(inputs, "bye"),
            deliver_reply=lambda text: True,  # simulate every reply being barged-in on
        )
        session.run()
        # was_interrupted_last_turn should reflect the most recent speak()
        # (the closing message, which this test simulates as barged-in on).
        self.assertTrue(session.cm.state.was_interrupted_last_turn)

    def test_max_turns_safety_net_ends_call(self):
        # Every turn: user reply, model responds with no tool call and a
        # never-progressing message — must hit MAX_TURNS and stop, not loop forever.
        responses = [_message("Hello ji...")]
        for _ in range(60):
            responses.append(_message("Still thinking about it..."))
        llm = ScriptedLLM(responses)

        session = PoojaSession(
            llm=llm,
            get_user_input=lambda: "hmm not sure yet",
            deliver_reply=lambda text: None,
        )
        cm = session.run()
        self.assertTrue(cm.get_state().call_ended)


if __name__ == "__main__":
    unittest.main()
