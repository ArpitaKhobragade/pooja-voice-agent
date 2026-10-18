"""
MOCKED tests for Groq reliability: GroqLLM's single-bounded-fallback
policy and error categorization (local/llm_groq.py), and
PoojaSession/pooja_cli.py's handling of malformed tool calls and total
LLM failure (never crashing, never applying invalid state changes). No
real network calls, no API key needed.
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.llm_groq import GroqLLM, GroqAllModelsFailedError, GroqLLMError  # noqa: E402
from pooja_cli import PoojaSession, _parse_and_validate_tool_call  # noqa: E402


def _completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _api_error(status_code: int, msg: str = "error"):
    e = RuntimeError(msg)
    e.status_code = status_code
    return e


class TestGroqLLMFallbackPolicy(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_primary_success_never_touches_fallback(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _completion(content="hi")

        with patch("groq.Groq", return_value=fake_client):
            llm = GroqLLM(model="primary-model", fallback_model="fallback-model")
            msg = llm.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(msg.content, "hi")
        self.assertEqual(llm.model, "primary-model")
        self.assertEqual(fake_client.chat.completions.create.call_count, 1)

    def test_primary_fails_fallback_succeeds(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _api_error(404, "model_not_found"),
            _completion(content="fallback worked"),
        ]

        with patch("groq.Groq", return_value=fake_client):
            llm = GroqLLM(model="primary-model", fallback_model="fallback-model")
            msg = llm.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(msg.content, "fallback worked")
        self.assertEqual(llm.model, "fallback-model")
        self.assertEqual(fake_client.chat.completions.create.call_count, 2)

    def test_both_models_fail_raises_without_infinite_retry(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _api_error(404, "model_not_found"),
            _api_error(404, "model_not_found"),
        ]

        with patch("groq.Groq", return_value=fake_client):
            llm = GroqLLM(model="primary-model", fallback_model="fallback-model")
            with self.assertRaises(GroqAllModelsFailedError):
                llm.chat([{"role": "user", "content": "hi"}])

        # Exactly 2 calls — primary once, fallback once, no further retries.
        self.assertEqual(fake_client.chat.completions.create.call_count, 2)

    def test_failed_primary_does_not_stick_to_broken_fallback_on_next_call(self):
        """Regression: previously, once a call fell back, self._model was
        permanently overwritten to the fallback for the rest of the
        session — so if the fallback was itself broken (e.g. a retired
        model returning 404), EVERY subsequent turn crashed. Each chat()
        call must retry the primary fresh."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _api_error(404, "primary down"),   # turn 1: primary fails
            _completion(content="fallback 1"),  # turn 1: fallback succeeds
            _api_error(500, "primary down again"),  # turn 2: primary fails again
            _completion(content="fallback 2"),  # turn 2: fallback succeeds again
        ]

        with patch("groq.Groq", return_value=fake_client):
            llm = GroqLLM(model="primary-model", fallback_model="fallback-model")
            msg1 = llm.chat([{"role": "user", "content": "turn 1"}])
            msg2 = llm.chat([{"role": "user", "content": "turn 2"}])

        self.assertEqual(msg1.content, "fallback 1")
        self.assertEqual(msg2.content, "fallback 2")
        # Primary model string on the instance was never permanently
        # overwritten — it's still the configured primary.
        self.assertEqual(llm.primary_model, "primary-model")

    def test_no_distinct_fallback_configured_raises_immediately(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = _api_error(500, "boom")

        with patch("groq.Groq", return_value=fake_client):
            llm = GroqLLM(model="only-model", fallback_model="only-model")
            with self.assertRaises(GroqAllModelsFailedError):
                llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(fake_client.chat.completions.create.call_count, 1)

    def test_error_categorization_in_message(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _api_error(401, "bad key"),
            _api_error(401, "bad key"),
        ]
        with patch("groq.Groq", return_value=fake_client):
            llm = GroqLLM(model="m1", fallback_model="m2")
            with self.assertRaises(GroqAllModelsFailedError) as ctx:
                llm.chat([{"role": "user", "content": "hi"}])
        self.assertIn("authentication failed", str(ctx.exception))

    def test_missing_api_key_raises_clear_error(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaises(GroqLLMError):
                GroqLLM()

    def test_tool_choice_auto_only_when_tools_given(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _completion(content="hi")
        with patch("groq.Groq", return_value=fake_client):
            llm = GroqLLM(model="m1")
            llm.chat([{"role": "user", "content": "hi"}])  # no tools
            llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

        calls = fake_client.chat.completions.create.call_args_list
        self.assertNotIn("tools", calls[0].kwargs)
        self.assertEqual(calls[1].kwargs.get("tool_choice"), "auto")


def _tc(call_id="c1", name="update_state", arguments="{}"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class TestToolCallValidation(unittest.TestCase):
    def test_valid_call_passes(self):
        valid, args, err = _parse_and_validate_tool_call(
            _tc(arguments=json.dumps({"slot": "permission_granted", "value": "yes"}))
        )
        self.assertTrue(valid)
        self.assertEqual(args["slot"], "permission_granted")
        self.assertEqual(err, "")

    def test_malformed_json_rejected(self):
        valid, args, err = _parse_and_validate_tool_call(_tc(arguments="{slot: not valid json"))
        self.assertFalse(valid)
        self.assertIn("malformed JSON", err)

    def test_missing_fields_rejected(self):
        valid, args, err = _parse_and_validate_tool_call(_tc(arguments=json.dumps({"slot": "permission_granted"})))
        self.assertFalse(valid)
        self.assertIn("'value'", err)

    def test_unknown_slot_rejected(self):
        valid, args, err = _parse_and_validate_tool_call(
            _tc(arguments=json.dumps({"slot": "hallucinated_slot_name", "value": "yes"}))
        )
        self.assertFalse(valid)
        self.assertIn("slot", err)

    def test_non_object_arguments_rejected(self):
        valid, args, err = _parse_and_validate_tool_call(_tc(arguments=json.dumps(["not", "an", "object"])))
        self.assertFalse(valid)

    def test_unknown_tool_name_rejected(self):
        valid, args, err = _parse_and_validate_tool_call(
            _tc(name="delete_everything", arguments=json.dumps({"slot": "permission_granted", "value": "yes"}))
        )
        self.assertFalse(valid)
        self.assertIn("unknown tool", err)


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._responses.pop(0)


class FailingLLM:
    """Always raises GroqAllModelsFailedError, simulating both Groq
    models being unavailable for the entire session."""

    def chat(self, messages, tools=None):
        raise GroqAllModelsFailedError("both models down (simulated)")


class TestMalformedToolCallsNeverCorruptState(unittest.TestCase):
    def test_malformed_tool_call_does_not_apply_and_session_continues(self):
        llm = ScriptedLLM(
            [
                SimpleNamespace(content="Hello Sir...", tool_calls=None),  # opening
                SimpleNamespace(
                    content=None,
                    tool_calls=[_tc(arguments="{not valid json")],
                ),
                SimpleNamespace(content="Let me ask that differently.", tool_calls=None),
            ]
        )
        inputs = iter(["some reply", "bye"])
        session = PoojaSession(
            llm=llm,
            get_user_input=lambda: next(inputs, "bye"),
            deliver_reply=lambda text: False,
        )
        cm = session.run()

        # No slot was ever filled — the malformed call was rejected, not applied.
        self.assertFalse(cm.get_state().get_slot("permission_granted").is_filled())
        # The session kept going and ended cleanly via the deterministic "bye" path.
        self.assertTrue(cm.get_state().call_ended)

    def test_unknown_slot_tool_call_rejected_without_crashing(self):
        llm = ScriptedLLM(
            [
                SimpleNamespace(content="Hello Sir...", tool_calls=None),
                SimpleNamespace(
                    content=None,
                    tool_calls=[_tc(arguments=json.dumps({"slot": "made_up_slot", "value": "x"}))],
                ),
                SimpleNamespace(content="Okay, noted.", tool_calls=None),
            ]
        )
        inputs = iter(["some reply", "bye"])
        session = PoojaSession(
            llm=llm,
            get_user_input=lambda: next(inputs, "bye"),
            deliver_reply=lambda text: False,
        )
        cm = session.run()
        self.assertTrue(cm.get_state().call_ended)


class TestGroqTotalFailureDoesNotCrashSession(unittest.TestCase):
    def test_opening_failure_ends_gracefully(self):
        session = PoojaSession(
            llm=FailingLLM(),
            get_user_input=lambda: "should never be called",
            deliver_reply=lambda text: False,
        )
        cm = session.run()  # must not raise
        self.assertTrue(cm.get_state().call_ended)

    def test_mid_conversation_failure_ends_gracefully(self):
        class OnceThenFailLLM:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content="Hello Sir...", tool_calls=None)
                raise GroqAllModelsFailedError("down now")

        inputs = iter(["Haan ji bataiye"])
        session = PoojaSession(
            llm=OnceThenFailLLM(),
            get_user_input=lambda: next(inputs, "bye"),
            deliver_reply=lambda text: False,
        )
        cm = session.run()  # must not raise
        self.assertTrue(cm.get_state().call_ended)


if __name__ == "__main__":
    unittest.main()
