"""
Unit tests for the parts of the system that don't need any API keys:
state store, objective function, and ConversationManager business rules
for Pooja / HostelPGDekho.com. Run with:
    python -m pytest backend/tests/ -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversation_manager import ConversationManager
from objective import determine_next_objective
from state_store import CallState


class TestStateStore(unittest.TestCase):
    def test_slot_starts_empty(self):
        state = CallState()
        self.assertFalse(state.get_slot("permission_granted").is_filled())

    def test_slot_update_keeps_history(self):
        state = CallState()
        state.update_slot("interest_decision", "maybe", confidence=0.6)
        state.update_slot("interest_decision", "yes", confidence=0.9)  # owner changed their mind
        slot = state.get_slot("interest_decision")
        self.assertEqual(slot.value, "yes")
        self.assertEqual(len(slot.history), 1)
        self.assertEqual(slot.history[0].value, "maybe")

    def test_missing_slots(self):
        state = CallState()
        state.update_slot("permission_granted", "yes")
        missing = state.missing_slots(["permission_granted", "interest_decision"])
        self.assertEqual(missing, ["interest_decision"])


class TestObjective(unittest.TestCase):
    def test_first_objective_is_permission(self):
        state = CallState()
        obj = determine_next_objective(state)
        self.assertEqual(obj.kind, "collect_slot")
        self.assertEqual(obj.slot, "permission_granted")

    def test_objective_skips_filled_slots_out_of_order(self):
        """Owner volunteers interest before permission is explicitly
        confirmed — objective should still pick the first missing slot,
        order-independent."""
        state = CallState()
        state.update_slot("interest_decision", "yes")
        obj = determine_next_objective(state)
        self.assertEqual(obj.slot, "permission_granted")
        self.assertNotIn("interest_decision", obj.missing_slots)

    def test_wrap_up_when_all_slots_filled_but_not_terminal(self):
        state = CallState()
        for slot in [
            "permission_granted",
            "benefit_smart_search_explained",
            "benefit_vacant_form_explained",
            "benefit_free_listing_explained",
        ]:
            state.update_slot(slot, "yes")
        # interest_decision deliberately left unfilled with a non-decision value
        obj = determine_next_objective(state)
        self.assertEqual(obj.kind, "collect_slot")
        self.assertEqual(obj.slot, "interest_decision")

    def test_terminal_when_interested_confirmed(self):
        state = CallState()
        state.interested_confirmed = True
        obj = determine_next_objective(state)
        self.assertEqual(obj.kind, "end_call")


class TestConversationManager(unittest.TestCase):
    def test_update_state_tool(self):
        cm = ConversationManager(phone_number="local-demo")
        result = cm.apply_tool_call(
            "update_state", {"slot": "benefit_smart_search_explained", "value": "yes"}
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            cm.get_state().get_slot("benefit_smart_search_explained").value, "yes"
        )

    def test_interest_yes_sets_terminal_flag(self):
        cm = ConversationManager(phone_number="local-demo")
        result = cm.apply_tool_call("update_state", {"slot": "interest_decision", "value": "haan"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], "yes")  # normalized from Hindi
        self.assertTrue(cm.get_state().interested_confirmed)
        self.assertTrue(cm.get_state().call_ended)

    def test_interest_no_sets_not_interested(self):
        cm = ConversationManager(phone_number="local-demo")
        cm.apply_tool_call("update_state", {"slot": "interest_decision", "value": "nahi"})
        self.assertTrue(cm.get_state().not_interested)
        self.assertTrue(cm.get_state().call_ended)
        obj = determine_next_objective(cm.get_state())
        self.assertEqual(obj.kind, "end_call")

    def test_interest_maybe_ends_call_without_either_outcome_flag(self):
        cm = ConversationManager(phone_number="local-demo")
        cm.apply_tool_call("update_state", {"slot": "interest_decision", "value": "maybe later"})
        self.assertTrue(cm.get_state().call_ended)
        self.assertFalse(cm.get_state().interested_confirmed)
        self.assertFalse(cm.get_state().not_interested)

    def test_permission_denied_ends_call(self):
        cm = ConversationManager(phone_number="local-demo")
        cm.apply_tool_call("update_state", {"slot": "permission_granted", "value": "no"})
        self.assertTrue(cm.get_state().call_ended)
        self.assertFalse(cm.get_state().not_interested)

    def test_invalid_yes_no_value_rejected(self):
        cm = ConversationManager(phone_number="local-demo")
        result = cm.apply_tool_call(
            "update_state",
            {"slot": "interest_decision", "value": "I am currently renovating three floors"},
        )
        self.assertFalse(result["ok"])
        self.assertFalse(cm.get_state().get_slot("interest_decision").is_filled())

    def test_redundant_update_is_idempotent(self):
        cm = ConversationManager(phone_number="local-demo")
        cm.apply_tool_call("update_state", {"slot": "permission_granted", "value": "yes"})
        result = cm.apply_tool_call("update_state", {"slot": "permission_granted", "value": "yes"})
        self.assertTrue(result.get("redundant"))

    def test_unknown_tool_rejected(self):
        cm = ConversationManager(phone_number="local-demo")
        result = cm.apply_tool_call("delete_database", {})
        self.assertFalse(result["ok"])

    def test_placeholder_value_rejected(self):
        cm = ConversationManager(phone_number="local-demo")
        result = cm.apply_tool_call("update_state", {"slot": "owner_name", "value": "unknown"})
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
