"""
Automated coverage for the fully offline, no-API-key demo
(local/offline_script.py) — drives the keyword-driven script through
every major objection scenario from the approved HostelPGDekho script,
feeding scripted "owner" replies via a monkeypatched input() instead of
a real keyboard, so this runs in CI with zero network calls.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.offline_script import run_offline_demo  # noqa: E402


def _run_with_replies(replies: list[str]):
    it = iter(replies)
    with patch("builtins.input", lambda *a, **k: next(it, "bye")), \
         patch("builtins.print"):  # keep test output quiet
        return run_offline_demo()


class TestOfflineScript(unittest.TestCase):
    def test_full_happy_path_through_all_benefits_to_interested(self):
        cm = _run_with_replies(
            ["Haan ji, bataiye", "theek hai", "theek hai", "theek hai", "Haan interested hoon"]
        )
        state = cm.get_state()
        self.assertEqual(state.get_slot("permission_granted").value, "yes")
        self.assertEqual(state.get_slot("benefit_smart_search_explained").value, "yes")
        self.assertEqual(state.get_slot("benefit_vacant_form_explained").value, "yes")
        self.assertEqual(state.get_slot("benefit_free_listing_explained").value, "yes")
        self.assertTrue(state.interested_confirmed)
        self.assertTrue(state.call_ended)

    def test_busy_then_declines_reschedule_ends_politely(self):
        cm = _run_with_replies(["Abhi time nahi hai", "Nahi, rehne do"])
        state = cm.get_state()
        self.assertTrue(state.call_ended)
        self.assertFalse(state.interested_confirmed)
        self.assertFalse(state.not_interested)  # declined to talk, not a product refusal

    def test_busy_then_accepts_reschedule_continues(self):
        cm = _run_with_replies(
            ["Abhi time nahi hai", "Haan, theek hai", "theek hai", "theek hai", "Nahi"]
        )
        state = cm.get_state()
        self.assertEqual(state.get_slot("permission_granted").value, "yes")

    def test_not_interested_mid_script_stops_immediately(self):
        cm = _run_with_replies(["yes", "Mujhe interest nahi hai"])
        state = cm.get_state()
        self.assertTrue(state.not_interested)
        self.assertTrue(state.call_ended)
        # Should not have reached the free-listing benefit — stopped early.
        self.assertFalse(state.get_slot("benefit_free_listing_explained").is_filled())

    def test_link_request_sets_flag_and_never_claims_message_sent(self):
        cm = _run_with_replies(["yes", "link bhej do", "theek hai", "Haan"])
        state = cm.get_state()
        self.assertEqual(state.get_slot("link_requested").value, "yes")
        self.assertTrue(state.link_sent)
        # The transcript must contain the honest WhatsApp-limitation line,
        # never a false claim that a message was sent.
        spoken = " ".join(t["text"] for t in state.turns if t["role"] == "assistant")
        self.assertIn("local demo", spoken.lower())
        self.assertNotIn("bhej diya", spoken.lower())  # "sent" (past tense claim) must not appear

    def test_stop_keyword_ends_immediately(self):
        cm = _run_with_replies(["yes", "band karo"])
        state = cm.get_state()
        self.assertTrue(state.call_ended)


if __name__ == "__main__":
    unittest.main()
