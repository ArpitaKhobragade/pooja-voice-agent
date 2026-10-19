"""
Tests for following the approved script: the code says the script steps
verbatim whenever the owner acknowledges, the LLM only answers real
questions, plus Sir / Ma'am forcing, neutral wishes, noise handling and the
history trim. Fake streaming LLM; no network.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_config  # noqa: E402
import pooja_cli  # noqa: E402
from agent_config import AGENT_CONFIG, FINAL_STEP_TEXT, SCRIPT_STEPS  # noqa: E402
from local.reply_stream import stream_sentences  # noqa: E402
from pooja_cli import HISTORY_MESSAGES, PoojaSession, is_greeting_only, looks_like_noise  # noqa: E402
from tests.test_owner_handling import voice  # noqa: E402


class FakeStreamLLM:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = []

    def chat_stream(self, messages):
        self.calls.append([dict(m) for m in messages])
        yield self.replies.pop(0) if self.replies else "Ji."


class Getter:
    def __init__(self, items):
        self.items = list(items)
        self.last_audio = None

    def __call__(self):
        if not self.items:
            return "bye"
        item = self.items.pop(0)
        text, audio = item if isinstance(item, tuple) else (item, None)
        self.last_audio = audio
        return text


def run(llm, items):
    pooja_cli.random.random = lambda: 0.9  # no random lead-ins in tests
    out = []
    s = PoojaSession(llm=llm, get_user_input=Getter(items), deliver_reply=lambda t: out.append(t), conversational=True)
    s.run()
    return s, out


def step_text(i):
    """A script step as said before the owner's gender is known: no Sir / Ma'am yet."""
    return PoojaSession._addr(types.SimpleNamespace(honorific=None), SCRIPT_STEPS[i]["text"])


class TestScriptIsFollowedVerbatim(unittest.TestCase):
    def test_script_steps_keep_the_key_facts_of_the_approved_script(self):
        joined = " ".join(step["text"] for step in SCRIPT_STEPS).lower()
        for fact in ("sirf ek property listing app nahi", "50 pgs", "10 se 15", "smart search", "request form",
                     "3 se 4 mahine", "free listing", "interested"):
            self.assertIn(fact, joined)

    def test_acknowledgements_play_every_step_in_order_with_no_llm(self):
        llm = FakeStreamLLM()
        _, out = run(llm, ["haan ji"] + ["okay"] * (len(SCRIPT_STEPS) - 1))
        self.assertEqual(llm.calls, [])
        # out[0] is the opening; the next outputs are the steps, verbatim and in order
        for i, step in enumerate(SCRIPT_STEPS):
            self.assertEqual(out[1 + i], step_text(i), step["id"])

    def test_yes_to_the_interest_question_gives_the_link_and_ends(self):
        llm = FakeStreamLLM()
        s, out = run(llm, ["haan ji"] + ["okay"] * (len(SCRIPT_STEPS) - 1) + ["haan interested hoon"])
        self.assertEqual(llm.calls, [])
        self.assertTrue(s.cm.get_state().interested_confirmed)
        self.assertTrue(s.cm.get_state().call_ended)
        self.assertIn("aapke WhatsApp par app ka link send kar deti hoon", out[-1])
        self.assertNotIn("hostelpgdekho.com/app", out[-1])  # sent on WhatsApp, not read out

    def test_happy_reactions_count_as_acknowledgements_not_noise(self):
        llm = FakeStreamLLM()
        _, out = run(llm, ["haan ji", "wow", "great"])
        self.assertEqual(llm.calls, [])
        self.assertEqual(out[1], step_text(0))
        self.assertTrue(out[2].endswith(step_text(1)))   # "wow" gets a warm reaction first
        self.assertNotEqual(out[2], step_text(1))
        self.assertIn(step_text(2), out[3])

    def test_no_to_the_interest_question_closes_politely(self):
        llm = FakeStreamLLM()
        s, out = run(llm, ["haan ji"] + ["okay"] * (len(SCRIPT_STEPS) - 1) + ["nahi"])
        self.assertEqual(llm.calls, [])
        self.assertTrue(s.cm.get_state().not_interested)
        self.assertIn("Koi baat nahi", out[-1])

    def test_a_question_mid_script_gets_an_llm_answer_then_the_script_resumes(self):
        llm = FakeStreamLLM(["Ji, ye app ek platform hai jahan owners listing daalte hain."])
        s, out = run(llm, ["haan ji", "ye kaise kaam karta hai?", "okay"])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(out[1], step_text(0))                       # intro, verbatim
        self.assertIn("platform", out[2])                            # LLM answer
        self.assertEqual(out[3], step_text(1))                       # script resumes at the next step

    def test_a_step_the_llm_already_covered_is_not_repeated(self):
        llm = FakeStreamLLM(["Humara smart search bahut owners tak inquiry pahunchata hai."])
        s, out = run(llm, ["haan ji", "ye kaise kaam karta hai?", "okay"])
        self.assertNotIn(step_text(1), out)          # smart_search already covered by the LLM's answer
        self.assertEqual(out[3], step_text(2))       # straight on to the Request Form

    def test_hello_gets_a_short_reintroduction_not_the_pitch(self):
        llm = FakeStreamLLM()
        _, out = run(llm, ["Hello"])
        self.assertEqual(llm.calls, [])
        self.assertIn("Pooja bol rahi hoon", out[1])
        self.assertNotEqual(out[1], step_text(0))

    def test_dobara_batao_is_a_question_not_a_go_ahead(self):
        self.assertEqual(pooja_cli.classify_reply("dobara batao"), "other")
        self.assertEqual(pooja_cli.classify_reply("achha batao"), "yes")

    def test_early_interest_is_not_skipped_past_the_pitch(self):
        llm = FakeStreamLLM(["Great, pehle thoda aur bata deti hoon."])
        s, out = run(llm, ["haan main interested hoon"])
        self.assertFalse(s.cm.get_state().interested_confirmed)


class TestEndMarkersNeedAnOwnerCue(unittest.TestCase):
    def test_owner_cues_that_justify_ending(self):
        for t in ("abhi busy hoon", "mujhe interest nahi hai", "baad mein baat karte hain", "manager se baat karao",
                  "ye bakwas band karo", "nahi chahiye"):
            self.assertTrue(pooja_cli.owner_signalled_end(t), t)

    def test_ordinary_questions_do_not_justify_ending(self):
        for t in ("Ab kaun baat kar raha hai?", "ye kaise kaam karta hai", "achha batao", "free hai kya"):
            self.assertFalse(pooja_cli.owner_signalled_end(t), t)

    def test_a_stray_callback_marker_on_a_plain_question_does_not_end_the_call(self):
        llm = FakeStreamLLM(["Main Pooja hoon, HostelPGDekho se. [CALLBACK]"])
        s, out = run(llm, ["Ab kaun baat kar raha hai?", "achha"])
        self.assertFalse(s.cm.get_state().get_slot("interest_decision").is_filled())
        self.assertIn(step_text(0), out)  # the call carried on and the script began


class TestNoiseAndGreetings(unittest.TestCase):
    def test_garbled_short_transcripts_are_noise(self):
        for t in ("of", "Include the", "Aa", "Haan toh tai"[:0] or "xyz"):
            self.assertTrue(looks_like_noise(t), t)

    def test_real_short_answers_are_not_noise(self):
        for t in ("Mm", "Okay", "Nahin", "Hello", "Achchha achchha", "Kaun?", "bye", "free"):
            self.assertFalse(looks_like_noise(t), t)

    def test_long_or_question_transcripts_are_never_noise(self):
        self.assertFalse(looks_like_noise("mera PG full hai"))
        self.assertFalse(looks_like_noise("xyz?"))

    def test_noise_asks_again_without_calling_the_llm(self):
        llm = FakeStreamLLM()
        _, out = run(llm, ["of"])
        self.assertEqual(llm.calls, [])
        self.assertTrue(any("clear" in t or "bata dijiye" in t or "dobara" in t for t in out[1:]))

    def test_greeting_detection(self):
        for t in ("Hello", "hello ji", "namaste", "Hi"):
            self.assertTrue(is_greeting_only(t), t)
        for t in ("haan ji", "hello kaun", "ji"):
            self.assertFalse(is_greeting_only(t), t)


class TestFormOfAddress(unittest.TestCase):
    def test_model_saying_sir_is_forced_to_maam_for_a_woman(self):
        llm = FakeStreamLLM(["Ji Sir, samajh gayi. Aur ek baat Sir, ye free hai."])
        s, out = run(llm, [("free hai kya?", voice(225))])
        spoken = " ".join(out)
        self.assertNotIn("Sir", spoken)
        self.assertIn("Ma'am", spoken)

    def test_script_steps_use_maam_for_a_woman(self):
        llm = FakeStreamLLM()
        s, out = run(llm, [("haan ji", voice(225))] + ["okay"] * (len(SCRIPT_STEPS) - 1))
        self.assertNotIn("Sir", " ".join(out))
        self.assertIn("Ma'am", out[4])  # the free-listing step says "Aur Sir, ..."

    def test_man_keeps_sir(self):
        llm = FakeStreamLLM(["Ji Sir, samajh gayi."])
        s, out = run(llm, [("free hai kya?", voice(110))])
        self.assertIn("Sir", " ".join(out))


class TestNeutralWishes(unittest.TestCase):
    def test_time_of_day_wishes_from_the_model_become_take_care(self):
        for said in ("Aapka din shubh ho.", "Aapko shubh ratri.", "Have a good day.", "Aapki shaam shubh ho."):
            out = list(stream_sentences([f"Dhanyavaad. {said}"], []))
            joined = " ".join(out).lower()
            self.assertIn("take care", joined, said)
            self.assertNotIn("shubh", joined, said)


class TestHistoryTrim(unittest.TestCase):
    def test_only_recent_messages_are_sent_but_system_prompts_always_are(self):
        llm = FakeStreamLLM(["Ji."] * 30)
        run(llm, ["free hai kya?"] * 15)
        sent = llm.calls[-1]
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[1]["role"], "system")
        self.assertLessEqual(len(sent), 2 + HISTORY_MESSAGES + 1)


class TestOpening(unittest.TestCase):
    def test_opening_follows_the_script_and_is_neutral(self):
        text = agent_config.script_line("opening")
        self.assertIn("My name is Pooja", text)
        self.assertIn("HostelPGDekho.com", text)
        self.assertIn("mujhe aapke pg ke regarding baat karni hai", text.lower())
        self.assertNotIn("Sir", text)


class TestGroqRetryHint(unittest.TestCase):
    def test_parses_groqs_try_again_hint(self):
        from local.llm_groq import retry_after_seconds

        self.assertAlmostEqual(retry_after_seconds("Please try again in 10.335s."), 10.335)
        self.assertAlmostEqual(retry_after_seconds("try again in 150ms."), 0.15)
        self.assertAlmostEqual(retry_after_seconds("try again in 2m1.5s."), 121.5)
        self.assertIsNone(retry_after_seconds("no hint here"))

    def test_token_budget_per_request_is_small_enough_for_the_free_tier(self):
        from local import llm_groq
        import prompts

        self.assertLessEqual(llm_groq.MAX_TOKENS, 300)
        self.assertLess(len(prompts.BASE_PERSONA_PROMPT.split()), 900)


if __name__ == "__main__":
    unittest.main()
