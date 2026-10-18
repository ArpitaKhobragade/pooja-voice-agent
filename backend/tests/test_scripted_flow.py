"""
Tests for the conversational (streamed) path, the sentence streamer, the
reply->state logic, and the TTS pronunciation helper. Fake streaming LLMs
are used, so no network or API key is needed.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.llm_groq import GroqAllModelsFailedError  # noqa: E402
from local.reply_stream import reply_effects, stream_sentences  # noqa: E402
from local.speech_text import speech_text  # noqa: E402
from pooja_cli import PoojaSession, classify_reply  # noqa: E402


class FakeStreamLLM:
    """Streams the queued replies in small chunks, like a real stream."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_messages = None

    def chat_stream(self, messages):
        self.calls += 1
        self.last_messages = [dict(m) for m in messages]
        if not self.replies:
            raise AssertionError("FakeStreamLLM ran out of replies")
        reply = self.replies.pop(0)
        for i in range(0, len(reply), 5):
            yield reply[i:i + 5]


class DownLLM:
    def chat_stream(self, messages):
        raise GroqAllModelsFailedError("down")
        yield  # pragma: no cover


def _run(llm, inputs):
    import pooja_cli
    pooja_cli.random.random = lambda: 0.9  # no random lead-ins in tests
    it = iter(inputs)
    out = []
    session = PoojaSession(
        llm=llm,
        get_user_input=lambda: next(it, "bye"),
        deliver_reply=lambda t: out.append(t),
        conversational=True,
    )
    return session, session.run(), out


class TestSpeechText(unittest.TestCase):
    def test_brand_is_spaced_into_words(self):
        from local.speech_text import BRAND_SPOKEN, DOTCOM_SPOKEN

        self.assertEqual(speech_text("HostelPGDekho.com se"), f"{BRAND_SPOKEN} {DOTCOM_SPOKEN} se")
        self.assertEqual(speech_text("HostelPGDekho sirf app"), f"{BRAND_SPOKEN} sirf app")

    def test_url_is_readable(self):
        from local.speech_text import BRAND_SPOKEN, DOTCOM_SPOKEN, SLASH_SPOKEN

        self.assertEqual(
            speech_text("https://hostelpgdekho.com/app"), f"{BRAND_SPOKEN} {DOTCOM_SPOKEN} {SLASH_SPOKEN} app"
        )


class TestClassifyReply(unittest.TestCase):
    def test_plain_acknowledgements(self):
        for t in ("haan ji bataiye", "Okay", "theek hai", "Haan, interested hoon"):
            self.assertEqual(classify_reply(t), "yes", t)

    def test_questions_and_objections_are_other(self):
        for t in ("Listing free hai kya?", "mera PG full hai", "aapka app kaise kaam karta hai"):
            self.assertEqual(classify_reply(t), "other", t)

    def test_devanagari_stt_output_is_understood(self):
        self.assertEqual(classify_reply("हाँ बताइए।"), "yes")
        self.assertEqual(classify_reply("नहीं।"), "no")

    def test_no_busy_link(self):
        self.assertEqual(classify_reply("nahi"), "no")
        self.assertEqual(classify_reply("abhi busy hoon"), "busy")
        self.assertEqual(classify_reply("link bhej do"), "link")


class TestStreamSentences(unittest.TestCase):
    def test_splits_into_sentences_as_they_complete(self):
        tags = []
        out = list(stream_sentences(["Achha ji. Samajh gayi, ", "Sir. Kya aap sunenge?"], tags))
        self.assertEqual(out, ["Achha ji.", "Samajh gayi, Sir.", "Kya aap sunenge?"])

    def test_does_not_split_urls_or_decimals(self):
        out = list(stream_sentences(["Link hai hostelpgdekho.com/app ji. Theek hai?"], []))
        self.assertEqual(out[0], "Link hai hostelpgdekho.com/app ji.")

    def test_markers_are_extracted_and_never_spoken(self):
        tags = []
        out = list(stream_sentences(["Dhanyavaad Sir, ", "din shubh ho. [INTER", "ESTED]"], tags))
        self.assertEqual(out, ["Dhanyavaad Sir, take care."])
        self.assertEqual(tags, ["INTERESTED"])

    def test_splits_when_model_forgets_the_space_after_a_full_stop(self):
        out = list(stream_sentences(["Main bata sakti hoon.Ji Sir, theek hai."], []))
        self.assertEqual(out, ["Main bata sakti hoon.", "Ji Sir, theek hai."])

    def test_markdown_is_stripped(self):
        self.assertEqual(list(stream_sentences(["**Haan** ji, bilkul."], [])), ["Haan ji, bilkul."])


class TestReplyEffects(unittest.TestCase):
    def test_detects_covered_points_and_link(self):
        e = reply_effects("Humara smart search aur Request Form dono hain, free listing bhi.",
                          "https://hostelpgdekho.com/app")
        self.assertEqual(
            set(e["slots"]),
            {"benefit_smart_search_explained", "benefit_vacant_form_explained", "benefit_free_listing_explained"},
        )
        self.assertFalse(e["link_read"])
        self.assertTrue(reply_effects("Yeh lijiye hostelpgdekho.com/app", "https://hostelpgdekho.com/app")["link_read"])


class TestConversationalFlow(unittest.TestCase):
    """Owner questions go to the LLM; plain acknowledgements get the script
    (covered in test_script_steps.py)."""

    def test_opening_is_the_scripted_line_and_a_question_goes_to_the_llm(self):
        llm = FakeStreamLLM(["Main Pooja hoon, HostelPGDekho se."])
        session, cm, out = _run(llm, ["aap kaun bol rahe ho?"])
        self.assertIn("My name is Pooja", out[0])
        self.assertEqual(llm.calls, 1)
        self.assertEqual(out[1], "Main Pooja hoon, HostelPGDekho se.")

    def test_plain_yes_is_answered_by_the_script_not_the_llm(self):
        llm = FakeStreamLLM([])
        session, cm, out = _run(llm, ["haan ji"])
        self.assertEqual(llm.calls, 0)
        self.assertTrue(cm.get_state().get_slot("permission_granted").is_filled())
        self.assertIn("sirf ek property listing app nahi", out[1])

    def test_first_turn_tells_llm_not_to_greet_again(self):
        llm = FakeStreamLLM(["Ji, batao."])
        session, cm, out = _run(llm, ["kaun bol raha hai?"])
        self.assertIn("Do not greet", llm.last_messages[1]["content"])

    def test_llm_is_told_to_answer_only_and_not_recite_the_script(self):
        llm = FakeStreamLLM(["Ji, free hai."])
        session, cm, out = _run(llm, ["free hai kya?"])
        notes = llm.last_messages[1]["content"]
        self.assertIn("do NOT recite script", notes)

    def test_pitch_points_in_the_reply_update_state(self):
        llm = FakeStreamLLM(["Humara smart search zyada owners tak inquiry pahunchata hai."])
        session, cm, out = _run(llm, ["ye kaise kaam karta hai?"])
        self.assertTrue(cm.get_state().get_slot("benefit_smart_search_explained").is_filled())

    def test_end_marker_ends_call_and_is_not_spoken(self):
        llm = FakeStreamLLM(["Theek hai Sir, dhanyavaad. [NOT_INTERESTED]"])
        session, cm, out = _run(llm, ["mujhe interest nahi hai, phir kabhi"])
        self.assertTrue(cm.get_state().call_ended)
        self.assertTrue(cm.get_state().not_interested)
        self.assertNotIn("NOT_INTERESTED", " ".join(out))

    def test_stray_interested_marker_before_pitch_is_ignored(self):
        llm = FakeStreamLLM(["Theek hai Sir, link hostelpgdekho.com/app hai. Dhanyavaad. [INTERESTED]"])
        session, cm, out = _run(llm, ["mujhe app ke baare mein batao"])
        self.assertFalse(cm.get_state().interested_confirmed)

    def test_end_marker_on_a_question_is_ignored(self):
        llm = FakeStreamLLM(["Ji Sir, listing free hai. Kya aap link dekhna chahenge? [CALLBACK]"])
        session, cm, out = _run(llm, ["free hai kya?"])
        self.assertFalse(cm.get_state().get_slot("interest_decision").is_filled())

    def test_marker_only_reply_still_gets_a_spoken_goodbye(self):
        llm = FakeStreamLLM(["[NOT_INTERESTED]"])
        session, cm, out = _run(llm, ["mujhe kuch bhi nahi chahiye bilkul bhi"])
        self.assertTrue(cm.get_state().call_ended)
        self.assertIn("Koi baat nahi", out[-1])

    def test_empty_llm_reply_is_not_silent(self):
        session, cm, out = _run(FakeStreamLLM([""]), ["mera PG full hai"])
        self.assertTrue(any("samajh nahi payi" in t for t in out))

    def test_good_night_is_mirrored_without_calling_the_llm(self):
        llm = FakeStreamLLM([])
        session, cm, out = _run(llm, ["Good night."])
        self.assertEqual(llm.calls, 0)
        self.assertIn("shubh ratri", out[-1])
        self.assertTrue(cm.get_state().call_ended)

    def test_stop_request_ends_without_the_llm(self):
        llm = FakeStreamLLM([])
        session, cm, out = _run(llm, ["dobara mat karna call"])
        self.assertEqual(llm.calls, 0)
        self.assertTrue(cm.get_state().call_ended)

    def test_one_transient_llm_failure_is_retried_not_fatal(self):
        class FlakyLLM:
            def __init__(self):
                self.calls = 0

            def chat_stream(self, messages):
                self.calls += 1
                if self.calls == 1:
                    raise GroqAllModelsFailedError("429")
                yield "Ji Sir, batati hoon."

        llm = FlakyLLM()
        with patch("pooja_cli.time.sleep"):
            session, cm, out = _run(llm, ["free hai kya?"])
        self.assertEqual(llm.calls, 2)
        self.assertTrue(any("Ek second" in t for t in out))
        self.assertTrue(any("batati hoon" in t for t in out))
        self.assertFalse(any("technical issue" in t for t in out))
        # the owner's message was recorded once, not twice
        self.assertEqual(sum(1 for m in session.messages if m["content"] == "free hai kya?"), 1)

    def test_dobara_batao_goes_to_the_llm_not_an_exact_replay(self):
        llm = FakeStreamLLM(["Seedhi baat: baaki apps pe inquiry kam owners ko milti hai."])
        session, cm, out = _run(llm, ["haan", "dobara batao"])
        self.assertEqual(llm.calls, 1)
        self.assertNotEqual(out[1], out[2])

    def test_llm_outage_ends_the_call_gracefully(self):
        with patch("pooja_cli.time.sleep"):
            session, cm, out = _run(DownLLM(), ["free hai kya?"])
        self.assertTrue(cm.get_state().call_ended)
        self.assertIn("technical issue", out[-1])

    def test_interrupted_before_speaking_stays_silent(self):
        llm = FakeStreamLLM(["Ji Sir, batati hoon."])
        out = []
        it = iter(["free hai kya?", "bye"])
        session = PoojaSession(
            llm=llm,
            get_user_input=lambda: next(it, "bye"),
            deliver_reply=lambda t: out.append(t),
            conversational=True,
            deliver_stream=lambda sentences: ("", True),  # owner talked over her before any audio
        )
        session.run()
        self.assertFalse(any("samajh nahi payi" in t for t in out))

    def test_no_back_to_back_questions(self):
        llm = FakeStreamLLM(["Kya aap sunna chahenge?", "Achha Sir, samajh gayi."])
        session, cm, out = _run(llm, ["free hai kya?", "mera PG full hai"])
        self.assertIn("must NOT end with a question", llm.last_messages[-1]["content"])

    def test_conversation_history_is_passed_to_the_llm(self):
        llm = FakeStreamLLM(["Ji, batati hoon.", "Achha, samajh gayi."])
        session, cm, out = _run(llm, ["free hai kya?", "mera PG full hai"])
        roles = [m["role"] for m in llm.last_messages]
        self.assertEqual(roles[:2], ["system", "system"])
        self.assertIn("mera PG full hai", llm.last_messages[-2]["content"])
        self.assertIn("Roman script only", llm.last_messages[-1]["content"])  # per-turn style reminder


if __name__ == "__main__":
    unittest.main()
