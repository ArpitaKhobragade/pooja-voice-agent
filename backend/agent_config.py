"""
Agent configuration for Pooja, HostelPGDekho.com's AI voice agent.

Single plain dict, isolated in its own file, read by objective.py and
prompts.py. ConversationManager and the pipeline code never hardcode
"Pooja" or "HostelPGDekho" directly — everything business-specific
lives here, so this agent could be re-pointed at a different script by
editing only this file.
"""

import datetime
import os

AGENT_CONFIG = {
    "persona_name": "Pooja",
    "company_name": "HostelPGDekho.com",
    "tts_aggregation_mode": "sentence",  # options: "sentence", "phrase", "token"

    # The app URL Pooja offers to "send" (she can't actually send WhatsApp
    # messages in this local demo — she displays this instead). Configurable
    # so a fabricated production URL is never hardcoded into the prompt.
    "app_url": os.environ.get("HOSTELPGDEKHO_APP_URL", "https://hostelpgdekho.com/app"),

    # Phases of the approved sales script, filled in priority order.
    # objective.py picks the first missing one each turn — this is NOT a
    # rigid sequence; an owner who volunteers interest early shouldn't
    # confuse it, we just look at what's still missing.
    "required_slots": [
        "permission_granted",
        "benefit_smart_search_explained",
        "benefit_vacant_form_explained",
        "benefit_free_listing_explained",
        "interest_decision",
    ],

    # Any one of these CallState flags being true ends the call.
    "terminal_conditions": [
        "interested_confirmed",
        "not_interested",
        "call_ended",
    ],

    # Human-readable instructions per slot, used to build the "do this
    # next" objective for the LLM. These paraphrase (not quote verbatim)
    # the approved HostelPGDekho sales script so Pooja delivers it in
    # short conversational turns instead of reading it continuously.
    "slot_descriptions": {
        "permission_granted": (
            "Greet the owner as Pooja from HostelPGDekho.com and ask if now is a "
            "good time to talk about their PG. If they say they're busy, ask if "
            "another time would work, and if they decline, respect it and end the "
            "call politely — do not push further."
        ),
        "benefit_smart_search_explained": (
            "Explain that HostelPGDekho is more than a listing app: on other "
            "portals, when a PG is full the owner can miss inquiries because "
            "users only message a handful of the listed PGs. HostelPGDekho's "
            "smart search helps route relevant inquiries to more PG owners, not "
            "just the first few. Do not invent statistics or guarantee inquiries."
        ),
        "benefit_vacant_form_explained": (
            "Explain the Vacant-Bed Request Form: even if their PG is full right "
            "now, when a bed opens up in a few months they can fill in the "
            "vacancy details on the platform, which helps relevant new inquiries "
            "reach them for that vacancy. Do not promise guaranteed leads."
        ),
        "benefit_free_listing_explained": (
            "Mention that while some platforms charge for listings, "
            "HostelPGDekho is currently offering PG owners free listing. Do not "
            "invent any other pricing details or conditions."
        ),
        "interest_decision": (
            "Ask whether the owner would like to use HostelPGDekho and list "
            "their vacant beds. Accept a clear yes, no, or 'maybe later' answer "
            "— do not pressure them if they hesitate or decline."
        ),
    },

    # Optional, not part of the required sequence — triggered whenever the
    # owner asks for the app link, at any point in the call.
    "link_request_instruction": (
        "If the owner asks for the app link or for it to be sent on WhatsApp: "
        "this is a local demo and cannot actually send WhatsApp messages. Say so "
        "plainly, do not claim a message was sent, and offer to read out the "
        "configured app URL instead."
    ),

    # Literal approved-script lines. {greeting}, {day_wish} and {app_url} are
    # filled in by script_line() below (greeting/wish follow the clock).
    "script_lines": {
        "opening": (
            "Hello ji, {greeting}. My name is Pooja, and I'm calling from HostelPGDekho.com. "
            "Actually mujhe aapke PG ke regarding baat karni hai, toh kya hum baat kar sakte hain?"
        ),
        "ask_reschedule": "Koi baat nahi, Sir. Kya main aapko baad mein callback kar sakti hoon?",
        "ask_okay_followup": "Sir, kya aap chahenge main thodi si process short mein batau?",
        "ask_vacancy_feature_permission": (
            "Samajh gayi, Sir. Jab kabhi koi bed vacant ho, uske liye hum relevant "
            "inquiries tak pahunchne mein help karte hain — kya aap uske baare mein "
            "sunna chahenge?"
        ),
        # Short spoken beats (each roughly 10-14 seconds), so the call is a
        # back-and-forth instead of one long monologue.
        "beat_1": (
            "Sir, main batana chahungi ki HostelPGDekho sirf ek property listing "
            "app nahi hai, iske kuch features baaki apps se kaafi better hain. "
            "Jaise, baaki portals par user ko har PG par alag se inquiry bhejni padti hai."
        ),
        "beat_2": (
            "Ab samajhiye 50 PGs hain, toh user zyada se zyada 10-15 PGs par hi "
            "inquiry karega, aur baaki owners ko inquiry milti hi nahi. Humare "
            "smart search se hum zyada se zyada PG owners tak inquiry pahunchate "
            "hain. Samajh aa raha hai, Sir?"
        ),
        "beat_3": (
            "Ab maan lijiye Sir, aapka PG abhi full hai, lekin 3-4 mahine baad "
            "ek bed khali hota hai. Doosre apps par us time inquiry milne ka "
            "koi option nahi hota, lekin humare app mein Request Form hai."
        ),
        "beat_4": (
            "Usme aap apni vacancy ki details fill kar dijiye, aur us request ke "
            "basis par aapko nayi inquiries mil sakti hain. Aur Sir, jahan baaki "
            "platforms listing ke paise charge karte hain, hum abhi free listing de rahe hain."
        ),
        "ask_interest": (
            "Toh Sir, main chahungi ki aap humare app ko use karein, aur iske "
            "features se apne PG ke beds ki requirement poori karein. "
            "Kya aap interested hain?"
        ),
        "link_offer_demo_limitation": (
            "Sir, yeh ek local demo hai, isliye main abhi WhatsApp par message "
            "nahi bhej sakti. Lekin main aapko app ka URL bata deti hoon: {app_url}"
        ),
        "closing_interested": (
            "Bahut bahut dhanyavaad, Sir! Aap {app_url} par jaakar start kar "
            "sakte hain. {day_wish}"
        ),
        "closing_not_interested": (
            "Koi baat nahi, Sir. Aapka time dene ke liye dhanyavaad. {day_wish}"
        ),
        "closing_busy_or_declined": (
            "Theek hai Sir, main aapka time nahi loongi. Dhanyavaad. {day_wish}"
        ),
        "reintro": (
            "Ji, main Pooja bol rahi hoon, HostelPGDekho.com se. Aapke PG ke regarding baat karni thi, "
            "kya hum abhi baat kar sakte hain?"
        ),
        "closing_no_response": (
            "Lagta hai abhi aapse baat nahi ho pa rahi, Sir. Main baad mein call karungi. Dhanyavaad!"
        ),
        "closing_goodnight": "Good night Sir! Aapko bhi shubh ratri. Dhanyavaad.",
    },
}


# The approved sales script, in order. Voice mode delivers these steps close to
# word for word (adapting only a little to what the owner just said), one step
# per turn, so the call follows the script while still reacting to the owner.
# "covered" is a pattern that shows the step has been said, whoever worded it.
# "Sir" is swapped for "Ma'am" when the owner sounds like a woman.
SCRIPT_STEPS = [
    {
        "id": "intro",
        "text": (
            "Sir, main batana chahungi ki HostelPGDekho sirf ek property listing app nahi hai. "
            "Iske kuch features aise hain jo baaki apps aur websites se bahut better hain."
        ),
        "covered": r"listing\s*app\s*nahi|property\s*listing",
    },
    {
        "id": "smart_search",
        "text": (
            "Jaise, baaki portals pe jab user PG search karne jaata hai, toh use har PG par alag se inquiry "
            "bhejni padti hai. Ab samajhiye 50 PGs hain, toh woh zyada se zyada 10 se 15 PGs par hi inquiry "
            "karega, baaki PG owners ko inquiry milti hi nahi. Lekin humare smart search feature ke through "
            "hum zyada se zyada PG owners tak inquiry pahunchate hain."
        ),
        "covered": r"smart\s*search",
    },
    {
        "id": "request_form",
        "text": (
            "Ab maan lijiye aapka PG abhi full hai, lekin 3 se 4 mahine baad aapke PG ka koi ek bed bhi khali "
            "hota hai. Tab doosre website ya app mein us time par inquiry milne ka koi option nahi hota, lekin "
            "humare app mein aisa option hai. Aap Request Form par jaakar apni vacancy ki details fill kar "
            "sakte hain, aur us request ke basis par aapko nayi inquiry mil jaati hai."
        ),
        "covered": r"request\s*form",
    },
    {
        "id": "free_listing",
        "text": (
            "Aur Sir, baaki platforms jo sirf listing ke liye paise charge kar rahe hain, hum abhi aap logon ko "
            "free listing de rahe hain."
        ),
        "covered": r"free\s*listing|listing.{0,30}free",
    },
    {
        "id": "ask_interest",
        "text": (
            "Toh main chahungi ki aap humare is app ko use karein, aur iske features ka use karke apne PGs ke "
            "beds ki requirement poori karein. Kya aap interested hain?"
        ),
        "covered": r"interested|use\s*kar(ein|na)",
    },
]
FINAL_STEP_TEXT = (
    "Main aapko app ka link bata deti hoon, aap start kar sakte hain. Demo mein main WhatsApp par bhej nahi "
    "sakti, isliye link yahin bol rahi hoon: {app_url}."
)


def _now():
    """Current time in the callers' timezone (default India). A cloud server runs on
    UTC, which would otherwise make the greeting wrong by 5.5 hours."""
    name = os.environ.get("POOJA_TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"
    try:
        from zoneinfo import ZoneInfo

        return datetime.datetime.now(ZoneInfo(name))
    except Exception:  # unknown zone or no tz database: fall back to the machine's clock
        return datetime.datetime.now()


def greeting() -> str:
    """Opening greeting. Uses the clock only where it can't embarrass: a
    morning/afternoon/evening word during normal calling hours, and a plain
    'namaste' late at night or very early (never 'good afternoon' at 2am)."""
    override = os.environ.get("POOJA_GREETING", "").strip()
    if override:
        return override
    hour = _now().hour
    if 5 <= hour < 12:
        return "good morning"
    if 12 <= hour < 17:
        return "good afternoon"
    if 17 <= hour < 21:
        return "good evening"
    return "namaste"


def day_wish() -> str:
    """Closing wish. Deliberately NOT time-of-day based: Pooja cannot know
    whether it is day or night where the owner is, so 'din shubh ho' or
    'shubh ratri' can be wrong. (If the owner themselves says good night,
    the session mirrors it.)"""
    return "take care!"


def script_line(key: str) -> str:
    """A script line with all placeholders filled in."""
    text = AGENT_CONFIG["script_lines"][key]
    return text.format(app_url=AGENT_CONFIG["app_url"], greeting=greeting(), day_wish=day_wish().capitalize())
