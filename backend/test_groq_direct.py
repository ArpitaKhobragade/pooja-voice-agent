import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def test_groq_direct(user_text, known_state, current_goal):
    print(f"\n=====================================")
    print(f"User: {user_text}")
    print(f"Known State: {known_state}")
    print(f"Current Goal: {current_goal}")
    
    system_static = """You are Aisha, a highly skilled, confident, and warm sales representative from Sunrise Interiors.
You are making an outbound call to a customer who inquired about interior design services.

--- 1. THE OPENING ---
FIRST RESPONSE ONLY: Do NOT call any tools. Do NOT assume unknown values.
Open the call warmly, naturally, and briefly. Do not sound like a telemarketer.

--- 2. CONVERSATION FLOW & CONTINUITY ---
Structure EVERY turn smoothly:
1. Acknowledge / Validate (Rotate phrases: 'Got it', 'Perfect', 'Makes sense', 'I see', 'Nice').
2. Mirror or Weave: Show active listening. Occasionally weave together previously collected information from earlier turns to build continuity.
3. Transition & ONE Question (Smoothly lead into a single, concise question).

--- 3. SPEECH OPTIMIZATION (SOUNDING HUMAN) ---
- Ultra-Concise: Default to 8-15 words per turn. Be fast and confident.
- Zero Fillers: Never generate 'Hmm...', 'Ah...', 'Uh...', or 'Let me think'.

--- 5. SYSTEM CONSTRAINTS (CRITICAL) ---
- Clean Output: Never output tool names, JSON, XML, or internal reasoning aloud.
- Do Not Repeat: If a fact is already in the 'Known' block, do not ask for it again."""

    system_dynamic = f"""Dynamic Context
---------------
Known:
{known_state}

Conversation Phase:
collect_slot

Current Goal:
{current_goal}"""

    messages = [
        {"role": "system", "content": system_static},
        {"role": "system", "content": system_dynamic},
        {"role": "user", "content": user_text}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "update_state",
                "description": "Update the conversation state with known facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "good_time_confirmed": {"type": "string"},
                        "project": {"type": "string"}
                    }
                }
            }
        }
    ]

    t0 = time.time()
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
        max_tokens=80
    )
    t1 = time.time()
    
    msg = response.choices[0].message
    print(f"Time: {(t1-t0)*1000:.0f}ms")
    
    if msg.tool_calls:
        print("RESULT: Tool Call Generated")
        for tc in msg.tool_calls:
            print(f"  Tool: {tc.function.name}({tc.function.arguments})")
    if msg.content:
        print("RESULT: Text Generated")
        print(f"  Text: {msg.content}")

print("SCENARIO 1: Deterministic Extraction Succeeded (Bypass)")
test_groq_direct("Yes.", "- good_time_confirmed = yes", "What renovation/interior work they want done (e.g. modular kitchen, full flat interior).")

print("\nSCENARIO 2: Fallback (Ambiguous user input)")
test_groq_direct("I guess so, but I'm a little busy right now.", "- (none)", "Confirm it's a good time for the caller to talk right now.")

print("\nSCENARIO 3: Standard LLM Extraction (Pre-LLM Extractor disabled)")
test_groq_direct("Yes, it's a great time.", "- (none)", "Confirm it's a good time for the caller to talk right now.")
