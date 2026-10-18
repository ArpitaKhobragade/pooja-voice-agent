"""
Tool schema for the local Pooja CLI, in plain OpenAI/Groq function-calling
JSON shape.

This mirrors tools.py (the Pipecat/FunctionSchema version used by the
legacy Twilio pipeline) but has no pipecat import, so `pooja_cli.py` can
run with just the `groq` SDK installed — pipecat and its heavier
dependencies (Twilio, ONNX turn-detection models, etc.) are only needed
if someone wants to run the original telephony pipeline.

Keep the slot enum here in sync with tools.py's update_state_tool.
"""

# Single source of truth for which slots the LLM is allowed to set,
# shared by the tool schema below AND pooja_cli.py's argument validator
# (so a malformed/hallucinated slot name is rejected before it ever
# reaches ConversationManager — see PoojaSession._run_tool_loop).
VALID_SLOTS = (
    "permission_granted",
    "benefit_smart_search_explained",
    "benefit_vacant_form_explained",
    "benefit_free_listing_explained",
    "interest_decision",
    "link_requested",
    "callback_requested",
    "owner_name",
)

UPDATE_STATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_state",
        "description": (
            "Record information ONLY after the owner explicitly states it. "
            "Never call this tool with guessed, assumed, inferred or unknown "
            "values. Do NOT call this tool to update a slot if it is already "
            "listed in the 'Known' section of your system prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {
                    "type": "string",
                    "description": "Which piece of information this is.",
                    "enum": list(VALID_SLOTS),
                },
                "value": {
                    "type": "string",
                    "description": "The value as the owner stated it, in their own words (keep it short).",
                },
            },
            "required": ["slot", "value"],
        },
    },
}

LOCAL_TOOLS = [UPDATE_STATE_TOOL]
