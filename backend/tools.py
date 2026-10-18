"""
Tool schemas exposed to Gemini during the call.

IMPORTANT (per architecture discussion): these tools express INTENT,
not final outcomes. Gemini calling book_meeting(...) means "the model
thinks the caller wants to book a meeting" — it's the
ConversationManager (conversation_manager.py) that decides whether
that's actually valid and finalizes the outcome. Keep the handler
logic in ConversationManager, not here — this file only declares the
shape of each tool for the LLM.
"""

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

update_state_tool = FunctionSchema(
    name="update_state",
    description=(
        "Record information ONLY after the caller explicitly states it. "
        "Never call this tool with guessed, assumed, inferred or unknown values. "
        "CRITICAL: Do NOT extract ambiguous or hallucinated values (e.g. 'Everything' for a project, or '3 kilometers' for office size). If an answer is unclear, DO NOT update the state. Ask a clarifying question instead.\n"
        "CRITICAL: Do NOT call this tool to update a slot if it is already listed in the 'Known' section of your system prompt. Only extract new information."
    ),
    properties={
        "slot": {
            "type": "string",
            "description": "Which piece of information this is.",
            "enum": [
                "permission_granted",
                "benefit_smart_search_explained",
                "benefit_vacant_form_explained",
                "benefit_free_listing_explained",
                "interest_decision",
                "link_requested",
                "callback_requested",
                "owner_name",
            ]

        },
        "value": {
            "type": "string",
            "description": "The value as the caller stated it, in their own words (keep it short).",
        },

    },
    required=["slot", "value"],
)

SIA_TOOLS = ToolsSchema(
    standard_tools=[
        update_state_tool,
    ]
)
