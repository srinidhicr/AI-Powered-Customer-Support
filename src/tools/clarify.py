# src/tools/clarify.py

import json
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal


# ── Pydantic input schema ─────────────────────────────────────────────────────

class ClarifyInput(BaseModel):
    query: str = Field(
        description="The original customer query."
    )
    reason: Literal["low_confidence", "out_of_scope", "ambiguous"] = Field(
        description=(
            "Why clarification is needed. Must be one of: "
            "'low_confidence' (classifier uncertain), "
            "'out_of_scope' (topic not supported), "
            "'ambiguous' (query could mean multiple things)."
        )
    )


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool(args_schema=ClarifyInput)
def clarify(query: str, reason: str) -> str:
    """Called when a query is ambiguous, out of scope, or has low classification confidence.
    Returns a clarifying question for the support agent to present to the customer.
    Call this INSTEAD of retrieve() when confidence is very low or the query is out of scope.
    Returns a JSON string with keys: clarifying_question, reason, action.
    """
    questions = {
        "low_confidence": (
            "Could you provide a bit more detail about your issue? "
            "For example, is this related to billing, a technical problem, "
            "or something else? This will help us route your query correctly."
        ),
        "out_of_scope": (
            "Your query doesn't appear to match our supported topics. "
            "Could you clarify what you need help with? "
            "We handle billing, technical issues, returns, and general inquiries."
        ),
        "ambiguous": (
            "Your query could relate to multiple areas. "
            "Are you asking about a billing matter, a technical issue, "
            "a return or exchange, or something else entirely?"
        ),
    }

    return json.dumps({
        "clarifying_question": questions.get(reason, questions["ambiguous"]),
        "reason"             : reason,
        "action"             : "present_to_agent",
    })
