# src/tools/clarify.py

import json
from langchain.tools import tool


@tool
def clarify(query: str, reason: str) -> str:
    """Called when a query is ambiguous, out of scope, or has low classification confidence.
    Returns a clarifying question for the support agent to present to the customer.
    Call this INSTEAD of retrieve() when confidence < 0.65 or in_scope is False.

    Args:
        query:  The original customer query.
        reason: One of: 'low_confidence', 'out_of_scope', 'ambiguous'

    Returns:
        JSON string with keys: clarifying_question, reason, action
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
        "action"             : "present_to_agent"
    })