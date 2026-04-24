# src/ui/guardrails.py
"""
Stateless guardrails

check()       → general guardrail check (greeting / noise / drift)
is_followup() → detects clarification / elaboration follow-ups so the
                pipeline can skip retrieval and answer from prior context
"""

from __future__ import annotations
from typing import Literal

from src.ui.turn_intent import DOMAIN_KW, low_information_reply, route_turn

GREETINGS: set[str] = {
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "hiya", "sup", "yo", "hi there", "hey there",
}

NOISE: set[str] = {
    "yes", "no", "ok", "okay", "sure", "alright", "got it", "noted",
    "k", "bye", "goodbye", "thanks", "thank you", "cheers", "see you",
    "sounds good", "perfect", "great",
}

# Words that almost exclusively appear in follow-up / clarification messages
_FOLLOWUP_TRIGGERS: set[str] = {
    "explain", "elaborate", "clarify", "simpler", "simple", "simplify",
    "understand", "confused", "confusing", "again", "rephrase", "reword",
    "what do you mean", "what does that mean", "can you explain",
    "in simple terms", "in simpler terms", "more detail", "more details",
    "break it down", "dumb it down", "layman", "example", "examples",
    "i didn't understand", "i don't understand", "didnt understand",
    "dont understand", "not clear", "unclear", "what is", "what are",
    "how does", "how do", "tell me more", "go on", "continue",
}

Action = Literal["pass", "greet", "noise", "drift"]


def check(message: str, ticket_category: str | None) -> dict | None:
    """
    Run all guardrail checks.
    Returns None → proceed to pipeline.
    Returns dict{"action", "reply"} → short-circuit with that reply.
    """
    lower = message.strip().lower()
    words = set(lower.split())

    # Greeting
    if lower in GREETINGS or bool(words & GREETINGS):
        return {
            "action": "greet",
            "reply": (
                "👋 Hello! I'm your support assistant. "
                "Please describe your issue and I'll get you the right help."
            ),
        }

    # Noise / filler
    if lower in NOISE:
        return {
            "action": "noise",
            "reply": "😊 Got it! Let me know if there's anything else I can help with.",
        }

    # Topic drift — only when a ticket is already open
    if ticket_category and len(message.split()) > 5:
        ticket_kws  = DOMAIN_KW.get(ticket_category, set())
        ticket_hits = len(words & ticket_kws)
        for cat, kws in DOMAIN_KW.items():
            if cat == ticket_category or not kws:
                continue
            if len(words & kws) >= 2 and ticket_hits == 0:
                return {
                    "action": "drift",
                    "reply": (
                        f"⚠️ This looks like a **{cat}** issue, but your current "
                        f"ticket is for **{ticket_category}**.\n\n"
                        "Please click **＋ New Ticket** to raise it separately, "
                        "or rephrase if it's related to your current issue."
                    ),
                }

    return None


def is_followup(message: str, ticket_category: str | None) -> bool:
    """
    Returns True when the message is a conversational follow-up that asks
    for clarification / elaboration of the *previous* bot reply, rather than
    a new standalone query that needs fresh retrieval.

    Criteria (ALL must be true):
      1. A ticket is already open (ticket_category is not None)
      2. Message is short (≤ 12 words)
      3. Message contains at least one follow-up trigger word
      4. Message contains NO domain keywords for any category
         (so we don't misclassify "explain my billing charge" as a follow-up)
    """
    if not ticket_category:
        return False

    lower  = message.strip().lower()
    words  = set(lower.split())

    if len(words) > 12:
        return False

    has_trigger = any(t in lower for t in _FOLLOWUP_TRIGGERS)
    if not has_trigger:
        return False

    # If the message contains real domain keywords it's a new query, not a follow-up
    all_domain_kws = set().union(*DOMAIN_KW.values())
    if words & all_domain_kws:
        return False

    return True
