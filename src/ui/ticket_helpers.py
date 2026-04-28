# src/ui/ticket_helpers.py
"""
Python helpers for ticket management and label formatting.
"""
 
from __future__ import annotations
import uuid
from datetime import datetime
 
from src.db.ticket_store import (
    create_ticket, get_ticket, list_tickets,
    add_message, get_messages, close_ticket,
)

CATEGORY_ICONS: dict[str, str] = {
    "Technical"            : "⚙️",
    "Billing and Payments" : "💳",
    "Product Inquiry"      : "📦",
    "Returns and Exchanges": "↩️",
    "Human Resources"      : "👤",
    "General Inquiry"      : "💬",
}
 
CATEGORY_COLORS: dict[str, str] = {
    "Technical"            : "#3b82f6",
    "Billing and Payments" : "#10b981",
    "Product Inquiry"      : "#8b5cf6",
    "Returns and Exchanges": "#f59e0b",
    "Human Resources"      : "#06b6d4",
    "General Inquiry"      : "#6b7280",
}

def icon(category: str) -> str:
    return CATEGORY_ICONS.get(category, "🎫")

def ticket_label(t: dict) -> str:
    status = "✅" if t["status"] == "resolved" else "🟢"
    return f"{status} {icon(t['category'])} {t['id'][:10]}  {t['category']}  ({t['created_at'][:10]})"

def make_ticket_id() -> str:
    return f"TKT-{uuid.uuid4().hex[:8].upper()}"

def ticket_info_md(ticket_id: str) -> str:
    """Return a one-line markdown summary for the header bar."""
    t = get_ticket(ticket_id)
    if not t:
        return ""
    status = "✅ Resolved" if t["status"] == "resolved" else "🟢 Open"
    return (
        f"{icon(t['category'])} **{t['category']}** &nbsp;|&nbsp; "
        f"{status} &nbsp;|&nbsp; `{ticket_id}`"
    )

def _last_substantive_assistant_msg(ticket_id: str) -> dict | None:
    """
    Return the most recent assistant message that was a real support answer
    (i.e. has source chunks attached, meaning it went through retrieval).
    Skips guardrail replies, routing notices, and out-of-scope messages.
    """
    db_msgs = get_messages(ticket_id)
    for m in reversed(db_msgs):
        if m["role"] != "assistant":
            continue
        # Skip ticket-open banners
        if m["content"].startswith("🎫"):
            continue
        # Skip messages with no chunks — these are guardrail/routing replies
        if m.get("chunks"):
            return m
    return None


def get_last_bot_reply(ticket_id: str) -> str:
    """
    Return the most recent assistant message text for a ticket.
    Used when the user sends a follow-up clarification so we can
    pass the prior answer as context instead of doing fresh retrieval.
    """
    msg = _last_substantive_assistant_msg(ticket_id)
    return msg["content"] if msg else ""


def get_last_assistant_chunks(ticket_id: str) -> list:
    """Return source chunks attached to the latest assistant reply."""
    msg = _last_substantive_assistant_msg(ticket_id)
    return msg.get("chunks", []) if msg else []


def get_last_user_message(ticket_id: str) -> str:
    """
    Return the user message that preceded the last substantive assistant reply.
    This is the original question the support answer was about.
    """
    db_msgs = get_messages(ticket_id)
    # Find the last substantive assistant message index
    target_idx = None
    for i in reversed(range(len(db_msgs))):
        m = db_msgs[i]
        if m["role"] == "assistant" and not m["content"].startswith("🎫") and m.get("chunks"):
            target_idx = i
            break
    if target_idx is None:
        # fallback: just return the last user message
        for m in reversed(db_msgs):
            if m["role"] == "user":
                return m["content"]
        return ""
    # Walk backwards from that assistant message to find the user turn before it
    for i in range(target_idx - 1, -1, -1):
        if db_msgs[i]["role"] == "user":
            return db_msgs[i]["content"]
    return ""


def get_prior_context(ticket_id: str, max_turns: int = 3) -> str:
    """
    Build a compact transcript using only substantive support exchanges.
    Skips out-of-scope replies, routing notices, and guardrail messages.
    """
    db_msgs = get_messages(ticket_id)
    
    # Keep only pairs where the assistant message has chunks (real support answers)
    # plus the user message that preceded each one
    substantive_pairs = []
    for i, m in enumerate(db_msgs):
        if (m["role"] == "assistant" 
                and not m["content"].startswith("🎫") 
                and m.get("chunks")):
            # find the user message before this
            for j in range(i - 1, -1, -1):
                if db_msgs[j]["role"] == "user":
                    substantive_pairs.append((db_msgs[j], m))
                    break

    # Take the last max_turns substantive pairs
    recent = substantive_pairs[-max_turns:]
    if not recent:
        return ""

    lines = []
    for user_msg, bot_msg in recent:
        lines.append(f"Customer: {user_msg['content'][:220]}")
        lines.append(f"Assistant: {bot_msg['content'][:220]}")

    return "\n\nRecent conversation:\n" + "\n".join(lines)
