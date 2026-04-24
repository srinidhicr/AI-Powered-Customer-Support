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

def get_prior_context(ticket_id: str, max_turns: int = 3) -> str:
    """
    Build a compact conversation transcript from recent turns in this ticket.
    Used to give the pipeline conversational grounding without dumping the full thread.
    """
    db_msgs = get_messages(ticket_id)
    conversation = [
        m for m in db_msgs
        if not (m["role"] == "assistant" and m["content"].startswith("🎫"))
    ]
    prior = conversation[-(max_turns * 2):-1]
    if not prior:
        return ""
    lines: list[str] = []
    for msg in prior:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:220]}")
    return "\n\nRecent conversation:\n" + "\n".join(lines)
 
 
def get_last_bot_reply(ticket_id: str) -> str:
    """
    Return the most recent assistant message text for a ticket.
    Used when the user sends a follow-up clarification so we can
    pass the prior answer as context instead of doing fresh retrieval.
    """
    db_msgs = get_messages(ticket_id)
    for m in reversed(db_msgs):
        if m["role"] == "assistant" and not m["content"].startswith("🎫"):
            return m["content"]
    return ""


def get_last_user_message(ticket_id: str) -> str:
    """Return the most recent user message for the ticket."""
    db_msgs = get_messages(ticket_id)
    for m in reversed(db_msgs):
        if m["role"] == "user":
            return m["content"]
    return ""


def get_last_assistant_chunks(ticket_id: str) -> list:
    """Return source chunks attached to the latest assistant reply."""
    db_msgs = get_messages(ticket_id)
    for m in reversed(db_msgs):
        if m["role"] == "assistant" and not m["content"].startswith("🎫"):
            return m.get("chunks", []) or []
    return []
 
