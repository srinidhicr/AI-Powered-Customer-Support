# src/ui/components/commands.py
"""
Slash-command handler.
Returns a 6-tuple matching the chat() output signature:
  (history, ticket_id, msg_clear, ticket_list_update, ticket_info, chunks_html)
"""

from __future__ import annotations
import gradio as gr

from src.db.ticket_store import close_ticket, get_ticket
from src.ui.ticket_helpers import ticket_label, icon, ticket_info_md
from src.db.ticket_store import list_tickets


def _refresh_list():
    tickets = list_tickets()
    if not tickets:
        return gr.update(choices=[], value=None)
    return gr.update(choices=[(ticket_label(t), t["id"]) for t in tickets])


def handle(cmd: str, history: list, ticket_id: str | None):
    """Dispatch a slash command and return the full output tuple."""
    cmd     = cmd.strip().lower()
    history = history + [{"role": "user", "content": cmd}]
    noup    = gr.update()

    if cmd == "/help":
        reply = (
            "**Available commands**\n\n"
            "- `/new` — Start a new ticket\n"
            "- `/resolve` — Mark current ticket resolved\n"
            "- `/status` — Show ticket info\n"
            "- `/help` — Show this menu"
        )
        history.append({"role": "assistant", "content": reply})
        return history, ticket_id, "", noup, "", ""

    if cmd == "/new":
        history.append({"role": "assistant", "content": "🎫 Ready. Describe your issue to open a new ticket."})
        return history, None, "", noup, "", ""

    if cmd == "/resolve":
        if not ticket_id:
            history.append({"role": "assistant", "content": "No active ticket to resolve."})
            return history, ticket_id, "", noup, "", ""
        close_ticket(ticket_id)
        history.append({"role": "assistant", "content": f"✅ Ticket `{ticket_id}` marked as resolved."})
        return history, ticket_id, "", _refresh_list(), "", ""

    if cmd == "/status":
        if not ticket_id:
            history.append({"role": "assistant", "content": "No active ticket selected."})
            return history, ticket_id, "", noup, "", ""
        t = get_ticket(ticket_id)
        reply = (
            f"**Ticket** `{t['id']}`\n\n"
            f"**Category:** {t['category']}\n\n"
            f"**Status:** {t['status']}\n\n"
            f"**Created:** {t['created_at'][:10]}"
        )
        history.append({"role": "assistant", "content": reply})
        return history, ticket_id, "", noup, "", ""

    history.append({"role": "assistant", "content": "Unknown command. Type `/help` for options."})
    return history, ticket_id, "", noup, "", ""