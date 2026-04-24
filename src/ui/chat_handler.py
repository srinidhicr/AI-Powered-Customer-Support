# src/ui/chat_handler.py
"""Core chat logic for the Gradio app."""

from __future__ import annotations
import gradio as gr
from datetime import datetime
 
from src.agents.orchestrator import run, get_final_draft, answer_followup
from src.db.ticket_store     import (
    create_ticket, get_ticket, list_tickets,
    add_message, get_messages, close_ticket,
)
from src.ui import guardrails
from src.ui import scheduler
from src.ui.turn_intent import classify_turn_intent
from src.ui.ticket_helpers   import (
    icon, ticket_label, make_ticket_id, ticket_info_md,
    get_prior_context, get_last_bot_reply, get_last_user_message,
    get_last_assistant_chunks, CATEGORY_COLORS,
)
from src.ui.components       import chunks as chunks_renderer
from src.ui.components       import commands
# === Helpers ===
def _refresh_list():
    tickets = list_tickets()
    if not tickets:
        return gr.update(choices=[], value=None)
    return gr.update(choices=[(ticket_label(t), t["id"]) for t in tickets])


def _scheduler_hidden_updates():
    return (
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(choices=scheduler.get_slot_options(), value=None),
        "",
        gr.update(value="", visible=False),
    )


def _scheduler_visible_updates(status: str = "", preview: str = ""):
    return (
        gr.update(visible=True),
        gr.update(value=""),
        gr.update(choices=scheduler.get_slot_options(), value=None),
        status,
        gr.update(value=preview, visible=bool(preview)),
    )


def _is_low_relevance_result(chunks: list) -> bool:
    if not chunks:
        return True
    best_rerank = max(float(chunk.get("rerank_score", 0) or 0) for chunk in chunks)
    return best_rerank < 1.5

def _handle_followup(message: str,history: list,ticket_id: str,use_cache: bool) -> tuple:
    """
    Handle a clarification / elaboration follow-up.
    Feeds the previous bot reply as context to the LLM so it can rephrase
    without triggering retrieval of completely unrelated KB chunks.
    """
    prior_reply = get_last_bot_reply(ticket_id)
    prior_user  = get_last_user_message(ticket_id)
    prior_chunks = get_last_assistant_chunks(ticket_id)
 
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "⏳ Generating response…"},
    ]
    yield history, ticket_id, "", gr.update(), "", "", *_scheduler_hidden_updates()
 
    draft = answer_followup(
        previous_question=prior_user,
        previous_response=prior_reply,
        followup_message=message,
        source_chunks=prior_chunks,
    )
    chunks = prior_chunks
 
    if not draft:
        draft = "I'm sorry, I wasn't able to elaborate further. Could you share more details about what was unclear?"
 
    add_message(ticket_id, "user",      message)
    add_message(ticket_id, "assistant", draft, chunks=chunks)
 
    history[-2] = {"role": "user",      "content": message}
    history[-1] = {"role": "assistant", "content": draft}
 
    yield history, ticket_id, "", _refresh_list(), ticket_info_md(ticket_id), chunks_renderer.render(chunks), *_scheduler_hidden_updates()

# === Main chat generator ===
def chat(message: str, history: list, ticket_id: str, use_cache: bool):
    """Main entry point for each chat message."""
    if not message.strip():
        yield history, ticket_id, "", gr.update(), "", "", *_scheduler_hidden_updates()
        return

    # Slash commands
    if message.strip().startswith("/"):
        command_result = commands.handle(message, history, ticket_id)
        yield (*command_result, *_scheduler_hidden_updates())
        return
    
    # Current ticket category (None if no ticket yet)
    active_category: str | None = None
    if ticket_id:
        t = get_ticket(ticket_id)
        active_category = t["category"] if t else None

    # Guardrails
    verdict = guardrails.check(message, active_category)
    if verdict:
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": verdict["reply"]},
        ]
        yield history, ticket_id, "", gr.update(), "", "", *_scheduler_hidden_updates()
        return

    low_info = guardrails.low_information_reply(message, active_category)
    if low_info:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": low_info["reply"]},
        ]
        yield history, ticket_id, "", gr.update(), ticket_info_md(ticket_id) if ticket_id else "", "", *_scheduler_hidden_updates()
        return

    if scheduler.wants_call_scheduling(message):
        reply = (
            "Please enter your email address and choose a time slot below. "
            "I’ll prepare the confirmation email and send it automatically if email delivery is configured."
        )
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        if ticket_id:
            add_message(ticket_id, "user", message)
            add_message(ticket_id, "assistant", reply)
        yield history, ticket_id, "", gr.update(), ticket_info_md(ticket_id) if ticket_id else "", "", *_scheduler_visible_updates()
        return
    
    # Follow-up detection
    if guardrails.is_followup(message, active_category):
        yield from _handle_followup(message, history, ticket_id, use_cache)
        return

    turn_decision = classify_turn_intent(
        message=message,
        ticket_category=active_category,
        last_user_message=get_last_user_message(ticket_id) if ticket_id else "",
    )
    category = turn_decision["category"]
    confidence = turn_decision["confidence"]

    if ticket_id:
        if turn_decision["intent"] != "same_issue":
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": turn_decision["reply"]},
            ]
            yield history, ticket_id, "", gr.update(), ticket_info_md(ticket_id), "", *_scheduler_hidden_updates()
            return
        category = active_category or category

    # Create ticket if none exists
    if not ticket_id:
        ticket_id = make_ticket_id()
        create_ticket(ticket_id, category, message[:120])

        color   = CATEGORY_COLORS.get(category, "#6b7280")
        sys_msg = (
            f"🎫 **Ticket `{ticket_id}` opened** &nbsp;·&nbsp; "
            f"**{icon(category)} {category}** &nbsp;·&nbsp; "
            f"confidence **{confidence:.0%}**"
        )
        history = history + [{"role": "assistant", "content": sys_msg}]
        add_message(ticket_id, "assistant", sys_msg)
 
        t_info = (
            f"{icon(category)} **{category}** &nbsp;|&nbsp; "
            f"🟢 Open &nbsp;|&nbsp; `{ticket_id}` &nbsp;|&nbsp; "
            f"{datetime.now().strftime('%Y-%m-%d')}"
        )
        yield history, ticket_id, "", _refresh_list(), t_info, "", *_scheduler_hidden_updates()

    # Thinking indicator
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "⏳ Generating response…"},
    ]
    yield history, ticket_id, "", gr.update(), "", "", *_scheduler_hidden_updates()

    # Context from prior turns
    ctx   = get_prior_context(ticket_id, max_turns=3)
    query = f"{message}{ctx}"

    # Run pipeline
    result    = run(query, use_cache=use_cache, forced_category=category)
    draft     = get_final_draft(result)
    chunks    = result.get("chunks", [])
    cache_hit = result.get("cache_hit", False)

    if not cache_hit and _is_low_relevance_result(chunks):
        draft = (
            "I’m not confident the retrieved knowledge matches this question closely enough to answer reliably. "
            "Please rephrase the issue with a bit more product-specific detail, or use **`/new`** if this is a different problem."
        )
        chunks = []
 
    if not draft:
        draft = "I was unable to generate a response. Please try rephrasing or contact a human agent."
    if cache_hit:
        draft += "\n\n⚡ *Answered from cache*"

    # Persist
    add_message(ticket_id, "user",      message)
    add_message(ticket_id, "assistant", draft, chunks=chunks)

    # Update history
    history[-2] = {"role": "user",      "content": message}
    history[-1] = {"role": "assistant", "content": draft}

    yield (
        history,
        ticket_id,
        "",
        _refresh_list(),
        ticket_info_md(ticket_id),
        chunks_renderer.render(chunks),
        *_scheduler_hidden_updates(),
    )

# === Ticket list actions ===
 
def select_ticket(ticket_id: str):
    """Load an existing ticket's history into the chatbot."""
    if not ticket_id:
        return [], ""
    t    = get_ticket(ticket_id)
    msgs = get_messages(ticket_id)
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    return history, ticket_info_md(ticket_id)
 
def resolve_ticket(ticket_id: str | None):
    if not ticket_id:
        return gr.update(), "No active ticket."
    close_ticket(ticket_id)
    return _refresh_list(), "✅ Ticket resolved."
 
def new_ticket():
    return [], None, "", ""


def schedule_call(history: list, ticket_id: str | None, email: str, slot: str):
    if not scheduler.is_valid_email(email):
        status = "Please enter a valid email address before scheduling the call."
        return history, ticket_id, _refresh_list(), ticket_info_md(ticket_id) if ticket_id else "", gr.update(visible=True), gr.update(value=email), gr.update(choices=scheduler.get_slot_options(), value=slot), status, gr.update(value="", visible=False)

    if not slot:
        status = "Please select a time slot for the meeting."
        return history, ticket_id, _refresh_list(), ticket_info_md(ticket_id) if ticket_id else "", gr.update(visible=True), gr.update(value=email), gr.update(choices=scheduler.get_slot_options(), value=None), status, gr.update(value="", visible=False)

    subject, body = scheduler.build_confirmation_email(ticket_id, slot)
    sent, delivery_status = scheduler.send_confirmation_email(email.strip(), subject, body)

    assistant_reply = (
        f"Meeting scheduled successfully for **{slot}**.\n\n"
        f"A confirmation email has been {'sent' if sent else 'prepared'} for **{email.strip()}**."
    )
    updated_history = history + [{"role": "assistant", "content": assistant_reply}]

    if ticket_id:
        add_message(ticket_id, "assistant", assistant_reply)

    return (
        updated_history,
        ticket_id,
        _refresh_list(),
        ticket_info_md(ticket_id) if ticket_id else "",
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(choices=scheduler.get_slot_options(), value=None),
        "",
        gr.update(value="", visible=False),
    )


def cancel_schedule():
    panel, email_box, slot_box, status, preview = _scheduler_hidden_updates()
    return panel, email_box, slot_box, status, preview
