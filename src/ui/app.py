# src/ui/app.py

import sys, os, json, uuid
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import gradio as gr
from datetime import datetime
from src.agents.orchestrator import run, get_final_draft
from src.models.classifier   import SupportClassifier
from src.db.ticket_store     import (
    create_ticket, get_ticket, list_tickets,
    add_message, get_messages, close_ticket
)

_clf = SupportClassifier()

CATEGORY_ICONS = {
    "Technical"          : "🔧",
    "Billing and Payments": "💳",
    "Product Inquiry"    : "📦",
    "Returns and Exchanges": "↩️",
    "Human Resources"    : "👤",
    "General Inquiry"    : "💬",
}

def _icon(cat: str) -> str:
    return CATEGORY_ICONS.get(cat, "🎫")

def _ticket_label(t: dict) -> str:
    icon   = _icon(t['category'])
    date   = t['created_at'][:10]
    status = "✅" if t['status'] == 'resolved' else "🟢"
    return f"{status} {icon} {t['id'][:8]} — {t['category']} ({date})"

def _category_matches(new_category: str, ticket_category: str) -> bool:
    """Check if a new query's category is compatible with the current ticket."""
    if new_category == ticket_category:
        return True
    # Allow related categories in the same ticket
    technical_group = {"Technical"}
    billing_group   = {"Billing and Payments"}
    product_group   = {"Product Inquiry", "Returns and Exchanges"}
    for group in [technical_group, billing_group, product_group]:
        if new_category in group and ticket_category in group:
            return True
    return False


def load_ticket_history(ticket_id: str) -> list:
    """Load message history for a ticket into Gradio chat format."""
    if not ticket_id:
        return []
    msgs = get_messages(ticket_id)
    history = []
    for m in msgs:
        history.append({"role": m["role"], "content": m["content"]})
    return history


def refresh_ticket_list():
    tickets = list_tickets()
    if not tickets:
        return gr.update(choices=[], value=None)
    choices = [(_ticket_label(t), t['id']) for t in tickets]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def new_ticket_click():
    """Reset UI to start a new ticket."""
    return [], None, "", gr.update(value="")


def select_ticket(ticket_id: str):
    """Load an existing ticket into the chat."""
    if not ticket_id:
        return [], ""
    ticket  = get_ticket(ticket_id)
    history = load_ticket_history(ticket_id)
    info    = f"{_icon(ticket['category'])} **{ticket['category']}** | Status: {'✅ Resolved' if ticket['status'] == 'resolved' else '🟢 Open'} | Created: {ticket['created_at'][:10]}"
    return history, info


def close_current_ticket(ticket_id: str):
    if not ticket_id:
        return "No active ticket.", gr.update()
    close_ticket(ticket_id)
    tickets = list_tickets()
    choices = [(_ticket_label(t), t['id']) for t in tickets]
    return "✅ Ticket marked as resolved.", gr.update(choices=choices)


def chat(message: str, history: list, current_ticket_id: str, use_cache: bool):
    if not message.strip():
        yield history, current_ticket_id, "", gr.update(), ""
        return

    # Quick classify to check category
    clf_result = _clf.predict(subject='', body=message, tags=None)
    new_category = clf_result['category']
    confidence   = clf_result['confidence']

    # ── Case 1: No active ticket — create one ────────────────────────────────
    if not current_ticket_id:
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        summary   = message[:120]
        create_ticket(ticket_id, new_category, summary)
        current_ticket_id = ticket_id

        system_msg = (
            f"🎫 **New ticket created: `{ticket_id}`**\n"
            f"{_icon(new_category)} Category: **{new_category}** | "
            f"Confidence: **{confidence:.1%}**"
        )
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": system_msg}
        ]
        add_message(ticket_id, "user",      message)
        add_message(ticket_id, "assistant", system_msg)

        ticket_info = f"{_icon(new_category)} **{new_category}** | 🟢 Open | Created: {datetime.now().strftime('%Y-%m-%d')}"
        yield history, current_ticket_id, "", gr.update(), ticket_info

    # ── Case 2: Active ticket — check category match ──────────────────────────
    else:
        ticket = get_ticket(current_ticket_id)

        if not _category_matches(new_category, ticket['category']):
            mismatch_msg = (
                f"⚠️ This message appears to be about **{new_category}**, "
                f"but the current ticket `{current_ticket_id}` is for **{ticket['category']}**.\n\n"
                f"Please **create a new ticket** for this issue, or rephrase your message "
                f"if it's related to your current {ticket['category']} issue."
            )
            history = history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": mismatch_msg}
            ]
            yield history, current_ticket_id, "", gr.update(), ""
            return

    # ── Generate response using full pipeline ─────────────────────────────────
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "⏳ Thinking..."}
    ]
    yield history, current_ticket_id, "", gr.update(), ""

    # Build context from previous messages in this ticket
    prev_messages = get_messages(current_ticket_id)
    context_summary = ""
    if len(prev_messages) > 2:
        prior = [m for m in prev_messages if m['role'] == 'user'][-3:]
        context_summary = " | Prior context: " + " → ".join(m['content'][:50] for m in prior)

    result      = run(message + context_summary, use_cache=use_cache)
    draft       = get_final_draft(result)
    cache_tag   = " ⚡ *cached*" if result.get("cache_hit") else ""
    full_response = (draft or "I was unable to generate a response. Please try rephrasing.") + cache_tag

    add_message(current_ticket_id, "user",      message)
    add_message(current_ticket_id, "assistant", full_response)

    history[-2] = {"role": "user",      "content": message}
    history[-1] = {"role": "assistant", "content": full_response}

    tickets = list_tickets()
    choices = [(_ticket_label(t), t['id']) for t in tickets]

    ticket_info = ""
    if current_ticket_id:
        t = get_ticket(current_ticket_id)
        ticket_info = f"{_icon(t['category'])} **{t['category']}** | 🟢 Open | ID: `{current_ticket_id}`"

    yield history, current_ticket_id, "", gr.update(choices=choices), ticket_info


def build_ui():
    with gr.Blocks(title="Customer Support Copilot") as demo:

        gr.HTML("""
        <div style="text-align:center; padding:16px 0 8px 0">
            <h1 style="margin:0">🤖 Customer Support Copilot</h1>
            <p style="color:gray; margin:4px 0 0 0">
                Ticket-based AI assistant for support agents
            </p>
        </div>
        """)

        with gr.Row():
            # ── Left sidebar — ticket list ────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 🎫 Tickets")

                new_ticket_btn = gr.Button("+ New Ticket", variant="primary", size="sm")

                ticket_list = gr.Radio(
                    label="Select ticket",
                    choices=[],
                    value=None,
                    interactive=True
                )

                close_btn    = gr.Button("✅ Mark Resolved", size="sm")
                close_status = gr.Markdown("")

            # ── Right — chat area ─────────────────────────────────────────────
            with gr.Column(scale=3):
                ticket_info_box = gr.Markdown("")

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                    show_label=False,
                )

                with gr.Row():
                    use_cache = gr.Checkbox(label="⚡ Cache", value=True, scale=0)
                    msg_box   = gr.Textbox(
                        placeholder="Describe your issue or ask a follow-up question...",
                        show_label=False,
                        scale=5,
                        container=False
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                with gr.Accordion("💡 Example queries", open=False):
                    gr.Examples(
                        examples=[
                            ["Our cloud storage platform is completely down. Multiple users cannot access files."],
                            ["I was charged twice for my subscription this month. Duplicate charge on April 14th."],
                            ["The VPN disconnects every 20 minutes after the latest firmware update."],
                            ["I'd like to return a device purchased 3 weeks ago. Build quality was poor."],
                            ["Does your analytics platform integrate with Salesforce and HubSpot?"],
                            ["Something is wrong and I need help urgently."],
                        ],
                        inputs=msg_box,
                        label=""
                    )

        # ── State ─────────────────────────────────────────────────────────────
        current_ticket = gr.State(None)

        # ── Event wiring ──────────────────────────────────────────────────────
        submit_kwargs = dict(
            fn=chat,
            inputs=[msg_box, chatbot, current_ticket, use_cache],
            outputs=[chatbot, current_ticket, msg_box, ticket_list, ticket_info_box]
        )
        msg_box.submit(**submit_kwargs)
        send_btn.click(**submit_kwargs)

        new_ticket_btn.click(
            fn=new_ticket_click,
            outputs=[chatbot, current_ticket, msg_box, ticket_info_box]
        )

        ticket_list.change(
            fn=select_ticket,
            inputs=[ticket_list],
            outputs=[chatbot, ticket_info_box]
        )

        # Update current_ticket state when selecting from list
        ticket_list.change(
            fn=lambda tid: tid,
            inputs=[ticket_list],
            outputs=[current_ticket]
        )

        close_btn.click(
            fn=close_current_ticket,
            inputs=[current_ticket],
            outputs=[close_status, ticket_list]
        )

        # Load ticket list on startup
        demo.load(fn=refresh_ticket_list, outputs=[ticket_list])

    return demo


if __name__ == '__main__':
    demo = build_ui()
    demo.launch(
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )