# src/ui/layout.py
"""
Gradio layout builder.
 
build() returns a gr.Blocks demo ready to .launch().
All event handlers are imported from chat_handler.
"""
 
from __future__ import annotations
import gradio as gr
 
from src.ui.styles      import CUSTOM_CSS
from src.ui.chat_handler import (
    chat, select_ticket, resolve_ticket, new_ticket,
    schedule_call, cancel_schedule,
)
from src.db.ticket_store import list_tickets
from src.ui.ticket_helpers import ticket_label

EXAMPLES = [
    "Our cloud storage platform is completely down. Multiple users cannot access files.",
    "I was charged twice for my subscription this month. Duplicate charge on April 14th.",
    "The VPN disconnects every 20 minutes after the latest firmware update.",
    "I'd like to return a device purchased 3 weeks ago. Build quality was poor.",
    "Does your analytics platform integrate with Salesforce and HubSpot?",
    "Something is wrong and I need help urgently.",
]

def _initial_ticket_list():
    tickets = list_tickets()
    if not tickets:
        return gr.update(choices=[], value=None)
    return gr.update(choices=[(ticket_label(t), t["id"]) for t in tickets])

def build():
    with gr.Blocks(
        title="Support Copilot",
        fill_height=True,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="app-header">
          <span style="font-size:22px">⚙</span>
          <div>
            <h1>Support Copilot</h1>
            <p>AI-powered agent assistant</p>
          </div>
        </div>
        """)

        with gr.Row(equal_height=True):
            # Sidebar
            with gr.Column(scale=1, min_width=240, elem_id="sidebar-col"):
                new_btn = gr.Button("＋ New Ticket", variant="primary", elem_id="new-ticket-btn",)

                gr.HTML('<div style="font-size:10px;font-weight:600;letter-spacing:.08em;'
                        'text-transform:uppercase;color:var(--t3);padding:14px 2px 6px">Tickets</div>')

                ticket_list = gr.Radio(
                    choices=[], value=None,
                    label="", interactive=True,
                    elem_id="ticket-list",
                )

                resolve_btn  = gr.Button("✓ Mark Resolved", elem_id="resolve-btn")
                resolve_note = gr.Markdown("", elem_id="resolve-note")

            # Chat panel
            with gr.Column(scale=4, elem_id="chat-col"):

                ticket_info = gr.Markdown("Describe your issue below to open a new ticket.", elem_id="ticket-info")
                chatbot = gr.Chatbot(
                    value=[], height=520, show_label=False,
                    elem_id="chatbot", render_markdown=True,
                )

                # display the source chunks
                chunks_out = gr.HTML("", elem_id="chunks-out")

                with gr.Column(visible=False, elem_id="scheduler-panel") as scheduler_panel:
                    gr.Markdown(
                        "### Schedule A Support Call\n"
                        "Enter your email address and choose a time slot. "
                        "A confirmation email will be prepared and sent automatically if SMTP is configured.",
                        elem_id="scheduler-copy",
                    )
                    with gr.Row(equal_height=True, elem_id="scheduler-row"):
                        email_input = gr.Textbox(
                            placeholder="name@example.com",
                            show_label=True,
                            label="Email",
                            elem_id="scheduler-email",
                            scale=3,
                        )
                        slot_input = gr.Dropdown(
                            choices=[],
                            value=None,
                            show_label=True,
                            label="Time Slot",
                            elem_id="scheduler-slot",
                            scale=4,
                        )
                    with gr.Row(equal_height=True, elem_id="scheduler-actions"):
                        schedule_btn = gr.Button("Schedule Meeting", variant="primary", elem_id="schedule-btn")
                        cancel_schedule_btn = gr.Button("Cancel", elem_id="cancel-schedule-btn")
                    schedule_status = gr.Markdown("", elem_id="schedule-status")
                    email_preview = gr.Textbox(
                        value="",
                        show_label=True,
                        label="Confirmation Email",
                        lines=8,
                        interactive=False,
                        visible=False,
                        elem_id="email-preview",
                    )

                # Input row
                with gr.Row(equal_height=True, elem_id="composer-row"):
                    use_cache = gr.Checkbox(
                        value=True, label="⚡ Cache", scale=0,
                        min_width=90, elem_id="cache-check",
                    )
                    msg_input = gr.Textbox(
                        placeholder="Describe your issue… or type /help",
                        lines=1, max_lines=5, show_label=False,
                        scale=6, min_width=360, elem_id="msg-input", autofocus=True,
                    )
                    send_btn = gr.Button(
                        "Send →", variant="primary", scale=1,
                        min_width=110, elem_id="send-btn",
                    )

                with gr.Accordion("💡 Example queries", open=False, elem_id="examples-accordion"):
                    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=msg_input, label="",)

        # State
        current_ticket = gr.State(None)

        # Event wiring
        chat_inputs  = [msg_input, chatbot, current_ticket, use_cache]
        chat_outputs = [
            chatbot, current_ticket, msg_input, ticket_list, ticket_info, chunks_out,
            scheduler_panel, email_input, slot_input, schedule_status, email_preview,
        ]

        msg_input.submit(fn=chat, inputs=chat_inputs, outputs=chat_outputs)
        send_btn.click(fn=chat, inputs=chat_inputs, outputs=chat_outputs)

        new_btn.click(
            fn=new_ticket,
            outputs=[chatbot, current_ticket, msg_input, ticket_info],
        )
        new_btn.click(
            fn=cancel_schedule,
            outputs=[scheduler_panel, email_input, slot_input, schedule_status, email_preview],
        )

        ticket_list.change(fn=select_ticket,inputs=[ticket_list],outputs=[chatbot, ticket_info],)
        ticket_list.change(
            fn=lambda tid: tid,
            inputs=[ticket_list],
            outputs=[current_ticket],
        )
        ticket_list.change(
            fn=cancel_schedule,
            outputs=[scheduler_panel, email_input, slot_input, schedule_status, email_preview],
        )

        resolve_btn.click(
            fn=resolve_ticket,
            inputs=[current_ticket],
            outputs=[ticket_list, resolve_note],
        )
        resolve_btn.click(
            fn=cancel_schedule,
            outputs=[scheduler_panel, email_input, slot_input, schedule_status, email_preview],
        )

        schedule_btn.click(
            fn=schedule_call,
            inputs=[chatbot, current_ticket, email_input, slot_input],
            outputs=[
                chatbot, current_ticket, ticket_list, ticket_info,
                scheduler_panel, email_input, slot_input, schedule_status, email_preview,
            ],
        )
        cancel_schedule_btn.click(
            fn=cancel_schedule,
            outputs=[scheduler_panel, email_input, slot_input, schedule_status, email_preview],
        )

        demo.load(fn=_initial_ticket_list, outputs=[ticket_list])

    return demo
