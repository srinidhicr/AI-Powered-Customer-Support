# src/ui/app_nicegui.py
# Run: python src/ui/app_nicegui.py

import sys, os, uuid, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from nicegui import ui, app
from datetime import datetime
import asyncio, concurrent.futures

from src.agents.orchestrator import run, get_final_draft
from src.models.classifier   import SupportClassifier
from src.db.ticket_store     import (
    create_ticket, get_ticket, list_tickets,
    add_message, get_messages, close_ticket
)

_clf      = SupportClassifier()
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORY_ICONS = {
    "Technical"            : "🔧",
    "Billing and Payments" : "💳",
    "Product Inquiry"      : "📦",
    "Returns and Exchanges": "↩️",
    "Human Resources"      : "👤",
    "General Inquiry"      : "💬",
}

EXAMPLES = [
    "Our cloud storage platform is down. Multiple users cannot access files.",
    "I was charged twice for my subscription this month.",
    "VPN disconnects every 20 minutes after the latest firmware update.",
    "I'd like to return a device. Build quality was poor.",
    "Does your platform integrate with Salesforce and HubSpot?",
]

NOISE = {"hi","hello","hey","thanks","ok","okay","bye","yes","no","sure","test"}

def _icon(cat: str) -> str:
    return CATEGORY_ICONS.get(cat, "🎫")

def _category_matches(a: str, b: str) -> bool:
    if a == b:
        return True
    groups = [
        {"Technical"},
        {"Billing and Payments"},
        {"Product Inquiry", "Returns and Exchanges"},
    ]
    return any(a in g and b in g for g in groups)

def _scroll_to_bottom():
    ui.run_javascript(
        'var el = document.querySelector(".q-scrollarea__container"); '
        'if(el) el.scrollTop = el.scrollHeight;'
    )

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, body { font-family: 'Inter', sans-serif !important; }
body    { background: #0d1117 !important; margin: 0; overflow: hidden; }

/* Sidebar */
.sidebar {
    width: 260px; min-width: 260px;
    background: #161b22;
    border-right: 1px solid #21262d;
    height: 100vh; overflow-y: auto;
    display: flex; flex-direction: column;
}
.sidebar-header { padding: 20px 16px 12px; border-bottom: 1px solid #21262d; }
.logo-title  { color: #e6edf3; font-size: 16px; font-weight: 600; }
.logo-sub    { color: #7d8590; font-size: 11px; margin-top: 2px; }
.sidebar-section { padding: 8px 12px 4px; color: #7d8590; font-size: 10px;
                   font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }

/* Ticket cards */
.tkt-card {
    margin: 3px 8px; padding: 10px 12px;
    background: #1c2128; border: 1px solid #21262d;
    border-radius: 8px; cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
}
.tkt-card:hover  { border-color: #388bfd; background: #1f2937; }
.tkt-card.active { border-color: #6366f1; background: #1f2937; }
.tkt-id    { color: #e6edf3; font-size: 11px; font-family: monospace; font-weight: 600; }
.tkt-cat   { color: #7d8590; font-size: 11px; margin-top: 1px; }
.tkt-badge { font-size: 10px; padding: 1px 6px; border-radius: 999px; font-weight: 600; }
.badge-open     { background: #1f3a1f; color: #56d364; border: 1px solid #2ea043; }
.badge-resolved { background: #1a2433; color: #7d8590; border: 1px solid #30363d; }

/* Chat panel */
.chat-panel {
    flex: 1; background: #0d1117;
    height: 100vh; display: flex; flex-direction: column;
    overflow: hidden;
}
.chat-header {
    background: #161b22; border-bottom: 1px solid #21262d;
    padding: 14px 20px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: space-between;
}
.header-title { color: #e6edf3; font-size: 14px; font-weight: 500; }
.header-meta  { color: #7d8590; font-size: 12px; margin-top: 2px; }

/* Messages */
.chat-scroll { flex: 1; overflow-y: auto; padding: 16px 20px; }

.msg-row-user   { display: flex; justify-content: flex-end; margin-bottom: 12px; }
.msg-row-bot    { display: flex; justify-content: flex-start; margin-bottom: 12px; }
.msg-row-system { display: flex; justify-content: center; margin-bottom: 10px; }

.msg-user {
    background: #6366f1; color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px; max-width: 68%;
    font-size: 14px; line-height: 1.6;
    word-wrap: break-word;
}
.msg-bot {
    background: #161b22; color: #e6edf3;
    border: 1px solid #21262d;
    border-radius: 4px 16px 16px 16px;
    padding: 10px 14px; max-width: 72%;
    font-size: 14px; line-height: 1.6;
    word-wrap: break-word;
}
.msg-system {
    background: #0d2318; color: #56d364;
    border: 1px solid #1a4731;
    border-radius: 8px; padding: 6px 14px;
    font-size: 12px; text-align: center;
}
.msg-warning {
    background: #261d0d; color: #d29922;
    border: 1px solid #4d3800;
    border-radius: 8px; padding: 8px 14px;
    font-size: 13px; max-width: 80%; margin: 0 auto 10px;
}
.msg-resolved {
    background: #1c1c1c; color: #6e7681;
    border: 1px solid #21262d;
    border-radius: 8px; padding: 8px 14px;
    font-size: 12px; text-align: center;
    margin-bottom: 10px;
}

/* Source chunks panel */
.chunks-panel {
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; margin-top: 6px; padding: 0;
    max-width: 72%; font-size: 12px;
}
.chunks-toggle {
    color: #388bfd; font-size: 11px; cursor: pointer;
    padding: 4px 8px; display: inline-block;
}
.chunk-item {
    background: #161b22; border-top: 1px solid #21262d;
    padding: 8px 12px; color: #7d8590; font-size: 12px; line-height: 1.5;
}
.chunk-score { color: #56d364; font-weight: 600; font-size: 11px; }
.chunk-subject { color: #388bfd; font-weight: 500; margin-bottom: 4px; }

/* Input area */
.input-area {
    background: #161b22; border-top: 1px solid #21262d;
    padding: 12px 20px; flex-shrink: 0;
}
.input-area textarea {
    background: #1c2128 !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 10px !important;
    font-size: 14px !important; padding: 10px 14px !important;
    min-height: 52px !important; resize: none;
}
.input-area textarea:focus {
    border-color: #6366f1 !important; outline: none !important;
}
.resolved-banner {
    background: #161b22; border-top: 1px solid #21262d;
    padding: 14px 20px; text-align: center;
    color: #7d8590; font-size: 13px; flex-shrink: 0;
}

/* Buttons */
.btn-primary {
    background: #6366f1 !important; color: white !important;
    border-radius: 8px !important; font-weight: 500 !important;
    font-size: 13px !important;
}
.btn-primary:hover { background: #4f52d0 !important; }
.btn-ghost {
    background: transparent !important; color: #7d8590 !important;
    border: 1px solid #21262d !important; border-radius: 8px !important;
    font-size: 12px !important;
}
.btn-ghost:hover { border-color: #388bfd !important; color: #388bfd !important; }
.btn-danger {
    background: transparent !important; color: #f85149 !important;
    border: 1px solid #21262d !important; border-radius: 8px !important;
    font-size: 12px !important;
}
.btn-danger:hover { border-color: #f85149 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 4px; }
"""

# ── Page ──────────────────────────────────────────────────────────────────────

@ui.page('/')
def main_page():

    state = {
        "ticket_id": None,
        "use_cache": True,
        "is_resolved": False,
    }

    # Containers that get rebuilt — stored in dicts so inner functions can mutate
    refs = {
        "chat_inner"    : None,   # ui.column inside scroll
        "header_title"  : None,   # label
        "header_meta"   : None,   # label
        "input_area"    : None,   # the whole input row
        "resolved_banner": None,  # shown when resolved
        "msg_input"     : None,   # textarea
        "sidebar_list"  : None,   # ui.column for ticket cards
        "send_btn"      : None,
    }

    # Stored per-message source chunks: msg_index → list of chunk dicts
    msg_sources = {}

    ui.add_head_html(f"<style>{CSS}</style>")

    # ── Layout ────────────────────────────────────────────────────────────────
    with ui.element('div').style('display:flex; width:100vw; height:100vh; overflow:hidden;'):

        # ── Sidebar ───────────────────────────────────────────────────────────
        with ui.element('div').classes('sidebar'):
            with ui.element('div').classes('sidebar-header'):
                ui.element('div').classes('logo-title').style('').set_text('')
                with ui.element('div').classes('logo-title'):
                    ui.label('🤖 Support Copilot')
                with ui.element('div').classes('logo-sub'):
                    ui.label('AI-powered ticket assistant')

            with ui.element('div').style('padding:12px;'):
                ui.button('+ New Ticket', on_click=lambda: action_new_ticket()).classes(
                    'btn-primary w-full'
                )

            with ui.element('div').classes('sidebar-section'):
                ui.label('TICKETS')

            sidebar_list = ui.column().style('padding: 0 4px; gap: 0;')
            refs['sidebar_list'] = sidebar_list

        # ── Chat panel ────────────────────────────────────────────────────────
        with ui.element('div').classes('chat-panel'):

            # Header
            with ui.element('div').classes('chat-header'):
                with ui.column().style('gap:2px;'):
                    header_title = ui.label('No active ticket').style(
                        'color:#e6edf3; font-size:14px; font-weight:500;'
                    )
                    header_meta = ui.label('Select a ticket or create a new one').style(
                        'color:#7d8590; font-size:12px;'
                    )
                    refs['header_title'] = header_title
                    refs['header_meta']  = header_meta

                with ui.row().style('gap:8px; align-items:center;'):
                    cache_chk = ui.checkbox('⚡ Cache', value=True).style(
                        'color:#7d8590; font-size:12px;'
                    )
                    cache_chk.on('update:model-value', lambda e: state.update({'use_cache': e.args}))

                    resolve_btn = ui.button('✅ Resolve', on_click=lambda: action_resolve()).classes('btn-ghost')
                    refs['resolve_btn'] = resolve_btn

            # Scroll area for chat
            with ui.element('div').classes('chat-scroll') as chat_scroll_el:
                refs['chat_scroll'] = chat_scroll_el
                chat_inner = ui.column().style('gap:0; width:100%;')
                refs['chat_inner'] = chat_inner

                with chat_inner:
                    _welcome_msg()

            # Input area (swapped out when resolved)
            input_area = ui.element('div').classes('input-area')
            refs['input_area'] = input_area
            resolved_banner = ui.element('div').classes('resolved-banner').style('display:none;')
            refs['resolved_banner'] = resolved_banner

            with resolved_banner:
                ui.label('🔒 This ticket is resolved. Create a new ticket to continue.')

            with input_area:
                _build_input_area(refs, state, msg_sources)

            # Examples
            with ui.element('div').style(
                'background:#161b22; border-top:1px solid #21262d; '
                'padding:8px 20px; display:flex; gap:8px; flex-wrap:wrap; align-items:center;'
            ):
                ui.label('Examples:').style('color:#7d8590; font-size:11px;')
                for ex in EXAMPLES:
                    short = ex[:42] + '…' if len(ex) > 42 else ex
                    def _make(text=ex):
                        def fn():
                            refs['msg_input'].set_value(text)
                        return fn
                    ui.button(short, on_click=_make()).style(
                        'background:transparent; color:#7d8590; border:1px solid #21262d; '
                        'border-radius:999px; font-size:11px; padding:2px 10px; cursor:pointer;'
                    )

    # ── Helper UI builders ────────────────────────────────────────────────────

    def _welcome_msg():
        with refs['chat_inner']:
            with ui.element('div').classes('msg-row-system'):
                with ui.element('div').classes('msg-system'):
                    ui.label('👋 Welcome! Describe your issue to create a support ticket.')

    def _build_input_area(refs, state, msg_sources):
        with ui.row().style('gap:10px; align-items:flex-end; width:100%;'):
            msg_input = ui.textarea(placeholder='Describe your issue… or type /help').style(
                'flex:1; background:#1c2128; color:#e6edf3; border:1px solid #30363d; '
                'border-radius:10px; font-size:14px; padding:10px 14px; '
                'min-height:60px; max-height:120px; resize:none; outline:none;'
            ).props('autogrow outlined dark dense')
            refs['msg_input'] = msg_input

            send_btn = ui.button('Send ➤', on_click=lambda: asyncio.ensure_future(
                action_send(refs, state, msg_sources)
            )).classes('btn-primary').style('padding:10px 20px; align-self:flex-end;')
            refs['send_btn'] = send_btn

        msg_input.on('keydown.ctrl.enter', lambda: asyncio.ensure_future(
            action_send(refs, state, msg_sources)
        ))

    # ── Core actions ──────────────────────────────────────────────────────────

    def action_new_ticket():
        state['ticket_id']  = None
        state['is_resolved'] = False
        msg_sources.clear()

        refs['chat_inner'].clear()
        _welcome_msg()

        refs['header_title'].set_text('New Ticket')
        refs['header_meta'].set_text('Describe your issue below')

        _set_resolved_ui(False)
        _rebuild_sidebar()

    def action_resolve():
        tid = state['ticket_id']
        if not tid:
            ui.notify('No active ticket', type='warning')
            return
        close_ticket(tid)
        state['is_resolved'] = True

        with refs['chat_inner']:
            with ui.element('div').classes('msg-row-system'):
                with ui.element('div').classes('msg-resolved'):
                    ui.label(f'🔒 Ticket {tid} has been marked as resolved. No further messages can be sent.')

        _set_resolved_ui(True)
        refs['header_meta'].set_text('Status: ✅ Resolved')
        _rebuild_sidebar()
        ui.notify('Ticket resolved', type='positive')
        _scroll_to_bottom()

    def _set_resolved_ui(is_resolved: bool):
        if is_resolved:
            refs['input_area'].style('display:none;')
            refs['resolved_banner'].style('display:block;')
        else:
            refs['input_area'].style('display:flex; flex-direction:column;')
            refs['resolved_banner'].style('display:none;')

    def _load_ticket(ticket_id: str):
        ticket = get_ticket(ticket_id)
        if not ticket:
            return

        state['ticket_id']   = ticket_id
        state['is_resolved'] = ticket['status'] == 'resolved'
        msg_sources.clear()

        refs['chat_inner'].clear()

        msgs = get_messages(ticket_id)
        for m in msgs:
            mtype = 'user' if m['role'] == 'user' else 'bot'
            if m['content'].startswith('🎫'):
                mtype = 'system'
            _render_message(m['content'], mtype, chunks=None)

        refs['header_title'].set_text(
            f"{_icon(ticket['category'])} {ticket['category']} — {ticket_id}"
        )
        refs['header_meta'].set_text(
            f"Status: {'✅ Resolved' if ticket['status'] == 'resolved' else '🟢 Open'} | "
            f"Created: {ticket['created_at'][:10]}"
        )
        _set_resolved_ui(state['is_resolved'])
        _rebuild_sidebar()
        _scroll_to_bottom()

    def _rebuild_sidebar():
        refs['sidebar_list'].clear()
        tickets = list_tickets()
        with refs['sidebar_list']:
            if not tickets:
                ui.label('No tickets yet').style(
                    'color:#7d8590; font-size:12px; text-align:center; padding:20px;'
                )
                return
            for t in tickets:
                is_active  = t['id'] == state['ticket_id']
                active_cls = 'tkt-card active' if is_active else 'tkt-card'
                badge_cls  = 'tkt-badge badge-resolved' if t['status'] == 'resolved' else 'tkt-badge badge-open'
                badge_txt  = '✅ Resolved' if t['status'] == 'resolved' else '🟢 Open'

                def _make_select(tid=t['id']):
                    def fn():
                        _load_ticket(tid)
                    return fn

                with ui.element('div').classes(active_cls).on('click', _make_select()):
                    with ui.element('div').style('display:flex; justify-content:space-between; align-items:flex-start;'):
                        with ui.column().style('gap:2px;'):
                            ui.element('div').classes('tkt-id').set_text(
                                f"{_icon(t['category'])} {t['id']}"
                            )
                            ui.element('div').classes('tkt-cat').set_text(t['category'])
                            ui.element('div').style('color:#7d8590; font-size:10px;').set_text(
                                t['created_at'][:10]
                            )
                        ui.element('span').classes(badge_cls).set_text(badge_txt)

    def _render_message(text: str, msg_type: str, chunks=None):
        """Render a chat message with optional source chunks panel."""
        row_class = {
            'user'   : 'msg-row-user',
            'bot'    : 'msg-row-bot',
            'system' : 'msg-row-system',
            'warning': 'msg-row-system',
        }.get(msg_type, 'msg-row-bot')

        bubble_class = {
            'user'   : 'msg-user',
            'bot'    : 'msg-bot',
            'system' : 'msg-system',
            'warning': 'msg-warning',
        }.get(msg_type, 'msg-bot')

        with refs['chat_inner']:
            with ui.element('div').classes(row_class):
                with ui.column().style('gap:4px; max-width:72%;' if msg_type == 'bot' else 'gap:4px;'):
                    with ui.element('div').classes(bubble_class):
                        ui.markdown(text).style('font-size:14px; line-height:1.6;')

                    # Source chunks button — only for bot messages with chunks
                    if msg_type == 'bot' and chunks:
                        chunks_visible = {'v': False}
                        chunks_panel = ui.element('div').classes('chunks-panel').style('display:none;')

                        def _make_toggle(panel=chunks_panel, vis=chunks_visible):
                            def fn():
                                vis['v'] = not vis['v']
                                panel.style(
                                    'display:block;' if vis['v'] else 'display:none;'
                                )
                            return fn

                        toggle_btn = ui.button(
                            f'📎 View {len(chunks)} source chunks ▾',
                            on_click=_make_toggle()
                        ).style(
                            'background:transparent; color:#388bfd; font-size:11px; '
                            'padding:2px 0; cursor:pointer; border:none; text-align:left;'
                        )

                        with chunks_panel:
                            for i, chunk in enumerate(chunks[:5]):
                                with ui.element('div').classes('chunk-item'):
                                    subj  = chunk.get('subject', '') or chunk.get('id', f'Doc {i+1}')
                                    score = chunk.get('score', 0)
                                    text_ = chunk.get('answer', chunk.get('text', ''))[:200]
                                    with ui.element('div').classes('chunk-subject'):
                                        ui.label(f'📄 {subj}')
                                    with ui.element('div').classes('chunk-score'):
                                        ui.label(f'Relevance: {score:.3f}')
                                    ui.element('div').style(
                                        'color:#8b949e; font-size:11px; margin-top:4px; line-height:1.5;'
                                    ).set_text(text_ + '…')

    # ── Send message ──────────────────────────────────────────────────────────

    async def action_send(refs, state, msg_sources):
        if state['is_resolved']:
            ui.notify('This ticket is resolved. Create a new ticket.', type='warning')
            return

        msg_input = refs['msg_input']
        message   = msg_input.value.strip()
        if not message:
            return

        msg_input.set_value('')

        # Slash commands
        if message.startswith('/'):
            _handle_slash(message)
            return

        # Noise
        if message.lower() in NOISE:
            _render_message(message, 'user')
            _render_message('👋 Hello! Please describe your issue to create a support ticket.', 'bot')
            _scroll_to_bottom()
            return

        # Classify
        if state['ticket_id']:
            ticket = get_ticket(state['ticket_id'])
            if len(message.split()) <= 8:
                new_cat    = ticket['category']
                confidence = 1.0
            else:
                clf        = _clf.predict(subject='', body=message, tags=None)
                new_cat    = clf['category']
                confidence = clf['confidence']
        else:
            clf        = _clf.predict(subject='', body=message, tags=None)
            new_cat    = clf['category']
            confidence = clf['confidence']

        # Mismatch check
        if state['ticket_id'] and confidence >= 0.60:
            ticket = get_ticket(state['ticket_id'])
            if not _category_matches(new_cat, ticket['category']):
                _render_message(message, 'user')
                _render_message(
                    f"⚠️ This looks like a **{new_cat}** issue, but ticket "
                    f"`{state['ticket_id']}` is for **{ticket['category']}**.\n\n"
                    f"Click **+ New Ticket** to open a separate ticket for this issue.",
                    'warning'
                )
                _scroll_to_bottom()
                return

        # Create ticket if none
        if not state['ticket_id']:
            ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
            create_ticket(ticket_id, new_cat, message[:120])
            state['ticket_id'] = ticket_id

            sys_msg = (
                f"🎫 **Ticket created: `{ticket_id}`**  \n"
                f"{_icon(new_cat)} Category: **{new_cat}** | Confidence: {confidence:.1%}"
            )
            _render_message(sys_msg, 'system')
            add_message(ticket_id, 'assistant', sys_msg)

            refs['header_title'].set_text(f"{_icon(new_cat)} {new_cat} — {ticket_id}")
            refs['header_meta'].set_text(f"Status: 🟢 Open | Created: {datetime.now().strftime('%Y-%m-%d')}")
            _rebuild_sidebar()

        # Render user message
        _render_message(message, 'user')
        add_message(state['ticket_id'], 'user', message)

        # Thinking bubble
        with refs['chat_inner']:
            thinking_row = ui.element('div').classes('msg-row-bot')
            with thinking_row:
                with ui.element('div').classes('msg-bot').style('opacity:0.5;'):
                    ui.label('⏳ Thinking…')
        _scroll_to_bottom()

        # Build context from ticket history
        prev    = get_messages(state['ticket_id'])
        context = ''
        if len(prev) > 2:
            prior   = [m for m in prev if m['role'] == 'user'][-3:]
            context = ' | Prior context: ' + ' → '.join(m['content'][:60] for m in prior)

        # Run pipeline in thread
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: run(
                f"[KNOWN CATEGORY: {new_cat}] {message}{context}",
                use_cache=state['use_cache'],
                forced_category=new_cat
            )
        )

        draft     = get_final_draft(result)
        cache_tag = '  ⚡ *cached*' if result.get('cache_hit') else ''
        response  = (draft or 'I was unable to generate a response. Please try rephrasing.') + cache_tag

        # Extract source chunks from result messages
        chunks = _extract_chunks(result)

        # Remove thinking bubble
        thinking_row.delete()

        # Render response with source chunks
        _render_message(response, 'bot', chunks=chunks)
        add_message(state['ticket_id'], 'assistant', response)

        _rebuild_sidebar()
        _scroll_to_bottom()

    def _extract_chunks(result: dict) -> list:
        """Pull retrieved documents from the agent's tool messages."""
        chunks = []
        for msg in result.get('messages', []):
            content = getattr(msg, 'content', '')
            if isinstance(content, str) and '"documents"' in content:
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and 'documents' in data:
                        chunks = data['documents']
                        break
                    elif isinstance(data, list):
                        chunks = data
                        break
                except Exception:
                    pass
        return chunks

    def _handle_slash(cmd: str):
        cmd = cmd.strip().lower()
        _render_message(cmd, 'user')

        if cmd == '/help':
            _render_message(
                "**Available commands:**\n\n"
                "- `/new` — Start a new ticket\n"
                "- `/resolve` — Mark current ticket as resolved\n"
                "- `/status` — Show current ticket info\n"
                "- `/help` — Show this menu",
                'bot'
            )
        elif cmd == '/new':
            action_new_ticket()
        elif cmd == '/resolve':
            action_resolve()
        elif cmd == '/status':
            if not state['ticket_id']:
                _render_message('No active ticket selected.', 'bot')
            else:
                t = get_ticket(state['ticket_id'])
                _render_message(
                    f"**Ticket:** `{t['id']}`  \n"
                    f"**Category:** {_icon(t['category'])} {t['category']}  \n"
                    f"**Status:** {t['status']}  \n"
                    f"**Created:** {t['created_at'][:10]}",
                    'bot'
                )
        else:
            _render_message('Unknown command. Type `/help` for available commands.', 'bot')

        _scroll_to_bottom()

    # ── Init ──────────────────────────────────────────────────────────────────
    _rebuild_sidebar()


if __name__ == '__main__':
    ui.run(
        title='Customer Support Copilot',
        port=8080,
        dark=True,
        reload=False,
        favicon='🤖',
        storage_secret='support-copilot-secret'
    )
