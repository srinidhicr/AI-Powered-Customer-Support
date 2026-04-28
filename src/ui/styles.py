# src/ui/styles.py
"""All custom CSS for the Support Copilot Gradio UI."""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg0:#090b0f; --bg1:#0f1218; --bg2:#161920; --bg3:#1d2029;
    --br:#1e2433;  --ac:#5b7af5;  --ac-d:rgba(91,122,245,.13);
    --green:#34d399; --amber:#fbbf24;
    --t1:#dde2ef;  --t2:#7b849e;  --t3:#3e4560;
    --ff:'Inter',sans-serif; --fm:'JetBrains Mono',monospace;
}

body, .gradio-container {
    background: var(--bg0) !important;
    font-family: var(--ff) !important;
    color: var(--t1) !important;
}
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    min-height: 100vh !important;
}

/* Header */
#app-header {
    background: var(--bg1); border-bottom: 1px solid var(--br);
    padding: 14px 28px; display: flex; align-items: center; gap: 12px;
}
#app-header h1 { font-size: 16px; font-weight: 600; color: var(--t1); margin: 0; letter-spacing: -.3px; }
#app-header p  { font-size: 12px; color: var(--t3); margin: 0; }

/* Sidebar */
#sidebar-col {
    background: var(--bg1) !important; border-right: 1px solid var(--br) !important;
    border-radius: 0 !important; padding: 16px 12px !important;
    min-height: calc(100vh - 56px);
}
#sidebar-col .block { border: none !important; background: transparent !important; box-shadow: none !important; }

#new-ticket-btn {
    background: var(--ac) !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-size: 13px !important; font-weight: 500 !important;
    font-family: var(--ff) !important; padding: 8px 0 !important; width: 100% !important;
    cursor: pointer !important; transition: filter .15s !important;
}
#new-ticket-btn:hover { filter: brightness(1.1) !important; }

#resolve-btn {
    background: transparent !important; border: 1px solid var(--br) !important;
    color: var(--t2) !important; border-radius: 8px !important; font-size: 12px !important;
    font-family: var(--ff) !important; padding: 6px 0 !important; width: 100% !important;
    cursor: pointer !important; transition: all .12s !important; margin-top: 6px !important;
}
#resolve-btn:hover { border-color: var(--green) !important; color: var(--green) !important; }

#ticket-list { background: transparent !important; border: none !important; padding: 0 !important; }
#ticket-list .wrap { gap: 3px !important; }
#ticket-list label span {
    font-size: 12px !important; font-family: var(--fm) !important; color: var(--t1) !important;
    padding: 8px 10px !important; border-radius: 7px !important;
    border: 1px solid transparent !important; background: transparent !important;
    transition: all .12s !important; width: 100% !important; display: block !important;
}
#ticket-list label:hover span          { background: var(--bg2) !important; border-color: var(--br) !important; }
#ticket-list input[type=radio]:checked + span { background: var(--ac-d) !important; border-color: var(--ac) !important; }
#ticket-list .wrap input[type=radio]   { display: none !important; }

/* Chat panel */
#chat-col {
    background: var(--bg0) !important;
    padding: 0 !important;
    min-height: calc(100vh - 56px) !important;
    display: flex !important;
    flex-direction: column !important;
}
#chat-col .block { background: var(--bg0) !important; border: none !important; box-shadow: none !important; }

#chatbot {
    background: var(--bg0) !important;
    border: none !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
}
#chatbot .message-wrap { padding: 16px 24px !important; gap: 12px !important; }
#chatbot .user .message {
    background: var(--ac) !important;
    color: #fff !important;
    border-radius: 14px 14px 3px 14px !important;
    font-size: 13.5px !important;

    display: inline-block !important;   /* ✅ change */
    max-width: min(68%, 720px) !important;
    min-width: 60px !important;         /* ✅ add this */
    margin-left: auto !important;

    border: none !important;
    padding: 10px 14px !important;

    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
    word-break: normal !important;
}

#chatbot .bot .message {
    background: var(--bg2) !important;
    color: var(--t1) !important;
    border: 1px solid var(--br) !important;
    border-radius: 14px 14px 14px 3px !important;

    display: inline-block !important;   /* ✅ change */
    max-width: min(76%, 820px) !important;
    min-width: 60px !important;         /* ✅ add this */

    font-size: 13.5px !important;
    padding: 12px 16px !important;

    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
    word-break: normal !important;
}

#chatbot .user .message,
#chatbot .bot .message {
    display: inline-block !important;

    /* 👇 KEY FIXES */
    width: auto !important;          /* ensure natural sizing */
    min-width: 80px !important;      /* bump this up from 60 */
    max-width: 70% !important;       /* simplify (remove min()) */

    white-space: normal !important;  /* 👈 THIS matters */
    word-break: break-word !important;
}
#chatbot .message {
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
#chatbot .bot .message p       { margin-bottom: 6px !important; line-height: 1.65 !important; }
#chatbot .bot .message ul,
#chatbot .bot .message ol      { padding-left: 18px !important; margin: 5px 0 !important; }
#chatbot .bot .message li      { margin-bottom: 3px !important; }
#chatbot .bot .message strong  { color: #c8d0e8 !important; }
#chatbot .bot .message code    {
    font-family: var(--fm) !important; font-size: 12px !important;
    background: var(--bg3) !important; padding: 1px 5px !important; border-radius: 3px !important;
}
#chatbot .avatar-container { display: none !important; }

/* Input */
#composer-row {
    align-items: end !important;
    gap: 12px !important;
    flex-wrap: nowrap !important;
    margin: 12px 24px 0 !important;
    padding: 0 0 12px !important;
    position: sticky !important;
    bottom: 0 !important;
    background: linear-gradient(to top, rgba(9,11,15,1), rgba(9,11,15,0.94)) !important;
    z-index: 3 !important;
}
#composer-row > .gradio-checkbox,
#composer-row > .gradio-textbox,
#composer-row > .gradio-button {
    min-width: 0 !important;
}

#msg-input {
    flex: 1 1 auto !important;
}
#msg-input > .wrap,
#msg-input .wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#msg-input textarea {
    background: var(--bg2) !important; border: 1px solid var(--br) !important;
    border-radius: 10px !important; color: var(--t1) !important;
    font-family: var(--ff) !important; font-size: 13.5px !important;
    padding: 10px 14px !important; resize: none !important; transition: border-color .15s !important;
    min-height: 42px !important; line-height: 1.45 !important;
}
#msg-input textarea:focus           { border-color: var(--ac) !important; outline: none !important; }
#msg-input textarea::placeholder    { color: var(--t3) !important; }
#msg-input .label-wrap              { display: none !important; }

#send-btn {
    background: var(--ac) !important; color: #fff !important; border: none !important;
    border-radius: 10px !important; font-family: var(--ff) !important;
    font-size: 13px !important; font-weight: 500 !important; padding: 0 22px !important;
    height: 42px !important; cursor: pointer !important; transition: filter .15s !important;
    white-space: nowrap !important; align-self: flex-end !important;
}
#send-btn:hover { filter: brightness(1.1) !important; }

#cache-check {
    flex: 0 0 auto !important;
    background: transparent !important; border: none !important; padding: 0 !important;
}
#cache-check > .wrap,
#cache-check .wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-height: 42px !important;
    display: flex !important;
    align-items: center !important;
}
#cache-check label {
    font-size: 12px !important; color: var(--t2) !important;
    display: inline-flex !important; align-items: center !important; gap: 8px !important;
    white-space: nowrap !important;
}
#cache-check input[type=checkbox]   { accent-color: var(--ac) !important; }

#examples-accordion {
    background: transparent !important; border: 1px solid var(--br) !important;
    border-radius: 8px !important;
}
#examples-accordion .label-wrap span { font-size: 12px !important; color: var(--t2) !important; }

/* Chunks output */
#chunks-out { background: transparent !important; border: none !important; padding: 0 !important; }
#chunks-out .block {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
}

/* Scheduler */
#scheduler-panel {
    background: var(--bg1) !important;
    border: 1px solid var(--br) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 8px 24px 12px !important;
}
#scheduler-panel .block,
#scheduler-panel .wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#scheduler-copy p {
    margin: 0 0 10px !important;
    color: var(--t2) !important;
    line-height: 1.5 !important;
}
#scheduler-row,
#scheduler-actions {
    gap: 12px !important;
}
#scheduler-email textarea,
#scheduler-email input,
#scheduler-slot input,
#scheduler-slot .wrap,
#email-preview textarea {
    background: var(--bg2) !important;
    border: 1px solid var(--br) !important;
    color: var(--t1) !important;
    border-radius: 10px !important;
}
#scheduler-email label,
#scheduler-slot label,
#email-preview label {
    color: var(--t2) !important;
    font-size: 12px !important;
}
#schedule-btn {
    background: var(--ac) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
}
#cancel-schedule-btn {
    background: transparent !important;
    color: var(--t2) !important;
    border: 1px solid var(--br) !important;
    border-radius: 10px !important;
}
#schedule-status p {
    margin: 10px 0 0 !important;
    color: var(--t2) !important;
}

/* Scrollbars */
::-webkit-scrollbar       { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--br); border-radius: 2px; }

footer { display: none !important; }
"""
