import sys, os, uuid
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from src.agents.orchestrator import run, get_final_draft
from src.models.classifier import SupportClassifier
from src.db.ticket_store import (
    create_ticket, get_ticket, list_tickets,
    add_message, get_messages, close_ticket
)

clf = SupportClassifier()

CATEGORY_ICONS = {
    "Technical": "🔧",
    "Billing and Payments": "💳",
    "Product Inquiry": "📦",
    "Returns and Exchanges": "↩️",
    "Human Resources": "👤",
    "General Inquiry": "💬",
}

current_ticket_id = None


def icon(cat):
    return CATEGORY_ICONS.get(cat, "🎫")


def show_help():
    print("""
Commands:
/new         Start new ticket
/list        Show tickets
/switch ID   Switch ticket
/status      Show current ticket
/resolve     Resolve current ticket
/help        Show commands
/exit        Quit
""")


def create_new_ticket(message):
    global current_ticket_id

    pred = clf.predict(subject="", body=message, tags=None)
    category = pred["category"]
    confidence = pred["confidence"]

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    create_ticket(ticket_id, category, message[:120])

    add_message(ticket_id, "user", message)

    current_ticket_id = ticket_id

    print(f"\n🎫 New Ticket Created: {ticket_id}")
    print(f"{icon(category)} Category: {category}")
    print(f"Confidence: {confidence:.1%}\n")

    return category


def show_status():
    global current_ticket_id

    if not current_ticket_id:
        print("No active ticket.")
        return

    t = get_ticket(current_ticket_id)
    print(f"""
Ticket: {current_ticket_id}
Category: {t['category']}
Status: {t['status']}
Created: {t['created_at']}
""")


def show_tickets():
    tickets = list_tickets()
    if not tickets:
        print("No tickets found.")
        return

    for t in tickets:
        status = "✅" if t["status"] == "resolved" else "🟢"
        print(f"{status} {t['id']} | {t['category']} | {t['created_at'][:10]}")


def resolve_ticket():
    global current_ticket_id

    if not current_ticket_id:
        print("No active ticket.")
        return

    close_ticket(current_ticket_id)
    print(f"✅ Ticket {current_ticket_id} resolved.")


def switch_ticket(ticket_id):
    global current_ticket_id

    t = get_ticket(ticket_id)
    if not t:
        print("Ticket not found.")
        return

    current_ticket_id = ticket_id
    print(f"Switched to {ticket_id}")


def ask_bot(message):
    global current_ticket_id

    if not current_ticket_id:
        category = create_new_ticket(message)
    else:
        t = get_ticket(current_ticket_id)
        category = t["category"]

    # Build prior context
    prev = get_messages(current_ticket_id)
    context = ""

    if len(prev) > 2:
        prior = [m["content"] for m in prev if m["role"] == "user"][-3:]
        context = " | Prior context: " + " -> ".join(prior)

    result = run(
        message + context,
        use_cache=True,
        forced_category=category
    )

    reply = get_final_draft(result)
    if not reply:
        reply = "I couldn't generate a response."

    add_message(current_ticket_id, "user", message)
    add_message(current_ticket_id, "assistant", reply)

    print("\n🤖 Support Copilot:")
    print(reply)
    print()


def main():
    print("=" * 60)
    print(" Customer Support Copilot (Terminal Edition)")
    print("=" * 60)
    show_help()

    while True:
        try:
            msg = input("You > ").strip()

            if not msg:
                continue

            if msg == "/exit":
                break

            elif msg == "/help":
                show_help()

            elif msg == "/new":
                global current_ticket_id
                current_ticket_id = None
                print("Ready for new ticket.")

            elif msg == "/list":
                show_tickets()

            elif msg.startswith("/switch"):
                parts = msg.split()
                if len(parts) == 2:
                    switch_ticket(parts[1])

            elif msg == "/status":
                show_status()

            elif msg == "/resolve":
                resolve_ticket()

            else:
                ask_bot(msg)

        except KeyboardInterrupt:
            break

    print("\nGoodbye.")


if __name__ == "__main__":
    main()