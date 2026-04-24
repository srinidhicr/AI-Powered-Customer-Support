from __future__ import annotations

import os
import re
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage


_SCHEDULE_PATTERNS: tuple[str, ...] = (
    "schedule a call",
    "book a call",
    "arrange a call",
    "set up a call",
    "schedule meeting",
    "schedule a meeting",
    "book a meeting",
    "arrange a meeting",
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def wants_call_scheduling(message: str) -> bool:
    lower = message.strip().lower()
    return any(pattern in lower for pattern in _SCHEDULE_PATTERNS)


def get_slot_options(now: datetime | None = None) -> list[str]:
    now = now or datetime.now()
    slot_times = [(10, 0), (11, 30), (14, 0), (16, 30)]
    options: list[str] = []

    day = now
    while len(options) < 8:
        day += timedelta(days=1)
        if day.weekday() >= 5:
            continue
        for hour, minute in slot_times:
            start = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            end = start + timedelta(minutes=30)
            options.append(f"{start.strftime('%a, %d %b · %I:%M %p')} - {end.strftime('%I:%M %p')}")
            if len(options) >= 8:
                break
    return options


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match((email or "").strip()))


def build_confirmation_email(ticket_id: str | None, slot: str) -> tuple[str, str]:
    ticket_ref = ticket_id or "your support request"
    subject = f"Meeting scheduled successfully for {ticket_ref}"
    body = (
        "Hello,\n\n"
        "Your support meeting has been scheduled successfully.\n\n"
        f"Ticket: {ticket_ref}\n"
        f"Selected slot: {slot}\n\n"
        "If you need to reschedule, please reply to this email.\n\n"
        "Best regards,\n"
        "Support Copilot"
    )
    return subject, body


def send_confirmation_email(recipient: str, subject: str, body: str) -> tuple[bool, str]:
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587") or "587")
    username = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    sender = os.getenv("SMTP_FROM", username).strip()

    if not host or not sender:
        return False, "SMTP is not configured in the environment, so I composed the email draft but did not send it."

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(msg)
        return True, f"Confirmation email sent successfully to `{recipient}`."
    except Exception as exc:
        return False, f"Could not send the email automatically: {exc}"
