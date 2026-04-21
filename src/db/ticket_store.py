# src/db/ticket_store.py

import sqlite3, os, json
from datetime import datetime

DB_PATH = 'data/tickets.db'

def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id          TEXT PRIMARY KEY,
                category    TEXT,
                status      TEXT DEFAULT 'open',
                created_at  TEXT,
                summary     TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id   TEXT,
                role        TEXT,
                content     TEXT,
                created_at  TEXT,
                FOREIGN KEY (ticket_id) REFERENCES tickets(id)
            )
        """)

def create_ticket(ticket_id: str, category: str, summary: str):
    with _conn() as c:
        c.execute(
            "INSERT INTO tickets (id, category, status, created_at, summary) VALUES (?,?,?,?,?)",
            (ticket_id, category, 'open', datetime.now().isoformat(), summary)
        )

def get_ticket(ticket_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT id, category, status, created_at, summary FROM tickets WHERE id=?",
            (ticket_id,)
        ).fetchone()
    if not row:
        return None
    return {"id": row[0], "category": row[1], "status": row[2],
            "created_at": row[3], "summary": row[4]}

def list_tickets() -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, category, status, created_at, summary FROM tickets ORDER BY created_at DESC"
        ).fetchall()
    return [{"id": r[0], "category": r[1], "status": r[2],
             "created_at": r[3], "summary": r[4]} for r in rows]

def add_message(ticket_id: str, role: str, content: str):
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (ticket_id, role, content, created_at) VALUES (?,?,?,?)",
            (ticket_id, role, content, datetime.now().isoformat())
        )

def get_messages(ticket_id: str) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content, created_at FROM messages WHERE ticket_id=? ORDER BY id",
            (ticket_id,)
        ).fetchall()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]

def close_ticket(ticket_id: str):
    with _conn() as c:
        c.execute("UPDATE tickets SET status='resolved' WHERE id=?", (ticket_id,))

init_db()