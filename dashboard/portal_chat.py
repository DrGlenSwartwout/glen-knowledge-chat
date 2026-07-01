"""Persistent client-portal chat thread, keyed by client email.

Before this, the portal "Ask Dr. Glen" chat lived only in the browser (lost on
refresh) — only a rolled-up ASH ally summary was stored. This keeps the actual
turns so the thread is remembered across sessions and Dr. Glen can reply into it
from the console. Pure sqlite; caller passes the connection.
"""
import sqlite3
from datetime import datetime, timezone

# role values
CLIENT = "client"          # the portal client
ASSISTANT = "assistant"    # the AI concierge ("Ask Dr. Glen")
PRACTITIONER = "practitioner"  # a real reply from Dr. Glen / Rae


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS portal_chat_messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL, role TEXT NOT NULL,
        author TEXT, content TEXT NOT NULL, created_at TEXT)""")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_pcm_email ON portal_chat_messages(email, id)")
    cx.commit()


def add_message(cx, email, role, content, author=None):
    """Append a message to a client's thread. role = client|assistant|practitioner.
    Returns the new row id, or None if email/content is empty (no-op)."""
    init_table(cx)
    e = (email or "").strip().lower()
    c = (content or "").strip()
    if not e or not c:
        return None
    cur = cx.execute(
        "INSERT INTO portal_chat_messages(email,role,author,content,created_at) VALUES(?,?,?,?,?)",
        (e, role, (author or "").strip() or None, c, _now()))
    cx.commit()
    return cur.lastrowid


def list_messages(cx, email, limit=100):
    """The client's thread in chronological order (oldest first), capped at `limit`
    most-recent messages."""
    init_table(cx)
    cx.row_factory = sqlite3.Row
    e = (email or "").strip().lower()
    rows = cx.execute(
        "SELECT role, author, content, created_at FROM portal_chat_messages "
        "WHERE email=? ORDER BY id DESC LIMIT ?", (e, int(limit or 100))).fetchall()
    return [{"role": r["role"], "author": r["author"] or "", "content": r["content"],
             "created_at": r["created_at"]} for r in reversed(rows)]


def record_exchange(cx, email, query, answer, client_name=None):
    """Persist one concierge turn: the client's message + the AI answer."""
    add_message(cx, email, CLIENT, query, author=client_name or "You")
    add_message(cx, email, ASSISTANT, answer, author="Ask Dr. Glen")
