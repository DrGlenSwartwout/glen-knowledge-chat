"""Shared library of reusable customer-facing line-item note messages.

A note typed on an invoice line item auto-saves here when the invoice is saved.
The order-entry form offers the saved notes in a per-line dropdown; the owner can
prune ones no longer wanted. ONE shared list across every line item and every
invoice (not per-client, not per-SKU) — Glen 2026-07 decision.

Pure functions over a sqlite connection (testable), mirroring the client_prices
module's shape.
"""
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS invoice_line_snippets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            last_used_at TEXT NOT NULL
        )
    """)
    cx.commit()


def add(cx, text):
    """Upsert a snippet by its exact (stripped) text. New text inserts; existing
    text just touches last_used_at so it floats to the top of the dropdown. A
    blank/whitespace-only note is ignored (returns None). Returns the snippet id."""
    t = (text or "").strip()
    if not t:
        return None
    now = _now()
    cx.execute(
        "INSERT INTO invoice_line_snippets (text, created_at, last_used_at) "
        "VALUES (?,?,?) ON CONFLICT(text) DO UPDATE SET last_used_at=excluded.last_used_at",
        (t, now, now))
    cx.commit()
    row = cx.execute("SELECT id FROM invoice_line_snippets WHERE text=?", (t,)).fetchone()
    return int(row[0]) if row else None


def list_all(cx):
    """Every saved snippet as [{id, text}], most-recently-used first."""
    return [{"id": int(r[0]), "text": r[1]} for r in cx.execute(
        "SELECT id, text FROM invoice_line_snippets ORDER BY last_used_at DESC, id DESC"
    ).fetchall()]


def remove(cx, snippet_id):
    """Delete one snippet by id. Returns True iff a row was removed."""
    try:
        sid = int(snippet_id)
    except (TypeError, ValueError):
        return False
    cur = cx.execute("DELETE FROM invoice_line_snippets WHERE id=?", (sid,))
    cx.commit()
    return cur.rowcount > 0
