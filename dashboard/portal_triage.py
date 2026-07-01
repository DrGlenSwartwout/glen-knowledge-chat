"""Triage of client portal-chat messages.

Flags the messages that need Dr. Glen's personal attention — direct questions,
complaints, service issues/bugs, health concerns, or requests only he can fulfill
— so the console shows a queue and Glen gets emailed with full context + a
recommendation. Routine chat the AI already handled is NOT flagged.

The classifier's LLM call is injected (`complete(system, user) -> str`) so this is
testable offline. Pure sqlite store; caller passes the connection.
"""
import json
import re
import sqlite3
from datetime import datetime, timezone

CATEGORIES = ("question", "complaint", "issue", "bug", "health-concern", "request", "other")
URGENCIES = ("low", "medium", "high")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS portal_triage(
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, name TEXT, category TEXT,
        urgency TEXT, summary TEXT, recommendation TEXT, message TEXT, answer TEXT,
        created_at TEXT, resolved INTEGER NOT NULL DEFAULT 0)""")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_triage_open ON portal_triage(resolved, id)")
    cx.commit()


def add_item(cx, email, name, category, urgency, summary, recommendation, message, answer=""):
    init_table(cx)
    cur = cx.execute(
        "INSERT INTO portal_triage(email,name,category,urgency,summary,recommendation,"
        "message,answer,created_at,resolved) VALUES(?,?,?,?,?,?,?,?,?,0)",
        ((email or "").strip().lower(), (name or "").strip(), category, urgency,
         summary, recommendation, message, answer, _now()))
    cx.commit()
    return cur.lastrowid


def list_open(cx, limit=50):
    init_table(cx)
    cx.row_factory = sqlite3.Row
    rows = cx.execute("SELECT * FROM portal_triage WHERE resolved=0 ORDER BY id DESC LIMIT ?",
                      (int(limit or 50),)).fetchall()
    return [dict(r) for r in rows]


def open_count(cx):
    init_table(cx)
    return cx.execute("SELECT COUNT(*) FROM portal_triage WHERE resolved=0").fetchone()[0]


def resolve(cx, item_id):
    init_table(cx)
    cx.execute("UPDATE portal_triage SET resolved=1 WHERE id=?", (int(item_id),))
    cx.commit()


_TRIAGE_SYSTEM = (
    "You triage a client's message to a naturopathic practitioner (Dr. Glen Swartwout). "
    "Decide if it NEEDS DR. GLEN'S PERSONAL ATTENTION: a direct question meant for him, a "
    "complaint, a problem or bug with the service/portal, a health concern that goes beyond "
    "general information, or a request only he can fulfill (e.g. a consultation, a custom "
    "remedy, a review). Routine small-talk, thanks, or a question the AI already fully answered "
    "do NOT need his attention.\n"
    "Return STRICT JSON ONLY, no prose: "
    '{"needs_attention": true|false, "category": "question"|"complaint"|"issue"|"bug"|'
    '"health-concern"|"request"|"other", "urgency": "low"|"medium"|"high", '
    '"summary": "one-line what they need", "recommendation": "what Dr. Glen should do"}. '
    "If it does not need attention, return {\"needs_attention\": false} and nothing else matters."
)


def _parse(text):
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def classify(complete, query, answer="", history_text=""):
    """complete(system,user) -> JSON str. Returns a normalized dict when the message
    needs attention, else None. Never raises (returns None on any parse issue)."""
    if not (query or "").strip():
        return None
    user = (f"CONVERSATION SO FAR:\n{history_text}\n\n" if history_text else "") + \
           f"CLIENT MESSAGE:\n{query}\n\nAI ANSWER GIVEN:\n{answer or '(none)'}"
    data = _parse(complete(_TRIAGE_SYSTEM, user))
    if not data.get("needs_attention"):
        return None
    cat = data.get("category") if data.get("category") in CATEGORIES else "other"
    urg = data.get("urgency") if data.get("urgency") in URGENCIES else "medium"
    return {"needs_attention": True, "category": cat, "urgency": urg,
            "summary": (data.get("summary") or "").strip() or "(no summary)",
            "recommendation": (data.get("recommendation") or "").strip() or "(no recommendation)"}
