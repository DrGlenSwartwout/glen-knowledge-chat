"""Per-client boolean intake facts (e.g. on_areds2) that drive client-reported
condition-program modifiers. Pure sqlite, no Flask. Email is lower-cased."""
from datetime import datetime, timezone

def _now(): return datetime.now(timezone.utc).isoformat()

def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS client_facts (
        email TEXT NOT NULL, fact_key TEXT NOT NULL, value INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT, PRIMARY KEY (email, fact_key))""")
    cx.commit()

def set_fact(cx, email, key, value):
    init_table(cx)
    cx.execute("""INSERT INTO client_facts (email, fact_key, value, updated_at)
        VALUES (?,?,?,?) ON CONFLICT(email, fact_key) DO UPDATE SET
        value=excluded.value, updated_at=excluded.updated_at""",
        ((email or "").strip().lower(), key, 1 if value else 0, _now()))
    cx.commit()

def get_facts(cx, email):
    init_table(cx)
    rows = cx.execute("SELECT fact_key, value FROM client_facts WHERE email=?",
                      ((email or "").strip().lower(),)).fetchall()
    return {r["fact_key"]: bool(r["value"]) for r in rows}
