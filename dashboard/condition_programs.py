"""Slice 1 of Condition Support Programs: sqlite store for Glen's 9
authored eye-condition support programs. Pure sqlite, no Flask.

`seed_if_empty` loads Glen-approved content from data/condition_programs_seed.json
into an empty table on first use — it never overwrites an operator's edit made
via the console editor (`upsert`). Ground truth once the table is non-empty is
whatever the operator has saved, not the seed file.
"""
import json
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS condition_programs (
            condition_key TEXT PRIMARY KEY,
            label TEXT,
            consult_recommended INTEGER NOT NULL DEFAULT 0,
            items_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT
        )""")


def _row(r):
    if r is None:
        return None
    return {
        "condition_key": r["condition_key"],
        "label": r["label"],
        "consult_recommended": bool(r["consult_recommended"]),
        "items": json.loads(r["items_json"] or "[]"),
        "updated_at": r["updated_at"],
    }


def get(cx, key):
    r = cx.execute("SELECT * FROM condition_programs WHERE condition_key=?",
                   (key,)).fetchone()
    return _row(r)


def all(cx):
    rows = cx.execute(
        "SELECT * FROM condition_programs ORDER BY condition_key").fetchall()
    return [_row(r) for r in rows]


def upsert(cx, key, label, consult_recommended, items):
    now = _now()
    cx.execute("""
        INSERT INTO condition_programs (condition_key, label, consult_recommended, items_json, updated_at)
        VALUES (?,?,?,?,?)
        ON CONFLICT(condition_key) DO UPDATE SET
            label=excluded.label,
            consult_recommended=excluded.consult_recommended,
            items_json=excluded.items_json,
            updated_at=excluded.updated_at
        """, (key, label, 1 if consult_recommended else 0, json.dumps(items or []), now))
    cx.commit()


def seed_if_empty(cx, seed_dict):
    """Insert each program from seed_dict ONLY if the table is currently empty.
    Idempotent (safe to call on every request) and never touches a table that
    already has rows — including one seeded on a prior call or edited by an
    operator via the console."""
    (count,) = cx.execute("SELECT COUNT(*) FROM condition_programs").fetchone()
    if count:
        return
    now = _now()
    for key, prog in (seed_dict or {}).items():
        cx.execute("""
            INSERT OR IGNORE INTO condition_programs
                (condition_key, label, consult_recommended, items_json, updated_at)
            VALUES (?,?,?,?,?)
            """, (key, prog.get("label") or "",
                  1 if prog.get("consult_recommended") else 0,
                  json.dumps(prog.get("items") or []), now))
    cx.commit()
