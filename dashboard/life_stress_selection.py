"""Per-client Life Stress essence SELECTION (a saved preference — not an order).
Pure sqlite (LOG_DB), no Flask. Mirrors dashboard/client_conditions.py. Stores the
list of essence slugs a client checked on their Life Stress card. Never raises on
bad/blank data (returns [])."""
import json
from datetime import datetime, timezone


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS life_stress_selections (
        email TEXT PRIMARY KEY,
        slugs_json TEXT NOT NULL DEFAULT '[]',
        updated_at TEXT
    )""")


def get(cx, email):
    """The client's saved slug list, or [] (missing row, blank email, or bad JSON)."""
    e = (email or "").strip().lower()
    if not e:
        return []
    try:
        init_table(cx)
        row = cx.execute("SELECT slugs_json FROM life_stress_selections WHERE email=?",
                         (e,)).fetchone()
        if not row:
            return []
        val = json.loads(row[0])
        return [str(s) for s in val] if isinstance(val, list) else []
    except (ValueError, TypeError):
        return []


def set(cx, email, slugs):
    """Upsert the client's selection (a list of slugs). Stamps updated_at."""
    e = (email or "").strip().lower()
    if not e:
        return
    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps([str(s) for s in (slugs or [])])
    init_table(cx)
    cx.execute("""INSERT INTO life_stress_selections(email, slugs_json, updated_at)
                  VALUES(?,?,?)
                  ON CONFLICT(email) DO UPDATE SET slugs_json=excluded.slugs_json,
                    updated_at=excluded.updated_at""", (e, payload, now))
    cx.commit()


def clear(cx, email):
    e = (email or "").strip().lower()
    if not e:
        return
    init_table(cx)
    cx.execute("DELETE FROM life_stress_selections WHERE email=?", (e,))
    cx.commit()
