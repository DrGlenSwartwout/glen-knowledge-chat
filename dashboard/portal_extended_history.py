"""Portal-entered personal and family health-history categories."""
import json
from datetime import datetime, timezone


CATEGORIES = (
    "surgeries", "physical_trauma", "psychoemotional_trauma", "toxins",
    "vaccinations", "family_history", "diagnoses", "allergies",
    "dental", "sleep",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS portal_extended_history (
            email TEXT PRIMARY KEY,
            answers_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
    cx.commit()


def save(cx, email, answers):
    init_table(cx)
    clean = {}
    for category in CATEGORIES:
        yes = bool((answers or {}).get(category + "_yes"))
        clean[category + "_yes"] = yes
        clean[category + "_text"] = (
            str((answers or {}).get(category + "_text") or "").strip()
            if yes else ""
        )
    cx.execute("""
        INSERT INTO portal_extended_history (email, answers_json, updated_at)
        VALUES (?,?,?)
        ON CONFLICT(email) DO UPDATE SET
            answers_json=excluded.answers_json,
            updated_at=excluded.updated_at
        """, ((email or "").strip().lower(), json.dumps(clean), _now()))
    cx.commit()


def get(cx, email):
    init_table(cx)
    row = cx.execute(
        "SELECT answers_json, updated_at FROM portal_extended_history "
        "WHERE lower(email)=lower(?)", ((email or "").strip(),)
    ).fetchone()
    if not row:
        return None
    return {"answers": json.loads(row[0] or "{}"), "updated_at": row[1]}


def has(cx, email):
    init_table(cx)
    return bool(cx.execute(
        "SELECT 1 FROM portal_extended_history WHERE lower(email)=lower(?)",
        ((email or "").strip(),),
    ).fetchone())
