"""Portal-entered current prescriptions, OTC drugs, and supplements."""
from datetime import datetime, timezone


KINDS = ("prescriptions", "otc", "supplements")


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS portal_health_history (
            email TEXT PRIMARY KEY,
            prescriptions_yes INTEGER NOT NULL DEFAULT 0,
            prescriptions_text TEXT NOT NULL DEFAULT '',
            otc_yes INTEGER NOT NULL DEFAULT 0,
            otc_text TEXT NOT NULL DEFAULT '',
            supplements_yes INTEGER NOT NULL DEFAULT 0,
            supplements_text TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        )""")
    cx.commit()


def save(cx, email, data):
    init_table(cx)
    email = (email or "").strip().lower()
    values = []
    for kind in KINDS:
        yes = bool((data or {}).get(kind + "_yes"))
        text = str((data or {}).get(kind + "_text") or "").strip()
        values.extend((1 if yes else 0, text if yes else ""))
    cx.execute("""
        INSERT INTO portal_health_history (
            email, prescriptions_yes, prescriptions_text,
            otc_yes, otc_text, supplements_yes, supplements_text, updated_at
        ) VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(email) DO UPDATE SET
            prescriptions_yes=excluded.prescriptions_yes,
            prescriptions_text=excluded.prescriptions_text,
            otc_yes=excluded.otc_yes,
            otc_text=excluded.otc_text,
            supplements_yes=excluded.supplements_yes,
            supplements_text=excluded.supplements_text,
            updated_at=excluded.updated_at
        """, (email, *values, _now()))
    cx.commit()


def get(cx, email):
    init_table(cx)
    row = cx.execute("""
        SELECT prescriptions_yes, prescriptions_text, otc_yes, otc_text,
               supplements_yes, supplements_text, updated_at
        FROM portal_health_history WHERE lower(email)=lower(?)
        """, ((email or "").strip(),)).fetchone()
    if not row:
        return None
    return {
        "prescriptions_yes": bool(row[0]), "prescriptions_text": row[1],
        "otc_yes": bool(row[2]), "otc_text": row[3],
        "supplements_yes": bool(row[4]), "supplements_text": row[5],
        "updated_at": row[6],
    }


def has(cx, email):
    init_table(cx)
    return bool(cx.execute(
        "SELECT 1 FROM portal_health_history WHERE lower(email)=lower(?)",
        ((email or "").strip(),),
    ).fetchone())
