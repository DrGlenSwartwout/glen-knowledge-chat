"""Biofield Consult: eligibility gate + paid-test detection. Stdlib-only; import
without importing app."""
import sqlite3, json as _json
from datetime import datetime, timezone

CONSULT = {"session_type": "biofield-consult", "practitioner": "glen",
           "duration_min": 30, "medium": "video", "test_slug": "biofield-analysis"}


def init_consult_tables(cx) -> None:
    cx.execute("""CREATE TABLE IF NOT EXISTS consult_eligibility (
        email TEXT NOT NULL, session_type TEXT NOT NULL,
        ready INTEGER DEFAULT 0, marked_at TEXT,
        PRIMARY KEY (email, session_type))""")
    cx.commit()


def set_consult_ready(cx, email: str, ready: bool,
                      session_type: str = "biofield-consult") -> bool:
    email = (email or "").strip().lower()
    now = datetime.now(timezone.utc).isoformat()
    cx.execute("INSERT INTO consult_eligibility (email, session_type, ready, marked_at) "
               "VALUES (?,?,?,?) ON CONFLICT(email, session_type) "
               "DO UPDATE SET ready=excluded.ready, marked_at=excluded.marked_at",
               (email, session_type, 1 if ready else 0, now))
    cx.commit()
    return bool(ready)


def consult_is_ready(cx, email: str, session_type: str = "biofield-consult") -> bool:
    email = (email or "").strip().lower()
    r = cx.execute("SELECT ready FROM consult_eligibility WHERE email=? AND session_type=?",
                   (email, session_type)).fetchone()
    return bool(r[0]) if r else False


def has_paid_purchase(cx, email: str, slug: str) -> bool:
    email = (email or "").strip().lower()
    try:
        rows = cx.execute("SELECT items_json, pay_status, paid_cents FROM orders "
                          "WHERE lower(email)=?", (email,)).fetchall()
    except sqlite3.OperationalError:
        return False
    for items, pay_status, paid_cents in rows:
        paid = (str(pay_status or "").lower() == "paid") or (int(paid_cents or 0) > 0)
        if not paid:
            continue
        try:
            for line in _json.loads(items or "[]"):
                if (line.get("slug") or "") == slug:
                    return True
        except Exception:
            continue
    return False
