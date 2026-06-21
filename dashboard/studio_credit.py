"""Studio-credit free month: claim store + approve/reject with a one-per-year
guard. Phase 1 = console side only (no public claim form). The grant+notify side
effect is injected at approve time so this module stays Flask-free and unit-testable."""
import sqlite3
import uuid
from datetime import datetime, timedelta


def _now():
    return datetime.utcnow().isoformat() + "Z"


def migrate(cx) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS studio_credit_claims (
            id            TEXT PRIMARY KEY,
            email         TEXT NOT NULL,
            invoice_ref   TEXT NOT NULL DEFAULT '',
            proof_note    TEXT NOT NULL DEFAULT '',
            status        TEXT NOT NULL DEFAULT 'pending',
            created_at    TEXT NOT NULL,
            created_by    TEXT NOT NULL DEFAULT '',
            decided_at    TEXT,
            decided_by    TEXT,
            decision_note TEXT NOT NULL DEFAULT '',
            membership_id TEXT,
            source        TEXT NOT NULL DEFAULT 'console'
        )
    """)
    cx.commit()


def _row(r):
    return dict(r) if r is not None else None


def add_claim(cx, *, email, invoice_ref="", proof_note="", source="console", created_by=""):
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("valid email required")
    cid = str(uuid.uuid4())
    cx.execute(
        "INSERT INTO studio_credit_claims "
        "(id, email, invoice_ref, proof_note, status, created_at, created_by, source) "
        "VALUES (?,?,?,?, 'pending', ?, ?, ?)",
        (cid, email, invoice_ref or "", proof_note or "", _now(), created_by or "", source or "console"))
    cx.commit()
    return get(cx, cid)


def get(cx, claim_id):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    return _row(cur.execute(
        "SELECT * FROM studio_credit_claims WHERE id=?", (claim_id,)).fetchone())


def list_claims(cx, status=None):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    if status:
        rows = cur.execute(
            "SELECT * FROM studio_credit_claims WHERE status=? ORDER BY created_at DESC",
            (status,)).fetchall()
    else:
        rows = cur.execute(
            "SELECT * FROM studio_credit_claims ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]
