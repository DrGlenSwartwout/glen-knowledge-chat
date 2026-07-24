# dashboard/course_entitlements.py
"""Email-keyed paid-access entitlements for MentorshipU courses.

Pure: stdlib + the caller's sqlite3 connection only, never imports app. A row
lifts an email to course access level 2 (paid). cert_onetime = lifetime
(expires_at NULL); membership = active through expires_at (epoch seconds).
Idempotent on (kind, stripe_ref) so a replayed Stripe webhook cannot double-grant.
"""
from __future__ import annotations

import time


def _norm(email: str | None) -> str:
    return (email or "").strip().lower()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def init_course_entitlements_table(cx) -> None:
    cx.execute(
        "CREATE TABLE IF NOT EXISTS course_entitlements("
        "id INTEGER PRIMARY KEY, email TEXT NOT NULL, kind TEXT NOT NULL, "
        "status TEXT NOT NULL, expires_at REAL, source TEXT NOT NULL, "
        "stripe_customer_id TEXT, stripe_ref TEXT, created_at TEXT, updated_at TEXT)"
    )
    cx.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_course_ent_ref "
        "ON course_entitlements(kind, stripe_ref) WHERE stripe_ref IS NOT NULL"
    )
    cx.commit()


def grant_cert(cx, email: str, *, source: str, stripe_ref: str | None = None,
               customer: str | None = None) -> None:
    """Grant a lifetime cert entitlement. Idempotent on (cert_onetime, stripe_ref)."""
    init_course_entitlements_table(cx)
    email, now = _norm(email), _now_iso()
    if stripe_ref:
        row = cx.execute(
            "SELECT id FROM course_entitlements WHERE kind='cert_onetime' AND stripe_ref=?",
            (stripe_ref,)).fetchone()
        if row:
            cx.execute("UPDATE course_entitlements SET status='active', email=?, updated_at=? "
                       "WHERE id=?", (email, now, row[0]))
            cx.commit()
            return
    cx.execute(
        "INSERT INTO course_entitlements(email, kind, status, expires_at, source, "
        "stripe_customer_id, stripe_ref, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (email, "cert_onetime", "active", None, source, customer, stripe_ref, now, now))
    cx.commit()


def grant_membership(cx, email: str, *, until_epoch: float | None, source: str,
                     stripe_ref: str | None = None, customer: str | None = None) -> None:
    """Grant/extend a membership through until_epoch. Extend never shortens.
    Idempotent on (membership, stripe_ref)."""
    init_course_entitlements_table(cx)
    email, now = _norm(email), _now_iso()
    if stripe_ref:
        row = cx.execute(
            "SELECT id, expires_at FROM course_entitlements WHERE kind='membership' AND stripe_ref=?",
            (stripe_ref,)).fetchone()
        if row:
            cur = row[1]
            new_exp = until_epoch if (cur is None or (until_epoch is not None and until_epoch > cur)) else cur
            cx.execute("UPDATE course_entitlements SET status='active', expires_at=?, email=?, "
                       "updated_at=? WHERE id=?", (new_exp, email, now, row[0]))
            cx.commit()
            return
    cx.execute(
        "INSERT INTO course_entitlements(email, kind, status, expires_at, source, "
        "stripe_customer_id, stripe_ref, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (email, "membership", "active", until_epoch, source, customer, stripe_ref, now, now))
    cx.commit()


def expire_membership(cx, *, stripe_ref: str) -> None:
    """Mark a Stripe-backed membership canceled so paid_level_for relocks it."""
    init_course_entitlements_table(cx)
    cx.execute("UPDATE course_entitlements SET status='canceled', updated_at=? "
               "WHERE kind='membership' AND stripe_ref=?", (_now_iso(), stripe_ref))
    cx.commit()


def paid_level_for(cx, email: str, now: float | None = None) -> int:
    """2 if the email has an active cert OR active (unexpired) membership, else 0.
    Never raises."""
    try:
        email = _norm(email)
        if not email:
            return 0
        now = float(now) if now is not None else time.time()
        row = cx.execute(
            "SELECT 1 FROM course_entitlements WHERE email=? AND status='active' AND ("
            "kind='cert_onetime' OR (kind='membership' AND (expires_at IS NULL OR expires_at > ?))"
            ") LIMIT 1", (email, now)).fetchone()
        return 2 if row else 0
    except Exception:
        return 0
