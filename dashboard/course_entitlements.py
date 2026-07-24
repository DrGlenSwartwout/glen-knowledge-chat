# dashboard/course_entitlements.py
"""Email-keyed paid-access entitlements for MentorshipU courses.

Pure: stdlib + the caller's sqlite3 connection only, never imports app. A row
lifts an email to course access level 2 (paid). cert_onetime = lifetime
(expires_at NULL); membership = active through expires_at (epoch seconds).
Idempotent on (kind, stripe_ref) so a replayed Stripe webhook cannot double-grant.
"""
from __future__ import annotations

import time
import sqlite3


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
    try:
        cx.execute(
            "INSERT INTO course_entitlements(email, kind, status, expires_at, source, "
            "stripe_customer_id, stripe_ref, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (email, "cert_onetime", "active", None, source, customer, stripe_ref, now, now))
        cx.commit()
    except sqlite3.IntegrityError:
        # A concurrent grant for the same (cert_onetime, stripe_ref) won the race; the
        # row already exists. Roll back and ensure it is active (idempotent success,
        # not a crash). Reachable only if two grants race the pre-SELECT; the UNIQUE
        # partial index guarantees no duplicate row.
        cx.rollback()
        if stripe_ref:
            cx.execute("UPDATE course_entitlements SET status='active', email=?, updated_at=? "
                       "WHERE kind='cert_onetime' AND stripe_ref=?", (email, now, stripe_ref))
            cx.commit()


def _membership_new_expiry(cur: float | None, until_epoch: float | None) -> float | None:
    """Extend-never-shorten with None == unlimited (+inf): stay unlimited if either
    side is unlimited; otherwise the later of the two epochs."""
    if cur is None or until_epoch is None:
        return None
    return max(cur, until_epoch)


def grant_membership(cx, email: str, *, until_epoch: float | None, source: str,
                     stripe_ref: str | None = None, customer: str | None = None) -> None:
    """Grant/extend a membership through until_epoch. Extend never shortens
    (None == unlimited). Idempotent on (membership, stripe_ref)."""
    init_course_entitlements_table(cx)
    email, now = _norm(email), _now_iso()
    if stripe_ref:
        row = cx.execute(
            "SELECT id, expires_at FROM course_entitlements WHERE kind='membership' AND stripe_ref=?",
            (stripe_ref,)).fetchone()
        if row:
            new_exp = _membership_new_expiry(row[1], until_epoch)
            cx.execute("UPDATE course_entitlements SET status='active', expires_at=?, email=?, "
                       "updated_at=? WHERE id=?", (new_exp, email, now, row[0]))
            cx.commit()
            return
    try:
        cx.execute(
            "INSERT INTO course_entitlements(email, kind, status, expires_at, source, "
            "stripe_customer_id, stripe_ref, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (email, "membership", "active", until_epoch, source, customer, stripe_ref, now, now))
        cx.commit()
    except sqlite3.IntegrityError:
        # Concurrent grant for the same (membership, stripe_ref) won the race; extend
        # the now-existing row instead of crashing (idempotent, never shortens).
        cx.rollback()
        if stripe_ref:
            row = cx.execute(
                "SELECT id, expires_at FROM course_entitlements WHERE kind='membership' AND stripe_ref=?",
                (stripe_ref,)).fetchone()
            if row:
                new_exp = _membership_new_expiry(row[1], until_epoch)
                cx.execute("UPDATE course_entitlements SET status='active', expires_at=?, email=?, "
                           "updated_at=? WHERE id=?", (new_exp, email, now, row[0]))
                cx.commit()


def expire_membership(cx, *, stripe_ref: str) -> None:
    """Mark a Stripe-backed membership canceled so paid_level_for relocks it."""
    init_course_entitlements_table(cx)
    cx.execute("UPDATE course_entitlements SET status='canceled', updated_at=? "
               "WHERE kind='membership' AND stripe_ref=?", (_now_iso(), stripe_ref))
    cx.commit()


def email_for_stripe_ref(cx, stripe_ref: str) -> str | None:
    """Return the membership row's stored email for a Stripe subscription ref, or None.
    Lets a renewal extend the right member even when the Stripe subscription metadata
    carries no email (anonymous checkout). Never raises."""
    try:
        init_course_entitlements_table(cx)
        row = cx.execute(
            "SELECT email FROM course_entitlements WHERE kind='membership' AND stripe_ref=? "
            "ORDER BY id DESC LIMIT 1", (stripe_ref,)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


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
