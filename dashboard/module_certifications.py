"""Storage for course-module certifications: a member buys certification for a
completed module ($200), which lands PENDING until an admin approves it. Pure:
stdlib + the caller's sqlite3 connection only. Money-adjacent — idempotent on
stripe_ref. Reads never raise."""
from __future__ import annotations
import sqlite3
import time


def _norm(s): return (s or "").strip().lower()
def _now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def init_table(cx) -> None:
    cx.execute("CREATE TABLE IF NOT EXISTS module_certifications("
               "email TEXT NOT NULL, course TEXT NOT NULL, module TEXT NOT NULL, "
               "status TEXT NOT NULL DEFAULT 'pending', stripe_ref TEXT, amount_cents INTEGER, "
               "created_at TEXT, approved_at TEXT, UNIQUE(email, course, module))")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_modcert_stripe_ref "
               "ON module_certifications(stripe_ref)")
    cx.commit()


def record_purchase(cx, email, course, module, stripe_ref, amount_cents) -> bool:
    """Insert a pending certification purchase row. Idempotent on stripe_ref:
    returns True only the FIRST time this stripe_ref is seen — returns False on
    replay. Also returns False (no insert) if an approved row already exists for
    this (email, course, module). Never raises."""
    try:
        init_table(cx)
        email = _norm(email)
        existing = status_for(cx, email, course, module)
        if existing == "approved":
            return False
        try:
            cx.execute("INSERT INTO module_certifications"
                       "(email, course, module, status, stripe_ref, amount_cents, created_at) "
                       "VALUES(?,?,?,?,?,?,?)",
                       (email, course, module, "pending", stripe_ref, amount_cents, _now()))
            cx.commit()
            return True
        except sqlite3.IntegrityError:
            cx.rollback()  # replayed stripe_ref — already recorded, do not duplicate
            return False
    except Exception:
        return False


def approve(cx, email, course, module, now_iso) -> bool:
    """Flip a pending row to approved. Returns True if a row was flipped, False
    if there was no row or it was not pending. Never raises."""
    try:
        init_table(cx)
        cur = cx.execute("UPDATE module_certifications SET status='approved', approved_at=? "
                         "WHERE email=? AND course=? AND module=? AND status='pending'",
                         (now_iso, _norm(email), course, module))
        cx.commit()
        return (cur.rowcount or 0) > 0
    except Exception:
        return False


def status_for(cx, email, course, module) -> str | None:
    try:
        init_table(cx)
        row = cx.execute("SELECT status FROM module_certifications "
                         "WHERE email=? AND course=? AND module=?",
                         (_norm(email), course, module)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def certified_modules(cx, email, course) -> set:
    try:
        init_table(cx)
        rows = cx.execute("SELECT module FROM module_certifications "
                          "WHERE email=? AND course=? AND status='approved'",
                          (_norm(email), course)).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def all_certified(cx, email, course, required_modules) -> bool:
    try:
        if not required_modules:
            return False
        certified = certified_modules(cx, email, course)
        return all(m in certified for m in required_modules)
    except Exception:
        return False
