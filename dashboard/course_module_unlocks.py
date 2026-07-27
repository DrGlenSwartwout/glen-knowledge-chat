"""The MentorshipU drip's unlocked-module set + the learner's 'unlock next'
preference. Pure: stdlib + the caller's sqlite3 connection only. Reads never raise."""
from __future__ import annotations
import sqlite3
import time


def _norm(s): return (s or "").strip().lower()
def _now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def init_unlock_tables(cx) -> None:
    cx.execute("CREATE TABLE IF NOT EXISTS course_module_unlocks("
               "email TEXT NOT NULL, course TEXT NOT NULL, module TEXT NOT NULL, "
               "unlocked_at TEXT, source TEXT, UNIQUE(email, course, module))")
    cx.execute("CREATE TABLE IF NOT EXISTS course_unlock_pref("
               "email TEXT NOT NULL, course TEXT NOT NULL, module TEXT, set_at TEXT, "
               "UNIQUE(email, course))")
    cx.commit()


def unlock_module(cx, email, course, module, source="drip") -> None:
    init_unlock_tables(cx)
    cx.execute("INSERT OR IGNORE INTO course_module_unlocks(email, course, module, unlocked_at, source) "
               "VALUES(?,?,?,?,?)", (_norm(email), course, module, _now(), source))
    cx.commit()


def unlocked_modules(cx, email, course) -> set:
    try:
        rows = cx.execute("SELECT module FROM course_module_unlocks WHERE email=? AND course=?",
                          (_norm(email), course)).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def set_unlock_pref(cx, email, course, module) -> None:
    init_unlock_tables(cx)
    cx.execute("INSERT INTO course_unlock_pref(email, course, module, set_at) VALUES(?,?,?,?) "
               "ON CONFLICT(email, course) DO UPDATE SET module=excluded.module, set_at=excluded.set_at",
               (_norm(email), course, module, _now()))
    cx.commit()


def take_unlock_pref(cx, email, course):
    try:
        init_unlock_tables(cx)
        row = cx.execute("SELECT module FROM course_unlock_pref WHERE email=? AND course=?",
                         (_norm(email), course)).fetchone()
        if not row or not row[0]:
            return None
        cx.execute("DELETE FROM course_unlock_pref WHERE email=? AND course=?", (_norm(email), course))
        cx.commit()
        return row[0]
    except Exception:
        return None


def init_drip_charges_table(cx) -> None:
    cx.execute("CREATE TABLE IF NOT EXISTS course_drip_charges("
               "sub_id TEXT NOT NULL, invoice_id TEXT NOT NULL, created_at TEXT)")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_drip_charge_invoice "
               "ON course_drip_charges(invoice_id)")
    cx.commit()


def record_drip_charge(cx, sub_id, invoice_id) -> bool:
    """Record a paid drip invoice (idempotent on invoice_id, mirroring
    course_entitlements.record_plan_charge's idempotency). Returns True only the
    FIRST time this invoice_id is seen — the caller unlocks a module only when this
    is True, so a replayed invoice.paid never unlocks a second module. Returns
    False on replay or on any error (never raises)."""
    try:
        init_drip_charges_table(cx)
        try:
            cx.execute("INSERT INTO course_drip_charges(sub_id, invoice_id, created_at) "
                       "VALUES(?,?,?)", (sub_id, invoice_id, _now()))
            cx.commit()
            return True
        except sqlite3.IntegrityError:
            cx.rollback()  # replayed invoice — already unlocked its module, do not repeat
            return False
    except Exception:
        return False


def next_module_to_unlock(cx, email, course, ordered_modules):
    try:
        init_unlock_tables(cx)
        unlocked = unlocked_modules(cx, email, course)
        pref = None
        row = cx.execute("SELECT module FROM course_unlock_pref WHERE email=? AND course=?",
                         (_norm(email), course)).fetchone()
        if row and row[0] and row[0] not in unlocked:
            pref = row[0]
        if pref:
            return pref
        for m in ordered_modules:
            if m not in unlocked:
                return m
        return None
    except Exception:
        return None
