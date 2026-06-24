"""Coaching activation — a 1-month coaching window per received remedy program.

Distinct from premium-access (`memberships`): a coaching window lives ONLY in
this table and never touches `memberships.expires_at`. Window ops here are pure
sqlite; eligibility (membership active at order date) is orchestrated in app.py.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

WINDOW_DAYS = int(os.environ.get("COACHING_WINDOW_DAYS", "30"))

# Remedy-program order sources that earn a coaching month. Excludes the $99
# membership charge ("membership"), practitioner stock ("wholesale"), and
# internal/gift flows ("personal", "dropship").
QUALIFYING_SOURCES = {"biofield", "funnel", "reorder", "portal-reorder",
                      "groovekart", "dispensary"}

_DDL = """
CREATE TABLE IF NOT EXISTS coaching_windows (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    email      TEXT NOT NULL,
    order_id   INTEGER,
    started_at TEXT NOT NULL,
    ends_at    TEXT NOT NULL,
    source     TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_coaching_email ON coaching_windows(email);
CREATE INDEX IF NOT EXISTS ix_coaching_order ON coaching_windows(order_id);
"""


def init_coaching_table(cx) -> None:
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            cx.execute(stmt)
    cx.commit()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _with_days_remaining(d: dict) -> dict:
    try:
        end = datetime.fromisoformat(d["ends_at"].rstrip("Z"))
        d["days_remaining"] = max(0, (end - datetime.utcnow()).days)
    except Exception:
        d["days_remaining"] = 0
    return d


def active_window(cx, email):
    if not email:
        return None
    row = cx.execute(
        "SELECT * FROM coaching_windows WHERE email=? AND ends_at > ? "
        "ORDER BY ends_at DESC LIMIT 1",
        (email, _now_iso())).fetchone()
    return _with_days_remaining(dict(row)) if row else None


def window_for_order(cx, order_id):
    if order_id is None:
        return None
    row = cx.execute(
        "SELECT * FROM coaching_windows WHERE order_id=? ORDER BY id DESC LIMIT 1",
        (order_id,)).fetchone()
    return _with_days_remaining(dict(row)) if row else None


def open_window(cx, *, email, order_id, days, source, now=None):
    """Open a coaching window unless one is already active (no-stacking) or this
    order already activated one (one-per-order). Returns {created, window}."""
    aw = active_window(cx, email)
    if aw:
        return {"created": False, "window": aw}
    if order_id is not None:
        wo = window_for_order(cx, order_id)
        if wo:
            return {"created": False, "window": wo}
    now = now or datetime.utcnow()
    started = now.isoformat() + "Z"
    ends = (now + timedelta(days=int(days))).isoformat() + "Z"
    cur = cx.execute(
        "INSERT INTO coaching_windows (email, order_id, started_at, ends_at, source, created_at) "
        "VALUES (?,?,?,?,?,?)", (email, order_id, started, ends, source, started))
    cx.commit()
    win = {"id": cur.lastrowid, "email": email, "order_id": order_id,
           "started_at": started, "ends_at": ends, "source": source, "created_at": started}
    return {"created": True, "window": _with_days_remaining(win)}


def list_windows(cx, *, active_only=False, limit=500):
    if active_only:
        rows = cx.execute(
            "SELECT * FROM coaching_windows WHERE ends_at > ? ORDER BY started_at DESC LIMIT ?",
            (_now_iso(), int(limit))).fetchall()
    else:
        rows = cx.execute(
            "SELECT * FROM coaching_windows ORDER BY started_at DESC LIMIT ?",
            (int(limit),)).fetchall()
    return [_with_days_remaining(dict(r)) for r in rows]


# --- Admin action (register on import) ---
from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER


def _grant_exec(params, ctx):
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    email = ((params or {}).get("email") or "").strip().lower()
    if not email:
        raise ValueError("email required")
    res = open_window(cx, email=email, order_id=None, days=WINDOW_DAYS, source="admin")
    w = res["window"]
    verb = "started" if res["created"] else "already active"
    return {"email": email, "created": res["created"], "ends_at": w["ends_at"],
            "message": f"Coaching for {email} {verb} (through {w['ends_at'][:10]})."}


action(key="coaching.grant", module="coaching", title="Grant coaching month",
       description="Manually start a 1-month coaching window for a member (bypasses "
                   "order-month eligibility; respects no-stacking).",
       risk_tier=LOW_WRITE, permission=(OWNER,))(_grant_exec)
