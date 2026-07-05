"""Throttled Stripe-failure alerting. Every Stripe session-create failure (any
checkout path) records a row here; the first failure in a throttle window emails
the owner, and the console Money signal turns red while failures are recent.

Best-effort by contract: nothing here may raise into a payment/checkout path —
record_failure swallows every error (it runs inside except blocks)."""
import os
import socket
import sqlite3
from datetime import datetime, timezone, timedelta

OWNER_EMAIL = os.environ.get("GLEN_EMAIL", "drglenswartwout@gmail.com")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _db_path(cx):
    """Absolute path of the connection's main DB file ('' for in-memory/unknown).
    Lets an alert say WHERE the row landed, so an 'alerted but not in the console'
    incident is traceable to the writing process's DB instead of a mystery."""
    try:
        for _seq, name, file in cx.execute("PRAGMA database_list").fetchall():
            if name == "main":
                return file or ""
    except Exception:
        pass
    return ""


def init_stripe_alerts_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS stripe_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            emailed_at TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_stripe_failures_created ON stripe_failures(created_at)")
    cx.commit()


def _send_alert(context, error, when, *, db_path="", persisted=True):
    """Best-effort owner email. Returns True if sent. Carries host + ledger DB path
    + persistence status so a failure that DIDN'T reach the durable ledger (email
    fired but no console row) is self-diagnosing on the next incident."""
    try:
        from dashboard.inbox import send_email
        ledger = ("recorded to the ledger" if persisted else
                  "⚠️ NOT PERSISTED — this alert did NOT reach the durable ledger")
        body = (f"A Stripe card-payment attempt failed.\n\n"
                f"Context: {context}\n"
                f"When: {when}\n"
                f"Error: {error}\n\n"
                f"Persistence: {ledger}\n"
                f"Host: {socket.gethostname()}\n"
                f"Ledger DB: {db_path or '(in-memory / unknown)'}\n\n"
                f"Customers are shown the Zelle/Wise fallback where applicable. "
                f"Card checkout may be down — check the Stripe key/status. "
                f"If this says NOT PERSISTED, or the Ledger DB is not the prod "
                f"/data disk, this failure will NOT appear in the Money console.")
        send_email(OWNER_EMAIL, "⚠️ Card payment failure (Stripe)", body,
                   from_name="Remedy Match Ops")
        return True
    except Exception as e:  # never propagate
        print(f"[stripe-alert] email skipped: {e!r}", flush=True)
        return False


def record_failure(cx, context, error, *, throttle_min=20, now=None, notify=True):
    """Record one Stripe failure; email the owner at most once per throttle_min.
    Never raises (runs inside checkout except blocks)."""
    try:
        init_stripe_alerts_table(cx)
        ts = now or _now()
        cur = cx.execute(
            "INSERT INTO stripe_failures (context, error, created_at) VALUES (?,?,?)",
            (str(context or ""), str(error or "")[:500], ts))
        rowid = cur.lastrowid
        cx.commit()
        # Durability read-back: prove the row is queryable AFTER commit, and capture
        # WHICH db file it landed in. If a future incident emails but shows no console
        # row, the alert itself will now say persisted=False and/or the wrong DB path.
        db_path = _db_path(cx)
        persisted = bool(cx.execute(
            "SELECT 1 FROM stripe_failures WHERE id=?", (rowid,)).fetchone())
        emailed = False
        if notify:
            cutoff = (datetime.fromisoformat(ts) - timedelta(minutes=throttle_min)).isoformat()
            recent = cx.execute(
                "SELECT 1 FROM stripe_failures WHERE emailed_at IS NOT NULL AND emailed_at >= ? LIMIT 1",
                (cutoff,)).fetchone()
            if not recent and _send_alert(context, error, ts,
                                          db_path=db_path, persisted=persisted):
                cx.execute("UPDATE stripe_failures SET emailed_at=? WHERE id=?", (ts, rowid))
                cx.commit()
                emailed = True
        return {"id": rowid, "emailed": emailed, "persisted": persisted, "db_path": db_path}
    except Exception as e:
        print(f"[stripe-alert] record skipped: {e!r}", flush=True)
        return {"id": 0, "emailed": False, "persisted": False, "db_path": ""}


def recent_failure_count(cx, *, minutes=30, now=None):
    """Count failures within the window. 0 on any error/missing table."""
    try:
        ts = now or _now()
        cutoff = (datetime.fromisoformat(ts) - timedelta(minutes=minutes)).isoformat()
        row = cx.execute(
            "SELECT COUNT(*) FROM stripe_failures WHERE created_at >= ?", (cutoff,)).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0
