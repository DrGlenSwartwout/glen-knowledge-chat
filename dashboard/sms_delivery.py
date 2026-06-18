"""Outbound SMS delivery status log (Twilio status callbacks). One row per
MessageSid, status updated in place (queued -> sent -> delivered/failed)."""
import datetime
import sqlite3

FAILED = ("failed", "undelivered")


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS sms_delivery_log (
        message_sid TEXT PRIMARY KEY, to_number TEXT, status TEXT,
        error_code TEXT, created_at TEXT, updated_at TEXT)""")
    cx.commit()


def record(cx, message_sid, to_number, status, error_code=""):
    init_table(cx)
    now = _now()
    cx.execute("""INSERT INTO sms_delivery_log (message_sid, to_number, status, error_code, created_at, updated_at)
                  VALUES (?,?,?,?,?,?)
                  ON CONFLICT(message_sid) DO UPDATE SET to_number=excluded.to_number,
                      status=excluded.status, error_code=excluded.error_code, updated_at=?""",
               (message_sid or "", to_number or "", status or "", error_code or "", now, now, now))
    cx.commit()


def recent(cx, limit=50, failed_only=False):
    init_table(cx)
    q = "SELECT message_sid, to_number, status, error_code, updated_at FROM sms_delivery_log"
    if failed_only:
        q += " WHERE status IN ('failed','undelivered')"
    q += " ORDER BY updated_at DESC LIMIT ?"
    return [{"message_sid": r[0], "to_number": r[1], "status": r[2],
             "error_code": r[3], "updated_at": r[4]}
            for r in cx.execute(q, (limit,)).fetchall()]
