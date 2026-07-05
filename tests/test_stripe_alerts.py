"""Throttled Stripe-failure alerting: record/throttle/email + recent count."""
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import stripe_alerts as SA  # noqa: E402
from dashboard import inbox as INBOX  # noqa: E402


def _db():
    cx = sqlite3.connect(":memory:")
    SA.init_stripe_alerts_table(cx)
    return cx


def _t(mins=0):
    return (datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=mins)).isoformat()


def test_first_failure_emails_then_throttles(monkeypatch):
    sent = []
    monkeypatch.setattr(INBOX, "send_email", lambda *a, **k: sent.append((a, k)) or {"id": "x"})
    cx = _db()

    r1 = SA.record_failure(cx, "retail", "boom", throttle_min=20, now=_t(0))
    assert r1["emailed"] is True and r1["id"] > 0
    assert len(sent) == 1

    # second failure within the window -> recorded but NOT emailed
    r2 = SA.record_failure(cx, "invoice-card", "boom2", throttle_min=20, now=_t(5))
    assert r2["emailed"] is False
    assert len(sent) == 1

    # third failure beyond the window -> emails again
    r3 = SA.record_failure(cx, "reorder", "boom3", throttle_min=20, now=_t(25))
    assert r3["emailed"] is True
    assert len(sent) == 2

    # all three rows are persisted regardless of email throttling
    assert cx.execute("SELECT COUNT(*) FROM stripe_failures").fetchone()[0] == 3


def test_record_never_raises_when_email_fails(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("gmail down")
    monkeypatch.setattr(INBOX, "send_email", _boom)
    cx = _db()
    r = SA.record_failure(cx, "retail", "stripe 401", throttle_min=20, now=_t(0))
    assert r["emailed"] is False
    assert r["id"] > 0  # still recorded
    assert cx.execute("SELECT COUNT(*) FROM stripe_failures").fetchone()[0] == 1


def test_recent_failure_count_window(monkeypatch):
    monkeypatch.setattr(INBOX, "send_email", lambda *a, **k: {"id": "x"})
    cx = _db()
    SA.record_failure(cx, "retail", "e", now=_t(0), notify=False)
    SA.record_failure(cx, "retail", "e", now=_t(20), notify=False)
    # one old failure outside a 30m window from t=60
    assert SA.recent_failure_count(cx, minutes=30, now=_t(60)) == 0
    assert SA.recent_failure_count(cx, minutes=30, now=_t(25)) == 2
    assert SA.recent_failure_count(cx, minutes=30, now=_t(45)) == 1


def test_recent_count_no_table_is_zero():
    cx = sqlite3.connect(":memory:")  # table not initialized
    assert SA.recent_failure_count(cx, minutes=30) == 0


def test_record_returns_persisted_and_db_path(tmp_path, monkeypatch):
    """The return exposes WHERE the row landed + a post-commit read-back, so an
    'alerted but not in the console' incident is self-diagnosing."""
    monkeypatch.setattr(INBOX, "send_email", lambda *a, **k: {"id": "x"})
    dbfile = tmp_path / "chat_log.db"
    cx = sqlite3.connect(str(dbfile))
    SA.init_stripe_alerts_table(cx)
    r = SA.record_failure(cx, "invoice-card", "stripe 401 unauthorized",
                          throttle_min=20, now=_t(0))
    assert r["id"] > 0
    assert r["persisted"] is True
    assert r["db_path"] == str(dbfile)  # absolute file the row committed to


def test_alert_email_is_self_locating(monkeypatch):
    """The owner email carries host + ledger DB path + persistence status, so a
    non-durable write (email but no console row) is visible on the next incident."""
    sent = {}
    monkeypatch.setattr(INBOX, "send_email",
                        lambda to, subj, body, **k: sent.update(body=body) or {"id": "x"})
    import socket
    cx = _db()
    SA.record_failure(cx, "invoice-card", "stripe 401", throttle_min=20, now=_t(0))
    body = sent["body"]
    assert socket.gethostname() in body
    assert "Ledger DB:" in body
    assert "Persistence:" in body


class _Empty:
    def fetchone(self):
        return None


class _NoReadbackConn(sqlite3.Connection):
    """A connection whose post-commit read-back finds nothing — simulates a write
    that didn't land durably (email fires but the ledger stays empty)."""
    def execute(self, sql, *a):
        if sql.strip().startswith("SELECT 1 FROM stripe_failures WHERE id="):
            return _Empty()
        return super().execute(sql, *a)


def test_persisted_false_surfaces_in_email(monkeypatch):
    """If the read-back can't find the row (durability failure), the email says so
    instead of silently implying the failure reached the ledger."""
    captured = {}
    monkeypatch.setattr(INBOX, "send_email",
                        lambda to, subj, body, **k: captured.update(body=body) or {"id": "x"})
    cx = sqlite3.connect(":memory:", factory=_NoReadbackConn)
    SA.init_stripe_alerts_table(cx)
    r = SA.record_failure(cx, "invoice-card", "stripe 401", throttle_min=20, now=_t(0))
    assert r["persisted"] is False
    assert "NOT PERSISTED" in captured["body"]
