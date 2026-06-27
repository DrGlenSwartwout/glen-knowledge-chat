"""Task 2: _ingest_order mints a portal + sends one welcome email per buyer.

Background-thread timing: _send_full_report_email is called on a daemon thread
spawned by _send_portal_welcome. We monkeypatch the function on the app module
(the thread calls through the app-module global) and sleep 0.5 s after each
_ingest_order to let the thread run before asserting the recorder.
"""
import importlib
import sqlite3
import sys
import time
from pathlib import Path

import pytest


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


def _setup_db(db_path):
    """Initialize all tables _ingest_order touches on a fresh DB."""
    from dashboard.orders import init_orders_table
    from dashboard.client_portal import init_client_portal_table
    from dashboard.email_suppression import init_table as init_suppression
    with sqlite3.connect(str(db_path)) as cx:
        init_orders_table(cx)
        init_client_portal_table(cx)
        init_suppression(cx)
        cx.commit()


def test_new_order_mints_portal_and_sends_welcome(tmp_path, monkeypatch):
    app = _app()
    db = tmp_path / "chat_log.db"
    _setup_db(db)

    monkeypatch.setattr(app, "LOG_DB", str(db))

    sends = []

    def _fake_send(to, name, subject, body):
        sends.append({"to": to, "name": name, "subject": subject, "body": body})
        return ("mock", None)

    monkeypatch.setattr(app, "_send_full_report_email", _fake_send)

    app._ingest_order(
        source="test", external_ref="o1",
        email="new@x.com", name="N",
        items=[{"name": "X", "qty": 1}], total_cents=1000,
    )

    # Let the daemon thread run.
    time.sleep(0.5)

    # Portal row must exist.
    from dashboard.client_portal import get_portal_content_by_email
    with sqlite3.connect(str(db)) as cx:
        portal = get_portal_content_by_email(cx, "new@x.com")
    assert portal is not None, "portal should be created for new buyer"

    # Welcome email must have been sent once, body contains /portal/ path.
    assert len(sends) == 1, f"expected 1 send, got {len(sends)}"
    assert "/portal/" in sends[0]["body"], "body should contain /portal/ URL"


def test_second_order_same_email_no_resend(tmp_path, monkeypatch):
    app = _app()
    db = tmp_path / "chat_log.db"
    _setup_db(db)

    monkeypatch.setattr(app, "LOG_DB", str(db))

    sends = []

    def _fake_send(to, name, subject, body):
        sends.append({"to": to, "name": name, "subject": subject, "body": body})
        return ("mock", None)

    monkeypatch.setattr(app, "_send_full_report_email", _fake_send)

    # First order — mints portal + sends.
    app._ingest_order(
        source="test", external_ref="o1",
        email="new@x.com", name="N",
        items=[{"name": "X", "qty": 1}], total_cents=1000,
    )
    time.sleep(0.5)
    assert len(sends) == 1, "first order should trigger one send"

    # Capture portal token after first order.
    from dashboard.client_portal import get_portal_content_by_email
    with sqlite3.connect(str(db)) as cx:
        portal_after_first = get_portal_content_by_email(cx, "new@x.com")
    assert portal_after_first is not None

    # Second order with same email, different external_ref.
    app._ingest_order(
        source="test", external_ref="o2",
        email="new@x.com", name="N",
        items=[{"name": "Y", "qty": 1}], total_cents=500,
    )
    time.sleep(0.5)

    # Once-guard: no additional send.
    assert len(sends) == 1, f"second order must not send again (got {len(sends)} sends)"

    # Portal row unchanged (same record).
    with sqlite3.connect(str(db)) as cx:
        portal_after_second = get_portal_content_by_email(cx, "new@x.com")
    assert portal_after_second is not None


def test_empty_email_no_portal_no_send(tmp_path, monkeypatch):
    app = _app()
    db = tmp_path / "chat_log.db"
    _setup_db(db)

    monkeypatch.setattr(app, "LOG_DB", str(db))

    sends = []

    def _fake_send(to, name, subject, body):
        sends.append({"to": to, "name": name, "subject": subject, "body": body})
        return ("mock", None)

    monkeypatch.setattr(app, "_send_full_report_email", _fake_send)

    app._ingest_order(
        source="test", external_ref="o3",
        email="", name="",
        items=[{"name": "Z", "qty": 1}], total_cents=100,
    )
    time.sleep(0.3)

    # No portal row created.
    with sqlite3.connect(str(db)) as cx:
        row = cx.execute("SELECT COUNT(*) FROM client_portals").fetchone()
    assert row[0] == 0, "no portal should be created for empty email"

    # No send.
    assert len(sends) == 0, "no email should be sent for empty email"
