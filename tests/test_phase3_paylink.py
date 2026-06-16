"""Phase 3 customer pay-link: order-invoice tokens (scope + expiry), the
claimed-payment / invoice-sent order helpers, send_invoice gating, and the
HTML-email extension."""
import base64
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _orders_db():
    from dashboard import orders as O
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    return O, cx


# ── invoice tokens ───────────────────────────────────────────────────────────

def test_invoice_token_roundtrip_and_single_order_scope(tmp_path):
    from dashboard import practitioner_portal as PP
    db = str(tmp_path / "tok.db")
    t1 = PP.create_order_invoice_token(42, db_path=db)
    t2 = PP.create_order_invoice_token(99, db_path=db)
    assert PP.order_id_from_invoice_token(t1, db_path=db) == "42"
    assert PP.order_id_from_invoice_token(t2, db_path=db) == "99"
    assert PP.order_id_from_invoice_token("not-a-real-token", db_path=db) is None
    # purpose-scoped: a practitioner session token is NOT a valid invoice token
    sess = PP.create_session_token(7, db_path=db)
    assert PP.order_id_from_invoice_token(sess, db_path=db) is None


def test_invoice_token_expires(tmp_path):
    from dashboard import practitioner_portal as PP
    db = str(tmp_path / "tok.db")
    t = PP.create_order_invoice_token(5, ttl_days=30, db_path=db)
    later = datetime.now(timezone.utc) + timedelta(days=31)
    assert PP.order_id_from_invoice_token(t, now=later, db_path=db) is None


# ── payment helpers ──────────────────────────────────────────────────────────

def test_claimed_payment_does_not_mark_paid_or_fulfill():
    O, cx = _orders_db()
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-1",
                         status="proposed", total_cents=5000, email="a@b.com")
    assert O.set_order_payment_claimed(cx, oid, method="zelle")
    o = O.get_order(cx, oid)
    assert o["pay_status"] == "claimed"
    assert o["pay_method"] == "zelle"
    assert o["status"] == "proposed"  # stays pre-fulfillment until OWNER confirms


def test_mark_invoice_sent():
    O, cx = _orders_db()
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-2",
                         status="proposed", total_cents=100)
    assert O.mark_invoice_sent(cx, oid)
    assert O.get_order(cx, oid)["invoice_sent_at"]


# ── send_invoice action gating ───────────────────────────────────────────────

def test_send_invoice_blocked_when_flag_off(monkeypatch):
    O, cx = _orders_db()
    monkeypatch.delenv("INVOICE_PAYLINK_ENABLED", raising=False)
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-3",
                         status="proposed", total_cents=100, email="a@b.com")
    with pytest.raises(ValueError, match="disabled"):
        O._send_invoice_exec({"order_id": oid}, {"cx": cx})


def test_send_invoice_requires_proposed_or_confirmed(monkeypatch):
    O, cx = _orders_db()
    monkeypatch.setenv("INVOICE_PAYLINK_ENABLED", "1")
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-4",
                         status="new", total_cents=100, email="a@b.com")
    with pytest.raises(ValueError, match="proposed/confirmed"):
        O._send_invoice_exec({"order_id": oid}, {"cx": cx})


# ── HTML email extension ─────────────────────────────────────────────────────

def test_send_email_html_is_multipart_and_plain_still_works(monkeypatch):
    from dashboard import inbox

    captured = {}

    class _Send:
        def execute(self):
            return {"id": "1", "threadId": "2"}

    class _Msgs:
        def send(self, userId, body):
            captured["raw"] = body["raw"]
            return _Send()

    class _Users:
        def messages(self):
            return _Msgs()

    class _Svc:
        def users(self):
            return _Users()

    monkeypatch.setattr(inbox, "_get_gmail_service", lambda: _Svc())

    inbox.send_email("a@b.com", "Hi", "plain body", html="<p>hello world</p>")
    raw = base64.urlsafe_b64decode(captured["raw"]).decode()
    assert "multipart/alternative" in raw and "text/html" in raw

    inbox.send_email("a@b.com", "Hi", "plain only")
    raw2 = base64.urlsafe_b64decode(captured["raw"]).decode()
    assert "text/html" not in raw2 and "multipart" not in raw2
