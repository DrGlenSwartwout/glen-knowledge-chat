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


def test_record_payment_on_cart_new_order():
    # A portal-reorder lands unpaid in 'new' (Cart). Paying in person by check/cash
    # records on the 'new' order: marks it paid, stays 'new' (now in the Paid lane).
    O, cx = _orders_db()
    oid = O.upsert_order(cx, source="portal-reorder", external_ref="QB-CART",
                         status="new", total_cents=65000, email="k@b.com")
    res = O._record_payment_exec({"order_id": oid, "method": "Check"}, {"cx": cx})
    assert res["pay_status"] == "paid"
    o = O.get_order(cx, oid)
    assert o["pay_status"] == "paid" and o["pay_method"] == "Check" and o["status"] == "new"


def test_record_payment_rejects_already_paid():
    O, cx = _orders_db()
    oid = O.upsert_order(cx, source="portal-reorder", external_ref="QB-PAID",
                         status="new", total_cents=5000, email="k@b.com")
    O.set_order_payment(cx, oid, method="Cash", amount_cents=5000)
    with pytest.raises(ValueError, match="already marked paid"):
        O._record_payment_exec({"order_id": oid, "method": "Cash"}, {"cx": cx})


def test_record_payment_rejects_shipped():
    O, cx = _orders_db()
    oid = O.upsert_order(cx, source="in-house", external_ref="SHIP-1",
                         status="shipped", total_cents=5000, email="k@b.com")
    with pytest.raises(ValueError, match="before it ships"):
        O._record_payment_exec({"order_id": oid, "method": "Check"}, {"cx": cx})


# ── send_invoice action gating ───────────────────────────────────────────────

def test_send_invoice_blocked_when_flag_off(monkeypatch):
    O, cx = _orders_db()
    monkeypatch.delenv("INVOICE_PAYLINK_ENABLED", raising=False)
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-3",
                         status="proposed", total_cents=100, email="a@b.com")
    with pytest.raises(ValueError, match="disabled"):
        O._send_invoice_exec({"order_id": oid}, {"cx": cx})


def test_send_invoice_rejects_fulfilled_order(monkeypatch):
    # A 'new' order is now sendable (Cart + Paid lanes); once it ships, it's not.
    O, cx = _orders_db()
    monkeypatch.setenv("INVOICE_PAYLINK_ENABLED", "1")
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-4",
                         status="shipped", total_cents=100, email="a@b.com")
    with pytest.raises(ValueError, match="before it ships"):
        O._send_invoice_exec({"order_id": oid}, {"cx": cx})


def _capture_send_email(monkeypatch):
    """Patch dashboard.inbox.send_email and return the dict it captures."""
    from dashboard import inbox
    cap = {}

    def _fake(to, subject, plain, *, from_name=None, html=None):
        cap.update(to=to, subject=subject, plain=plain, from_name=from_name, html=html)
        return {"id": "1"}

    monkeypatch.setattr(inbox, "send_email", _fake)
    return cap


def test_send_invoice_paid_order_sends_receipt(monkeypatch):
    O, cx = _orders_db()
    monkeypatch.setenv("INVOICE_PAYLINK_ENABLED", "1")
    cap = _capture_send_email(monkeypatch)
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-PAID",
                         status="confirmed", total_cents=5000, email="a@b.com")
    O.set_order_payment(cx, oid, method="Zelle", amount_cents=5000)  # -> new + paid
    res = O._send_invoice_exec({"order_id": oid}, {"cx": cx})
    assert "receipt" in cap["subject"].lower()
    assert "paid in full" in cap["plain"].lower()
    assert "pay here" not in cap["plain"].lower()
    assert res["message"].startswith("Receipt ")
    assert O.get_order(cx, oid)["invoice_sent_at"]


def test_send_invoice_unpaid_cart_sends_pay_link(monkeypatch):
    O, cx = _orders_db()
    monkeypatch.setenv("INVOICE_PAYLINK_ENABLED", "1")
    cap = _capture_send_email(monkeypatch)
    oid = O.upsert_order(cx, source="portal-reorder", external_ref="INH-CART",
                         status="new", total_cents=100, email="a@b.com")
    res = O._send_invoice_exec({"order_id": oid}, {"cx": cx})
    assert "invoice" in cap["subject"].lower() and "receipt" not in cap["subject"].lower()
    assert "pay here" in cap["plain"].lower()
    assert res["message"].startswith("Invoice ")


def test_paid_order_auto_publishes_portal_receipt():
    # A paid order auto-gets a token + portal_published so it's a clickable receipt
    # in the client's portal History — via BOTH paid-setters, no operator click.
    O, cx = _orders_db()
    for col, ddl in (("portal_published", "INTEGER NOT NULL DEFAULT 0"), ("invoice_token", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE orders ADD COLUMN {col} {ddl}")
        except Exception:
            pass
    a = O.upsert_order(cx, source="in-house", external_ref="AUTO-1",
                       status="proposed", total_cents=5000, email="a@b.com")
    O.set_order_payment(cx, a, method="Zelle", amount_cents=5000)
    ra = cx.execute("SELECT portal_published, invoice_token FROM orders WHERE id=?", (a,)).fetchone()
    assert ra[0] == 1 and ra[1]
    b = O.upsert_order(cx, source="in-house", external_ref="AUTO-2",
                       status="done", total_cents=100, email="a@b.com")
    O.mark_order_paid_keep_status(cx, b, method="card", amount_cents=100)
    rb = cx.execute("SELECT portal_published, invoice_token FROM orders WHERE id=?", (b,)).fetchone()
    assert rb[0] == 1 and rb[1]


def test_send_invoice_also_publishes_to_portal(monkeypatch):
    # Send & Publish: send_invoice now also surfaces the invoice on the client's portal.
    O, cx = _orders_db()
    monkeypatch.setenv("INVOICE_PAYLINK_ENABLED", "1")
    _capture_send_email(monkeypatch)
    for col, ddl in (("portal_published", "INTEGER NOT NULL DEFAULT 0"), ("invoice_token", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE orders ADD COLUMN {col} {ddl}")
        except Exception:
            pass
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-PUB",
                         status="proposed", total_cents=5000, email="a@b.com")
    O._send_invoice_exec({"order_id": oid}, {"cx": cx})
    r = cx.execute("SELECT portal_published, invoice_token FROM orders WHERE id=?", (oid,)).fetchone()
    assert r[0] == 1 and r[1]                     # published to portal with a pay token


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


def test_card_pay_stripe_failure_returns_502_not_500(monkeypatch, tmp_path):
    """A Stripe error on the card path must degrade to a clean JSON 502, never a 500."""
    import importlib, sqlite3
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        app_module = importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")

    db = str(tmp_path / "pay.db")
    # Tokens (chat_log.db) + orders (LOG_DB) share one sqlite file for the test.
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import practitioner_portal as PP
    monkeypatch.setattr(PP, "_LOG_DB", Path(db))
    from dashboard import orders as O

    cx = sqlite3.connect(db)
    O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-PAYTEST",
                         status="proposed", email="t@example.com", total_cents=14314)
    cx.commit(); cx.close()
    token = PP.create_order_invoice_token(oid, db_path=db)

    # Enable the flag + Stripe, but make the Stripe session call blow up.
    monkeypatch.setattr(app_module, "INVOICE_PAYLINK_ENABLED", True)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True)
    from dashboard import stripe_pay
    def _boom(*a, **k):
        raise RuntimeError("stripe 401 unauthorized")
    monkeypatch.setattr(stripe_pay, "create_checkout_session", _boom)

    client = app_module.app.test_client()
    r = client.post(f"/api/invoice/{token}/pay", json={"method": "card"})
    assert r.status_code == 502, f"expected clean 502, got {r.status_code}"
    body = r.get_json()
    assert body and body.get("ok") is False
    assert "unavailable" in body.get("error", "").lower()
