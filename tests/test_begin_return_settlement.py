"""Characterization tests for /begin/checkout-return per-kind settlement.

These PIN the CURRENT (pre-refactor) settlement behavior of the redirect handler
so the Task 3 extraction (inline per-kind settlers -> named deps dispatched via
dashboard.order_settlement) is provably behavior-preserving. They must pass
UNCHANGED before and after the refactor.

Patterns (DB isolation, monkeypatch get_session/get_payment_intent, seeding,
spies) mirror tests/test_begin_checkout_paid_only.py's pinning test.
"""

import json
import sqlite3

import app
from dashboard import orders as O


def _client():
    return app.app.test_client()


def _isolate_db(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    cx = sqlite3.connect(db)
    try:
        O.init_orders_table(cx)
        cx.commit()
    finally:
        cx.close()
    return db


PRODUCT_SLUG = "brain-boost"


def _seed_order(db, token, *, email, channel="retail"):
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    try:
        oid = O.upsert_order(
            cx, source="funnel", external_ref=token, email=email, name="T",
            items=[{"slug": PRODUCT_SLUG, "qty": 1}], total_cents=7000, address={},
            channel=channel, get_cents=0, discount_cents=0, points_redeemed_cents=0,
            shipping_cents=0, status="new")
        O.set_order_qbo_lines(cx, token, {
            "lines": [{"name": "Brain Boost", "amount": 70.0, "qty": 1}],
            "discount_cents": 0, "tax_cents": 0})
    finally:
        cx.close()
    return oid


def _spy_common(monkeypatch, calls):
    """Spy the common settlers + booking so a test never hits QBO/points DBs."""
    monkeypatch.setattr(app, "_settle_order_points",
                        lambda order, *, order_ref: calls["points"].append(order_ref))
    monkeypatch.setattr(app, "_settle_referral",
                        lambda order, *, order_ref: calls["referral"].append(order_ref))
    import dashboard.qbo_sale as _qs
    monkeypatch.setattr(_qs, "book_sale_on_payment",
                        lambda cx, order: calls["booked"].append(order.get("external_ref")))


def _fresh_calls():
    return {"points": [], "referral": [], "booked": [], "create": [],
            "wallet": [], "seed": [], "fulfill": []}


def test_retail_settles_points_referral_and_books_only(monkeypatch, tmp_path):
    db = _isolate_db(monkeypatch, tmp_path)
    token = "chk_retail_1"
    _seed_order(db, token, email="r@b.com")
    calls = _fresh_calls()
    _spy_common(monkeypatch, calls)

    # guards: none of the per-kind settlers should fire for retail
    import dashboard.subscriptions as _subs
    monkeypatch.setattr(_subs, "create",
                        lambda *a, **k: calls["create"].append(k.get("email")) or 1)
    import dashboard.wallet as _wal
    monkeypatch.setattr(_wal, "earn_dropship_margin",
                        lambda *a, **k: calls["wallet"].append(a) or 1)
    import dashboard.biofield_store as _bf
    monkeypatch.setattr(_bf, "seed_paid",
                        lambda *a, **k: calls["seed"].append(k.get("order_ref")))

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 7000, "payment_intent": "pi_r",
        "metadata": {"kind": "retail", "invoice_id": token, "customer_id": "",
                     "slug": PRODUCT_SLUG}})

    r = _client().get("/begin/checkout-return?session_id=sess_retail")
    assert r.status_code in (301, 302)

    assert calls["points"] == [token]
    assert calls["referral"] == [token]
    assert calls["booked"] == [token]
    assert calls["create"] == []
    assert calls["wallet"] == []
    assert calls["seed"] == []


def test_subscribe_settles_common_and_creates_subscription(monkeypatch, tmp_path):
    db = _isolate_db(monkeypatch, tmp_path)
    token = "chk_sub_1"
    _seed_order(db, token, email="s@b.com")
    calls = _fresh_calls()
    _spy_common(monkeypatch, calls)

    import dashboard.subscriptions as _subs
    # spy create() -- current code calls bare create(); after the refactor
    # create_once() delegates to create(), so this spy fires in BOTH worlds.
    monkeypatch.setattr(_subs, "create",
                        lambda cx, **k: calls["create"].append(k.get("email")) or 1)

    import dashboard.biofield_store as _bf
    monkeypatch.setattr(_bf, "seed_paid",
                        lambda *a, **k: calls["seed"].append(k.get("order_ref")))
    import dashboard.wallet as _wal
    monkeypatch.setattr(_wal, "earn_dropship_margin",
                        lambda *a, **k: calls["wallet"].append(a) or 1)

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 7000, "payment_intent": "pi_s",
        "metadata": {"kind": "subscribe", "invoice_id": token, "customer_id": "",
                     "email": "s@b.com", "cadence_months": "1",
                     "items": json.dumps([{"slug": PRODUCT_SLUG, "qty": 1}]),
                     "ship": json.dumps({"state": "CA"})}})
    monkeypatch.setattr(sp, "get_payment_intent", lambda pi: {
        "customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})

    r = _client().get("/begin/checkout-return?session_id=sess_sub")
    assert r.status_code in (301, 302)

    assert calls["points"] == [token]
    assert calls["referral"] == [token]
    assert calls["create"] == ["s@b.com"]
    assert calls["seed"] == []
    assert calls["wallet"] == []


def test_client_credits_wallet_margin_and_settles_points(monkeypatch, tmp_path):
    db = _isolate_db(monkeypatch, tmp_path)
    token = "chk_client_1"
    _seed_order(db, token, email="c@b.com", channel="dispensary")
    calls = _fresh_calls()
    _spy_common(monkeypatch, calls)

    import dashboard.wallet as _wal
    monkeypatch.setattr(_wal, "earn_dropship_margin",
                        lambda pid, marg, **k: calls["wallet"].append((pid, marg)) or 1)
    # practitioner_portal.record_dispensary_order may or may not exist / may touch
    # external state -- neutralize it so the client block runs cleanly.
    monkeypatch.setattr(app._pp, "record_dispensary_order",
                        lambda *a, **k: None, raising=False)

    import dashboard.subscriptions as _subs
    monkeypatch.setattr(_subs, "create",
                        lambda *a, **k: calls["create"].append(k.get("email")) or 1)
    import dashboard.biofield_store as _bf
    monkeypatch.setattr(_bf, "seed_paid",
                        lambda *a, **k: calls["seed"].append(k.get("order_ref")))

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 7000, "payment_intent": "pi_c",
        "metadata": {"kind": "client", "invoice_id": token, "customer_id": "",
                     "practitioner_id": "prac_1", "margin_cents": "1500"}})

    r = _client().get("/begin/checkout-return?session_id=sess_client")
    assert r.status_code in (301, 302)

    # client passes the shared gate today -> common points fire (behavior-preserving),
    # AND its own wallet-margin credit fires.
    assert calls["wallet"] == [("prac_1", 1500)]
    assert calls["points"] == [token]
    assert calls["create"] == []
    assert calls["seed"] == []


def test_biofield_seeds_settles_common_and_fulfills(monkeypatch, tmp_path):
    db = _isolate_db(monkeypatch, tmp_path)
    token = "chk_bf_1"
    _seed_order(db, token, email="bf@b.com", channel="retail")
    calls = _fresh_calls()
    _spy_common(monkeypatch, calls)

    import dashboard.biofield_store as _bf
    monkeypatch.setattr(_bf, "seed_paid",
                        lambda cx, email, *, via, order_ref: calls["seed"].append(order_ref))
    monkeypatch.setattr(app, "_fulfill_biofield_program",
                        lambda sid: calls["fulfill"].append(sid))

    import dashboard.subscriptions as _subs
    monkeypatch.setattr(_subs, "create",
                        lambda *a, **k: calls["create"].append(k.get("email")) or 1)
    import dashboard.wallet as _wal
    monkeypatch.setattr(_wal, "earn_dropship_margin",
                        lambda *a, **k: calls["wallet"].append(a) or 1)

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 7000, "payment_intent": "pi_bf",
        "metadata": {"kind": "biofield", "invoice_id": token, "customer_id": "",
                     "email": "bf@b.com"}})

    r = _client().get("/begin/checkout-return?session_id=sess_bf")
    assert r.status_code in (301, 302)

    assert calls["seed"] == [token]
    assert calls["points"] == [token]
    assert calls["referral"] == [token]
    assert calls["fulfill"] == ["sess_bf"]
    assert calls["create"] == []
    assert calls["wallet"] == []
