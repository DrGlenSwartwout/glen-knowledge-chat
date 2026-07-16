"""Task 5: _ship_founding_reservation paid-only conversion (Pattern II — off-session
charge-then-book inline). Drops the QBO invoice; charges subtotal+shipping (never
GET); books ONE line-faithful Sales Receipt inline right after a successful charge."""
import json
import sqlite3

import app as appmod
from dashboard import orders as _orders_mod
from dashboard import qbo_billing
from dashboard import subscriptions as subs


def _isolate_db(monkeypatch, tmp_path):
    """Point app.LOG_DB at a throwaway sqlite file and init both the orders and
    subscriptions/founding schemas, mirroring tests/test_biofield_checkout_paid_only.py's
    _isolate_db -- these tests don't stub _ingest_order, so the orders table must
    actually exist on the fresh db."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db)
    try:
        _orders_mod.init_orders_table(cx)
        subs.init_subscriptions_table(cx)
        subs.migrate_add_founding_columns(cx)
        subs.migrate_add_failed_count(cx)
        cx.commit()
    finally:
        cx.close()
    return db


def _make_sub(db):
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    try:
        sid = subs.create_founding_reservation(
            cx, email="f@x.com", stripe_customer_id="cus_1",
            stripe_payment_method_id="pm_1",
            items=[{"slug": "neuro-magnesium", "qty": 1}],
            ship_address={"state": "HI", "name": "Founder"},
            founding_slug="neuro-magnesium",
        )
        cx.commit()
        return subs.get(cx, sid)
    finally:
        cx.close()


def _fake_pc(**overrides):
    pc = {
        "qbo_lines": [{"name": "Neuro Magnesium", "amount": 80.0, "qty": 1}],
        "shipping_cents": 600,
        "discount_cents": 0,
        "points_redeemed_cents": 0,
        "priced": {"total_cents": 8675, "get_cents": 75, "subtotal_cents": 8600},
    }
    pc.update(overrides)
    return pc


def _boom_invoice(*a, **k):
    raise AssertionError("_ship_founding_reservation must not call create_invoice (paid-only)")


def test_ship_success_charges_subtotal_plus_shipping_no_invoice(monkeypatch, tmp_path):
    db = _isolate_db(monkeypatch, tmp_path)
    sub = _make_sub(db)

    monkeypatch.setattr(appmod, "_price_cart",
                        lambda items, *, ship, subscriber_tier_pct=None, **k: _fake_pc())
    monkeypatch.setattr(qbo_billing, "create_invoice", _boom_invoice)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt",
                        lambda *a, **k: {"Id": "SR1"})

    charged = {}

    def _fake_charge(customer_id, pm_id, amount_cents, *, description, metadata):
        charged["amount_cents"] = amount_cents
        return {"status": "succeeded", "id": "ch_1"}

    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session", _fake_charge)

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    try:
        res = appmod._ship_founding_reservation(cx, sub)
    finally:
        cx.close()

    assert res["charged"] is True
    # Charge basis: subtotal + shipping = (total_cents - get_cents) + shipping_cents
    # = (8675 - 75) + 600 = 9200. GET (75) must never be charged.
    assert charged["amount_cents"] == 9200
    assert res["amount_cents"] == 9200


def test_ship_success_orders_keyed_on_charge_id_with_qbo_lines_persisted(monkeypatch, tmp_path):
    db = _isolate_db(monkeypatch, tmp_path)
    sub = _make_sub(db)

    monkeypatch.setattr(appmod, "_price_cart",
                        lambda items, *, ship, subscriber_tier_pct=None, **k: _fake_pc())
    monkeypatch.setattr(qbo_billing, "create_invoice", _boom_invoice)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt", lambda *a, **k: {"Id": "SR1"})
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "ch_1"})

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    try:
        res = appmod._ship_founding_reservation(cx, sub)
    finally:
        cx.close()
    assert res["charged"] is True

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    try:
        order = _orders_mod.find_order_by_external_ref(cx, "ch_1")
    finally:
        cx.close()
    assert order is not None
    assert order["source"] == "reorder"  # qualifies for delivery -> coaching window
    payload = json.loads(order["qbo_lines_json"])
    assert payload["lines"]  # merch line(s) + shipping line, line-faithful
    names = [l.get("name") for l in payload["lines"]]
    assert "Shipping (USPS)" in names


def test_ship_success_books_exactly_one_sales_receipt(monkeypatch, tmp_path):
    db = _isolate_db(monkeypatch, tmp_path)
    sub = _make_sub(db)

    monkeypatch.setattr(appmod, "_price_cart",
                        lambda items, *, ship, subscriber_tier_pct=None, **k: _fake_pc())
    monkeypatch.setattr(qbo_billing, "create_invoice", _boom_invoice)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    receipt_calls = []

    def _fake_receipt(cust, lines, *, discount_cents=0, tax_cents=0, email_to=None):
        receipt_calls.append((cust, lines))
        return {"Id": "SR1"}

    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _fake_receipt)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "ch_1"})

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    try:
        appmod._ship_founding_reservation(cx, sub)
    finally:
        cx.close()

    assert len(receipt_calls) == 1

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    try:
        order = _orders_mod.find_order_by_external_ref(cx, "ch_1")
    finally:
        cx.close()
    assert order["qbo_sales_receipt_id"] == "SR1"


def test_ship_failed_charge_no_order_no_booking(monkeypatch, tmp_path):
    db = _isolate_db(monkeypatch, tmp_path)
    sub = _make_sub(db)

    monkeypatch.setattr(appmod, "_price_cart",
                        lambda items, *, ship, subscriber_tier_pct=None, **k: _fake_pc())
    monkeypatch.setattr(qbo_billing, "create_invoice", _boom_invoice)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    def _boom_receipt(*a, **k):
        raise AssertionError("no Sales Receipt on a failed charge")

    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _boom_receipt)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "failed", "id": None})

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    try:
        res = appmod._ship_founding_reservation(cx, sub)
    finally:
        cx.close()

    assert res["charged"] is False

    cx = sqlite3.connect(db)
    try:
        n = cx.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    finally:
        cx.close()
    assert n == 0


def test_ship_rerun_same_charge_id_books_no_second_receipt(monkeypatch, tmp_path):
    """Simulates the atomic-claim guard: two separate invocations that happen to
    settle on the same Stripe charge id must never produce two Sales Receipts."""
    db = _isolate_db(monkeypatch, tmp_path)
    sub = _make_sub(db)

    monkeypatch.setattr(appmod, "_price_cart",
                        lambda items, *, ship, subscriber_tier_pct=None, **k: _fake_pc())
    monkeypatch.setattr(qbo_billing, "create_invoice", _boom_invoice)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    receipt_calls = []

    def _fake_receipt(cust, lines, *, discount_cents=0, tax_cents=0, email_to=None):
        receipt_calls.append(1)
        return {"Id": f"SR{len(receipt_calls)}"}

    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _fake_receipt)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "ch_dup"})

    for _ in range(2):
        cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
        try:
            appmod._ship_founding_reservation(cx, sub)
        finally:
            cx.close()

    assert len(receipt_calls) == 1

    cx = sqlite3.connect(db)
    try:
        n = cx.execute(
            "SELECT COUNT(*) FROM orders WHERE external_ref=?", ("ch_dup",)).fetchone()[0]
    finally:
        cx.close()
    assert n == 1  # idempotent upsert on (source, external_ref) — one order, one receipt
