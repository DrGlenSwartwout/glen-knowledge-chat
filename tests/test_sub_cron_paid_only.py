"""Task 6: cron_charge_subscriptions x2 paid-only conversion (Pattern II --
recurring off-session charge-then-book inline).

Both the 'membership' (flat-amount) branch and the 'subscription' (priced
product cart) branch of /api/cron/charge-subscriptions drop the QBO invoice
and book ONE line-faithful Sales Receipt inline right after a successful
off-session charge. The charge amount itself is unchanged in both cases
(flat amount_cents for membership; subtotal+shipping, never GET, for the
product cart -- both already correct before this change).

Mirrors the harness in test_membership_charge_cron.py / test_subscriptions_cron.py:
seeds against the real LOG_DB (Doppler dev), stubs stripe + qbo_billing, posts to
the cron endpoint with the secret header.
"""
import os
import sqlite3

import app as appmod
from dashboard import orders as _orders_mod
from dashboard import qbo_billing
from dashboard import subscriptions as subs

MBR_EMAIL = "sub-cron-po-mbr@example.com"
PROD_EMAIL = "sub-cron-po-prod@example.com"


def _cron_secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _headers():
    return {"X-Cron-Secret": _cron_secret()}


def _enable(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _cron_secret())


def _boom_invoice(*a, **k):
    raise AssertionError("cron_charge_subscriptions must not call create_invoice (paid-only)")


def _boom_record_payment(*a, **k):
    raise AssertionError("cron_charge_subscriptions must not call record_payment (paid-only)")


def _mock_qb_no_invoice(monkeypatch):
    monkeypatch.setattr(appmod.qb, "create_invoice", _boom_invoice)
    monkeypatch.setattr(appmod.qb, "record_payment", _boom_record_payment)
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})


def _mock_booking(monkeypatch, receipt_calls, receipt_id="SR1"):
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    def _fake_receipt(cust, lines, *, discount_cents=0, tax_cents=0, email_to=None):
        receipt_calls.append({"lines": lines, "discount_cents": discount_cents,
                              "tax_cents": tax_cents})
        return {"Id": receipt_id if not isinstance(receipt_id, list)
                else receipt_id[len(receipt_calls) - 1]}

    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _fake_receipt)


def _wipe_order(cx, source, external_ref):
    """upsert_order is idempotent on (source, external_ref) against the REAL
    dev LOG_DB -- a leftover row from a prior run would carry forward its
    already-set qbo_sales_receipt_id and silently short-circuit the receipt
    booking on this run. Always start each test from a clean slate."""
    cx.execute("DELETE FROM orders WHERE source=? AND external_ref=?", (source, external_ref))
    cx.commit()


def _cleanup_membership(cx):
    cx.execute("DELETE FROM subscriptions WHERE email=?", (MBR_EMAIL,))
    cx.commit()


def _cleanup_product(cx):
    cx.execute("DELETE FROM subscriptions WHERE email=?", (PROD_EMAIL,))
    cx.commit()


def _seed_membership(cx, *, amount_cents=9900, next_charge_date="2000-01-01"):
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (MBR_EMAIL,))
    cx.commit()
    sid = subs.create_membership(
        cx, email=MBR_EMAIL, stripe_customer_id="cus_mbr_po",
        stripe_payment_method_id="pm_mbr_po", amount_cents=amount_cents,
        next_charge_date=next_charge_date, cadence_months=1,
    )
    cx.commit()
    return sid


def _seed_product_sub(cx, *, next_charge_date="2000-01-01"):
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (PROD_EMAIL,))
    cx.commit()
    sid = subs.create(
        cx, email=PROD_EMAIL, stripe_customer_id="cus_prod_po",
        stripe_payment_method_id="pm_prod_po",
        items=[{"slug": "x", "qty": 1}], cadence_months=1,
        ship_address={"state": "CA"}, next_charge_date=next_charge_date,
    )
    cx.commit()
    return sid


def _mock_price_cart(monkeypatch):
    monkeypatch.setattr(
        appmod, "_price_cart",
        lambda cart, **k: {
            "priced": {"subtotal_cents": 5000, "total_cents": 5000, "get_cents": 0},
            "qbo_lines": [{"name": "X", "amount": 50.0, "qty": 1}],
            "items_rec": [{"name": "X", "qty": 1, "desc": "X"}],
            "discount_cents": 0,
            "points_redeemed_cents": 0,
            "shipping_cents": 1265,  # charge = subtotal + shipping = 6265
        },
    )


# ── Membership branch (flat amount_cents) ────────────────────────────────────

def test_membership_charge_no_invoice_books_one_receipt(monkeypatch):
    _enable(monkeypatch)
    _mock_qb_no_invoice(monkeypatch)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "ch_mbr_1"})
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))
    receipt_calls = []
    _mock_booking(monkeypatch, receipt_calls, receipt_id="SR_MBR_1")

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    try:
        _wipe_order(cx, "membership", "ch_mbr_1")
        _seed_membership(cx, amount_cents=9900, next_charge_date="2000-01-01")
        c = appmod.app.test_client()
        r = c.post("/api/cron/charge-subscriptions", headers=_headers())
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        assert body["charged"] >= 1

        assert len(receipt_calls) == 1
        # One line, amount == the flat charge, no shipping/GET split.
        assert receipt_calls[0]["lines"][0]["amount"] == 99.00
        assert receipt_calls[0]["discount_cents"] == 0

        order = _orders_mod.find_order_by_external_ref(cx, "ch_mbr_1")
        assert order is not None
        assert order["source"] == "membership"
        assert order["total_cents"] == 9900
        assert order["qbo_sales_receipt_id"] == "SR_MBR_1"
    finally:
        _cleanup_membership(cx)
        _wipe_order(cx, "membership", "ch_mbr_1")
        cx.close()


def test_membership_failed_charge_books_none_and_bumps_failed_count(monkeypatch):
    _enable(monkeypatch)
    _mock_qb_no_invoice(monkeypatch)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "failed", "id": None,
                                        "decline_code": "insufficient_funds"})
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))
    receipt_calls = []

    def _boom_receipt(*a, **k):
        raise AssertionError("no Sales Receipt on a failed charge")

    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _boom_receipt)

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    try:
        sid = _seed_membership(cx, amount_cents=9900, next_charge_date="2000-01-01")
        c = appmod.app.test_client()
        r = c.post("/api/cron/charge-subscriptions", headers=_headers())
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["failed"] >= 1

        row = subs.get(cx, sid)
        assert row.get("failed_count", 0) >= 1
        assert row["order_count"] == 0
        assert receipt_calls == []
    finally:
        _cleanup_membership(cx)
        cx.close()


def test_membership_rerun_same_charge_id_books_no_second_receipt(monkeypatch):
    """Simulates a retry that settles on the same Stripe charge id: the atomic
    booking claim must prevent a second Sales Receipt for the same order."""
    _enable(monkeypatch)
    _mock_qb_no_invoice(monkeypatch)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "ch_mbr_dup"})
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))
    receipt_calls = []
    _mock_booking(monkeypatch, receipt_calls, receipt_id=["SR1", "SR2"])

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    try:
        _wipe_order(cx, "membership", "ch_mbr_dup")
        _seed_membership(cx, amount_cents=9900, next_charge_date="2000-01-01")
        c = appmod.app.test_client()
        r = c.post("/api/cron/charge-subscriptions", headers=_headers())
        assert r.status_code == 200, r.data
        assert len(receipt_calls) == 1

        # Re-run the whole booking tail directly against the same external_ref
        # (a same-charge-id retry) -- the atomic claim must no-op the 2nd call.
        from dashboard import qbo_sale as _qsale
        order = _orders_mod.find_order_by_external_ref(cx, "ch_mbr_dup")
        assert order is not None
        _qsale.book_sale_on_payment(cx, dict(order))
        assert len(receipt_calls) == 1, "a duplicate booking attempt must not create a 2nd receipt"

        n = cx.execute(
            "SELECT COUNT(*) FROM orders WHERE external_ref=?", ("ch_mbr_dup",)).fetchone()[0]
        assert n == 1
    finally:
        _cleanup_membership(cx)
        _wipe_order(cx, "membership", "ch_mbr_dup")
        cx.close()


# ── Subscription branch (priced product cart) ────────────────────────────────

def test_subscription_charge_no_invoice_books_one_receipt(monkeypatch):
    _enable(monkeypatch)
    _mock_qb_no_invoice(monkeypatch)
    _mock_price_cart(monkeypatch)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "ch_prod_1"})
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))
    receipt_calls = []
    _mock_booking(monkeypatch, receipt_calls, receipt_id="SR_PROD_1")

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    try:
        _wipe_order(cx, "subscription", "ch_prod_1")
        _seed_product_sub(cx, next_charge_date="2000-01-01")
        c = appmod.app.test_client()
        r = c.post("/api/cron/charge-subscriptions", headers=_headers())
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        assert body["charged"] >= 1

        assert len(receipt_calls) == 1
        names = [l.get("name") for l in receipt_calls[0]["lines"]]
        assert "Shipping (USPS)" in names  # merch + shipping, line-faithful

        order = _orders_mod.find_order_by_external_ref(cx, "ch_prod_1")
        assert order is not None
        assert order["source"] == "subscription"
        # subtotal(5000) + shipping(1265) = 6265, GET excluded from the charge/order total
        assert order["total_cents"] == 6265
        assert order["qbo_sales_receipt_id"] == "SR_PROD_1"
    finally:
        _cleanup_product(cx)
        _wipe_order(cx, "subscription", "ch_prod_1")
        cx.close()


def test_subscription_failed_charge_books_none_and_bumps_failed_count(monkeypatch):
    _enable(monkeypatch)
    _mock_qb_no_invoice(monkeypatch)
    _mock_price_cart(monkeypatch)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "failed", "id": None,
                                        "decline_code": "insufficient_funds"})
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))

    def _boom_receipt(*a, **k):
        raise AssertionError("no Sales Receipt on a failed charge")

    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _boom_receipt)

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    try:
        sid = _seed_product_sub(cx, next_charge_date="2000-01-01")
        c = appmod.app.test_client()
        r = c.post("/api/cron/charge-subscriptions", headers=_headers())
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["failed"] >= 1

        row = subs.get(cx, sid)
        assert row["order_count"] == 0
        assert row.get("failed_count", 0) >= 1
    finally:
        _cleanup_product(cx)
        cx.close()


def test_subscription_rerun_same_charge_id_books_no_second_receipt(monkeypatch):
    _enable(monkeypatch)
    _mock_qb_no_invoice(monkeypatch)
    _mock_price_cart(monkeypatch)
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "ch_prod_dup"})
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))
    receipt_calls = []
    _mock_booking(monkeypatch, receipt_calls, receipt_id=["SR1", "SR2"])

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    try:
        _wipe_order(cx, "subscription", "ch_prod_dup")
        _seed_product_sub(cx, next_charge_date="2000-01-01")
        c = appmod.app.test_client()
        r = c.post("/api/cron/charge-subscriptions", headers=_headers())
        assert r.status_code == 200, r.data
        assert len(receipt_calls) == 1

        from dashboard import qbo_sale as _qsale
        order = _orders_mod.find_order_by_external_ref(cx, "ch_prod_dup")
        assert order is not None
        _qsale.book_sale_on_payment(cx, dict(order))
        assert len(receipt_calls) == 1, "a duplicate booking attempt must not create a 2nd receipt"

        n = cx.execute(
            "SELECT COUNT(*) FROM orders WHERE external_ref=?", ("ch_prod_dup",)).fetchone()[0]
        assert n == 1
    finally:
        _cleanup_product(cx)
        _wipe_order(cx, "subscription", "ch_prod_dup")
        cx.close()
