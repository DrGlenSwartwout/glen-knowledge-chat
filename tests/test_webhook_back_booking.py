"""QBO webhook-back booking: /webhook/stripe books the paid-only Sales Receipt
directly on checkout.session.completed, so a closed browser tab (dropped
redirect) can't leave money collected with no QBO receipt + an order stuck
unpaid. Mirrors the /practitioner/checkout-return paid-only branch
(app.py:~25845-25868), but fired from the webhook instead of the return page.

Guarded on qbo_lines_json (paid-only checkout orders only -- trials and
memberships have no qbo_lines_json and are untouched) and idempotent via
book_sale_on_payment's atomic claim (qbo_sales_receipt_id), so it can never
double-book against the redirect handler. Best-effort: any failure inside the
block must be swallowed and the webhook must still 200.
"""

import json
import sqlite3

import pytest

import app
from dashboard import orders as O
from dashboard import qbo_billing
from dashboard import stripe_pay


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


def _seed_paid_only_order(db, token, *, total_cents=70000, with_lines=True,
                           already_booked=False):
    cx = sqlite3.connect(db)
    try:
        oid = O.upsert_order(cx, source="wholesale", external_ref=token,
                              email="dr@x.com", name="Dr X", total_cents=total_cents,
                              channel="wholesale", practitioner_id="pid1")
        if with_lines:
            O.set_order_qbo_lines(cx, token, {
                "lines": [{"name": "X Formula", "amount": 25.0, "qty": 20, "item_id": "55"}],
                "discount_cents": 0, "tax_cents": 0,
            })
        if already_booked:
            O.claim_sales_receipt_slot(cx, oid)
            O.set_order_sales_receipt_id(cx, oid, "SR-EXISTING")
        cx.commit()
    finally:
        cx.close()
    return oid


def _noop_fulfillers(monkeypatch):
    # Isolate the new webhook-back-booking block from the pre-existing
    # fulfillers (each independently re-fetches the session and would
    # otherwise also run against our mocked get_session).
    for name in ("_fulfill_biofield_trial", "_fulfill_prepay_term",
                 "_fulfill_biofield_program", "_fulfill_continuous_care_monthly",
                 "_fulfill_membership_product", "_fulfill_masterclass",
                 "_fulfill_coach_sub", "_fulfill_family_plan"):
        monkeypatch.setattr(app, name, lambda sid: None)


def _event(session_id="cs_evt"):
    import json
    return json.dumps({"type": "checkout.session.completed",
                        "data": {"object": {"id": session_id}}}).encode()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    app.app.config["TESTING"] = True
    return app.app.test_client()


def _mock_sales_receipt_spy(monkeypatch):
    calls = {"n": 0}

    def _fake_create_sales_receipt(customer, lines, *, discount_cents=0, tax_cents=0,
                                    email_to=None, bank_account_id=None, private_note=None):
        calls["n"] += 1
        return {"Id": f"SR{calls['n']}"}

    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda email, name="": {"Id": "C9"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _fake_create_sales_receipt)
    return calls


def test_webhook_books_paid_only_order_when_redirect_missed(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "a" * 32
    _seed_paid_only_order(db, token)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token}, "payment_intent": "pi_1"})
    calls = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["pay_status"] == "paid"
    assert row["stripe_payment_intent"] == "pi_1"
    assert row["qbo_sales_receipt_id"] == "SR1"
    assert calls["n"] == 1


def test_webhook_noop_when_already_booked(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "b" * 32
    _seed_paid_only_order(db, token, already_booked=True)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token}, "payment_intent": "pi_2"})
    calls = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    assert calls["n"] == 0

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["qbo_sales_receipt_id"] == "SR-EXISTING"
    # book_sale_on_payment's guard (qbo_sales_receipt_id already set) fires before
    # set_order_payment runs, so an already-booked order's pay_status is untouched.
    assert row["pay_status"] == "unpaid"


def test_webhook_noop_for_non_paidonly_session(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "c" * 32
    # No matching order at all for this token (e.g. a trial/membership session
    # with no qbo_lines_json order ever created).
    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 100,
        "metadata": {"invoice_id": token}, "payment_intent": "pi_3"})
    calls = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    assert calls["n"] == 0

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row is None


def test_webhook_swallows_booking_error_returns_200(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "d" * 32
    _seed_paid_only_order(db, token)

    def _boom(sid):
        raise RuntimeError("stripe API blew up")
    monkeypatch.setattr(stripe_pay, "get_session", _boom)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200


# ── Closed-tab per-kind settlement parity (Task 4) ───────────────────────────
# These exercise order_settlement.settle_paid_order_effects being dispatched
# from the webhook's book-back block -- the same per-kind side-effects the
# /begin/checkout-return redirect settles, so a closed browser tab settles
# identically (points/referral/subscription row/wallet margin/biofield
# readiness). Patterns mirror tests/test_begin_return_settlement.py.

def _spy_common_settlers(monkeypatch, calls):
    monkeypatch.setattr(app, "_settle_order_points",
                        lambda order, *, order_ref: calls["points"].append(order_ref))
    monkeypatch.setattr(app, "_settle_referral",
                        lambda order, *, order_ref: calls["referral"].append(order_ref))


def test_webhook_closed_tab_biofield_seeds_readiness(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "e" * 32
    _seed_paid_only_order(db, token)
    calls = {"points": [], "referral": [], "seed": []}
    _spy_common_settlers(monkeypatch, calls)

    import dashboard.biofield_store as _bf
    monkeypatch.setattr(_bf, "seed_paid",
                        lambda cx, email, *, via, order_ref: calls["seed"].append(order_ref))

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token, "kind": "biofield", "email": "bf@x.com"},
        "payment_intent": "pi_bf"})
    _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    assert calls["seed"] == [token]
    assert calls["points"] == [token]
    assert calls["referral"] == [token]


def test_webhook_closed_tab_subscribe_creates_subscription_row(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "f" * 32
    _seed_paid_only_order(db, token)
    calls = {"points": [], "referral": [], "create_once": 0}
    _spy_common_settlers(monkeypatch, calls)

    import dashboard.subscriptions as _subs
    orig_create_once = _subs.create_once

    def _spy_create_once(cx, **kw):
        calls["create_once"] += 1
        return orig_create_once(cx, **kw)
    monkeypatch.setattr(_subs, "create_once", _spy_create_once)

    monkeypatch.setattr(stripe_pay, "get_payment_intent", lambda pi: {
        "customer": "cus_1", "payment_method": "pm_1"})
    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {
            "invoice_id": token, "kind": "subscribe", "email": "sub@x.com",
            "cadence_months": "1",
            "items": json.dumps([{"slug": "brain-boost", "qty": 1}]),
            "ship": json.dumps({"state": "CA"}),
        },
        "payment_intent": "pi_sub"})
    _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    assert calls["create_once"] == 1

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    rows = cx.execute("SELECT * FROM subscriptions WHERE order_ref=?", (token,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["email"] == "sub@x.com"


def test_webhook_closed_tab_client_credits_wallet(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "g" * 32
    _seed_paid_only_order(db, token)
    calls = {"points": [], "referral": [], "wallet": []}
    _spy_common_settlers(monkeypatch, calls)

    import dashboard.wallet as _wal
    monkeypatch.setattr(_wal, "earn_dropship_margin",
                        lambda pid, marg, **k: calls["wallet"].append((pid, marg)) or 1)
    monkeypatch.setattr(app._pp, "record_dispensary_order",
                        lambda *a, **k: None, raising=False)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {
            "invoice_id": token, "kind": "client",
            "practitioner_id": "prac_1", "margin_cents": "1500",
            "patient_email": "pat@x.com", "subtotal_cents": "7000",
        },
        "payment_intent": "pi_client"})
    _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    assert calls["wallet"] == [("prac_1", 1500)]
    assert calls["points"] == [token]
    assert calls["referral"] == [token]


def test_webhook_closed_tab_retail_settles_points_and_referral(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "h" * 32
    _seed_paid_only_order(db, token)
    calls = {"points": [], "referral": []}
    _spy_common_settlers(monkeypatch, calls)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token, "kind": "retail"},
        "payment_intent": "pi_retail"})
    _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    assert calls["points"] == [token]
    assert calls["referral"] == [token]


def test_webhook_settlement_idempotent_when_run_twice(monkeypatch, tmp_path, client):
    """A closed-tab order that gets the SAME webhook delivered twice (Stripe
    retry / duplicate event) must not double-create the subscription: the
    first call books + settles. On the second call, the booking guard
    (qbo_sales_receipt_id already set) only skips re-booking the Sales Receipt --
    settlement is a separate, independently-gated step. It is the settled_at
    gate (set via mark_order_settled after the first call's settlement) that
    blocks settle_paid_order_effects from re-running, so no duplicate row is
    created."""
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "i" * 32
    _seed_paid_only_order(db, token)
    calls = {"points": [], "referral": []}
    _spy_common_settlers(monkeypatch, calls)

    monkeypatch.setattr(stripe_pay, "get_payment_intent", lambda pi: {
        "customer": "cus_1", "payment_method": "pm_1"})
    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {
            "invoice_id": token, "kind": "subscribe", "email": "sub2@x.com",
            "cadence_months": "1",
            "items": json.dumps([{"slug": "brain-boost", "qty": 1}]),
            "ship": json.dumps({"state": "CA"}),
        },
        "payment_intent": "pi_sub2"})
    _mock_sales_receipt_spy(monkeypatch)

    r1 = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r1.status_code == 200
    r2 = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r2.status_code == 200

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    rows = cx.execute("SELECT * FROM subscriptions WHERE order_ref=?", (token,)).fetchall()
    assert len(rows) == 1
    # points/referral only fire once too -- the second call's settled_at gate
    # (not the booking guard) prevents settlement, and therefore re-dispatch.
    assert calls["points"] == [token]
    assert calls["referral"] == [token]


def test_webhook_settlement_raising_still_200s_and_books_receipt(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "j" * 32
    _seed_paid_only_order(db, token)

    from dashboard import order_settlement as _osx

    def _boom(**kwargs):
        raise RuntimeError("settlement blew up")
    monkeypatch.setattr(_osx, "settle_paid_order_effects", _boom)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token, "kind": "retail"},
        "payment_intent": "pi_boom"})
    calls = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["qbo_sales_receipt_id"] == "SR1"
    assert row["pay_status"] == "paid"
    assert calls["n"] == 1


# ── I1 crash-strand backfill: settlement decoupled from the booking guard,
# gated independently on settled_at (Task 3) ────────────────────────────────

def test_webhook_settles_when_booked_but_unsettled(monkeypatch, tmp_path, client):
    """Redirect crashed (or Stripe redelivered) after booking the receipt but
    before settling: qbo_sales_receipt_id is already set and settled_at is
    NULL. The webhook must still run settlement (the new independent branch)
    and mark settled_at -- this is the I1 crash-strand backfill."""
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "k" * 32
    _seed_paid_only_order(db, token, already_booked=True)
    calls = {"points": [], "referral": []}
    _spy_common_settlers(monkeypatch, calls)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token, "kind": "retail"},
        "payment_intent": "pi_backfill"})
    calls_sr = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    # Booking branch is skipped (already booked) -- no new receipt created.
    assert calls_sr["n"] == 0
    # Settlement branch runs independently.
    assert calls["points"] == [token]
    assert calls["referral"] == [token]

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["qbo_sales_receipt_id"] == "SR-EXISTING"
    assert row["settled_at"] is not None


def test_webhook_skips_settlement_when_already_settled(monkeypatch, tmp_path, client):
    """Order already booked AND already settled (settled_at set): a Stripe
    redelivery must not re-run settlement."""
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "l" * 32
    oid = _seed_paid_only_order(db, token, already_booked=True)
    cx0 = sqlite3.connect(db)
    try:
        assert O.mark_order_settled(cx0, oid) is True
    finally:
        cx0.close()
    calls = {"points": [], "referral": []}
    _spy_common_settlers(monkeypatch, calls)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token, "kind": "retail"},
        "payment_intent": "pi_already_settled"})
    calls_sr = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    assert calls_sr["n"] == 0
    assert calls["points"] == []
    assert calls["referral"] == []


def test_webhook_settle_raise_leaves_settled_at_null_for_retry(monkeypatch, tmp_path, client):
    """A total settle_paid_order_effects raise must leave settled_at NULL (not
    mark it) so a Stripe redelivery can re-attempt settlement -- the receipt
    booking itself must still go through."""
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "m" * 32
    _seed_paid_only_order(db, token)

    from dashboard import order_settlement as _osx

    def _boom(**kwargs):
        raise RuntimeError("settlement blew up")
    monkeypatch.setattr(_osx, "settle_paid_order_effects", _boom)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token, "kind": "retail"},
        "payment_intent": "pi_boom2"})
    calls = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["qbo_sales_receipt_id"] == "SR1"
    assert row["pay_status"] == "paid"
    assert calls["n"] == 1
    assert row["settled_at"] is None
