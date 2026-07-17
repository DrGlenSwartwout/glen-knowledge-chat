"""Personal ordering path for cert participants (Task 2).

A near-clone of the wholesale checkout with two crucial differences:
  (a) NO wholesale_unlocked gate, and
  (b) NEVER resale-exempt (resale_ok=False) — personal purchases are taxed.

On a fee-free (zelle/wise) order the participant earns 3.5% Wellness Credit
(``wallet.personal_earn_cents``). To avoid double-crediting, ``build_order`` is
called with ``method=None`` (which suppresses its internal 3% fee-free earn) and
the route credits the full 3.5% itself via the explicit-amount, invoice-
idempotent wallet credit primitive.

The heavy deps (portal data, the checkout engine, order ingestion, the wallet)
are stubbed by monkeypatching the imported names in ``app``.
"""

import json
import sqlite3

import pytest

import app as _appmod_top
from dashboard import orders as O
from dashboard import qbo_billing
from dashboard import stripe_pay


# ── fixtures / stubs ───────────────────────────────────────────────────────────

PORTAL = {
    "cart": [{"slug": "x", "qty": 2}],
    "modules_completed": 3,
    "email": "c@x.com",
    "name": "C",
    "wholesale_unlocked": False,       # crux: personal path must NOT require this
    "resale_license_number": None,
    "wallet_balance_cents": 0,
    "quote": {"total_bottles": 2, "subtotal_cents": 12000},
}


class _Recorder:
    """Records the kwargs/args of every call so tests can assert on them."""

    def __init__(self, ret=None):
        self.calls = []          # list of (args, kwargs)
        self._ret = ret

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._ret

    @property
    def called(self):
        return bool(self.calls)

    def last_kwargs(self):
        return self.calls[-1][1]

    def last_args(self):
        return self.calls[-1][0]


@pytest.fixture
def client(monkeypatch):
    import app as appmod
    appmod.app.config["TESTING"] = True

    # signed in by default (override per-test for the 401 case)
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "pid1")
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid: dict(PORTAL))

    build_order = _Recorder(ret={
        "ok": True, "invoice_id": "INV1", "doc_number": "1001",
        "total": 120.0, "get_cents": 500, "credit_redeemed_cents": 0,
    })
    monkeypatch.setattr(appmod._wc, "build_order", build_order)

    ingest = _Recorder(ret=None)
    monkeypatch.setattr(appmod, "_ingest_order", ingest)

    # explicit-amount, invoice-idempotent earned-credit primitive
    earn = _Recorder(ret=420)
    monkeypatch.setattr(appmod._wallet, "earn_personal", earn)

    monkeypatch.setattr(appmod._pp, "cart_clear", lambda pid: None)
    monkeypatch.setattr(appmod._pp, "record_order", lambda *a, **k: None)

    return appmod.app.test_client(), appmod, build_order, ingest, earn


# ── 0. the portal page still serves ─────────────────────────────────────────────

def test_portal_page_serves():
    import app as appmod
    appmod.app.config["TESTING"] = True
    r = appmod.app.test_client().get("/practitioner/portal")
    assert r.status_code == 200
    assert b"Practitioner Portal" in r.data


# ── 1. not signed in → 401 on quote + checkout ─────────────────────────────────

def test_quote_not_signed_in(client):
    c, appmod, *_ = client
    import app as _a
    _a._practitioner_session_pid = lambda: None
    r = c.post("/api/practitioner/personal/quote", json={})
    assert r.status_code == 401


def test_checkout_not_signed_in(client):
    c, appmod, *_ = client
    import app as _a
    _a._practitioner_session_pid = lambda: None
    r = c.post("/api/practitioner/personal/checkout", json={})
    assert r.status_code == 401


# ── 2. checkout succeeds WITHOUT wholesale_unlocked (no 403) ───────────────────

def test_checkout_ok_without_wholesale_unlock(client):
    c, appmod, build_order, ingest, earn = client
    r = c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["invoice_id"] == "INV1"


# ── 3. build_order called with resale_ok=False ─────────────────────────────────

def test_build_order_resale_ok_false(client):
    c, appmod, build_order, ingest, earn = client
    c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert build_order.called
    assert build_order.last_kwargs().get("resale_ok") is False


# ── 4. _ingest_order called with channel="personal" ───────────────────────────

def test_ingest_channel_personal(client):
    c, appmod, build_order, ingest, earn = client
    c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert ingest.called
    assert ingest.last_kwargs().get("channel") == "personal"


# ── 4b. the ingest carries the cart LINE ITEMS (so orders.items_json is real —
#        the dispense-ranking feature depends on this; guards a silent regression) ──

def test_ingest_personal_passes_cart_items(client):
    c, appmod, build_order, ingest, earn = client
    c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert ingest.last_kwargs().get("items") == PORTAL["cart"]


def test_ingest_wholesale_passes_cart_items(client, monkeypatch):
    c, appmod, build_order, ingest, earn = client
    wholesale = dict(PORTAL); wholesale["wholesale_unlocked"] = True
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid: dict(wholesale))
    r = c.post("/api/practitioner/checkout", json={"method": "zelle"})
    assert r.status_code == 200
    assert ingest.last_kwargs().get("channel") == "wholesale"
    assert ingest.last_kwargs().get("items") == PORTAL["cart"]


# ── 5. empty cart → 400 ────────────────────────────────────────────────────────

def test_empty_cart_400(client, monkeypatch):
    c, appmod, *_ = client
    empty = dict(PORTAL)
    empty["cart"] = []
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid: empty)
    r = c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert r.status_code == 400


# ── 6. fee-free earn: zelle credits 3.5% (420), card credits 0 ─────────────────

def test_zelle_credits_personal_earn(client):
    c, appmod, build_order, ingest, earn = client
    c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    # build_order must be called with method=None so it does NOT also earn 3%
    assert build_order.last_kwargs().get("method") is None
    assert earn.called
    # charged_cents = round(total*100) = 12000 ; 3.5% -> 420
    args, kwargs = earn.calls[-1]
    flat = list(args) + list(kwargs.values())
    assert 420 in flat


def test_wise_credits_personal_earn(client):
    c, appmod, build_order, ingest, earn = client
    c.post("/api/practitioner/personal/checkout", json={"method": "wise"})
    assert earn.called
    args, kwargs = earn.calls[-1]
    flat = list(args) + list(kwargs.values())
    assert 420 in flat


def test_card_credits_zero(client, monkeypatch):
    c, appmod, build_order, ingest, earn = client
    # force card to be honored (Stripe active) so we exercise the card branch
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_order",
                        lambda *a, **k: "https://stripe/x", raising=False)
    c.post("/api/practitioner/personal/checkout", json={"method": "card"})
    # card => personal_earn_cents == 0 => credit fn not called (or called with 0)
    if earn.called:
        args, kwargs = earn.calls[-1]
        flat = list(args) + list(kwargs.values())
        assert all(v != 420 for v in flat)
        assert 0 in flat or not any(isinstance(v, int) and v > 0 for v in flat)
    else:
        assert not earn.called


# ── quote happy path ───────────────────────────────────────────────────────────

def test_quote_ok(client):
    c, appmod, *_ = client
    r = c.post("/api/practitioner/personal/quote", json={})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["quote"] == PORTAL["quote"]
    assert body["wallet_balance_cents"] == 0


# ── 7. CRITICAL: qbo_payload persistence (booking-gap fix) ────────────────────
#
# build_order was converted to paid-only: no more create_invoice, the CALLER
# must persist out["qbo_payload"] via set_order_qbo_lines so the return-handler
# (card) or record_payment (alt-pay) can later book a real Sales Receipt. The
# wholesale route does this (app.py ~14049-14056); the personal route did not --
# so a personal order booked NEITHER an invoice nor a Sales Receipt. This test
# drives the personal checkout with a realistic build_order stub (qbo_payload +
# token invoice_id) against an isolated on-disk orders table and asserts the
# order row actually gets qbo_lines_json persisted, keyed on the token.

def _isolate_db(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(_appmod_top, "LOG_DB", db)
    cx = sqlite3.connect(db)
    try:
        O.init_orders_table(cx)
        cx.commit()
    finally:
        cx.close()
    return db


def _personal_fixed_out(token):
    return {
        "ok": True, "invoice_id": token, "customer_id": "", "doc_number": "",
        "total": 500.0, "subtotal_cents": 50000, "credit_redeemed_cents": 0,
        "fee_free_credit_cents": 0, "get_cents": 275, "method": None,
        "qbo_payload": {"lines": [{"name": "X Formula", "amount": 25.0, "qty": 20,
                                    "item_id": "55"}],
                       "discount_cents": 0, "tax_cents": 0},
    }


def test_route_persists_qbo_lines_for_personal_order(monkeypatch, tmp_path):
    _appmod_top.app.config["TESTING"] = True
    db = _isolate_db(monkeypatch, tmp_path)
    token = "c" * 32
    fixed_out = _personal_fixed_out(token)

    def boom(*a, **k):
        raise AssertionError("build_order must not touch QBO invoicing (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "apply_invoice_discount", boom)

    monkeypatch.setattr(_appmod_top, "_practitioner_session_pid", lambda: "pid1")
    monkeypatch.setattr(_appmod_top._pp, "portal_data", lambda pid: dict(PORTAL))
    monkeypatch.setattr(_appmod_top._pp, "cart_clear", lambda pid: None)
    monkeypatch.setattr(_appmod_top._pp, "record_order", lambda *a, **k: None)
    monkeypatch.setattr(_appmod_top._wc, "build_order", lambda *a, **k: dict(fixed_out))
    monkeypatch.setattr(_appmod_top._wallet, "earn_personal", lambda *a, **k: 0)

    r = _appmod_top.app.test_client().post(
        "/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body["invoice_id"] == token

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row is not None, "personal order was never ingested"
    assert row["source"] == "personal"
    assert row["qbo_lines_json"] is not None, (
        "qbo_payload was never persisted -- personal order books NEITHER an "
        "invoice nor a Sales Receipt (the booking gap this fix closes)")
    payload = json.loads(row["qbo_lines_json"])
    assert payload["lines"][0]["item_id"] == "55"
    assert payload["tax_cents"] == 0


def test_personal_card_return_books_one_sales_receipt(monkeypatch, tmp_path):
    """End-to-end: personal checkout persists qbo_lines, then the paid-only
    card-return branch (source-agnostic, guarded on qbo_lines_json) books
    exactly one Sales Receipt for it."""
    _appmod_top.app.config["TESTING"] = True
    db = _isolate_db(monkeypatch, tmp_path)
    token = "d" * 32
    fixed_out = _personal_fixed_out(token)

    monkeypatch.setattr(qbo_billing, "create_invoice",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("no invoice in paid-only flow")))
    monkeypatch.setattr(_appmod_top, "_practitioner_session_pid", lambda: "pid1")
    monkeypatch.setattr(_appmod_top._pp, "portal_data", lambda pid: dict(PORTAL))
    monkeypatch.setattr(_appmod_top._pp, "cart_clear", lambda pid: None)
    monkeypatch.setattr(_appmod_top._pp, "record_order", lambda *a, **k: None)
    monkeypatch.setattr(_appmod_top._wc, "build_order", lambda *a, **k: dict(fixed_out))
    monkeypatch.setattr(_appmod_top._wallet, "earn_personal", lambda *a, **k: 0)

    c = _appmod_top.app.test_client()
    r = c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert r.status_code == 200, r.get_data(as_text=True)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid", "amount_total": 50000,
        "metadata": {"invoice_id": token, "customer_id": ""},
        "payment_intent": "pi_personal_1",
    })
    calls = {"n": 0}

    def _fake_create_sales_receipt(customer, lines, *, discount_cents=0, tax_cents=0,
                                    email_to=None):
        calls["n"] += 1
        return {"Id": "SR-PERSONAL-1"}
    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda email, name="": {"Id": "C9"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _fake_create_sales_receipt)

    r2 = c.get(f"/practitioner/checkout-return?session_id=sess1&t={token}")
    assert r2.status_code in (301, 302)
    assert calls["n"] == 1

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["pay_status"] == "paid"
    assert row["qbo_sales_receipt_id"] == "SR-PERSONAL-1"
