"""QBO paid-only Stage 4, Task 1: dashboard.wholesale_checkout.build_order --
practitioner product-cart checkout.

Converts from "create ONE invoice, redeem credit, apply discount" to paid-only:
a fresh ``checkout_ref`` token is minted FIRST, credit is redeemed against that
token (never a QBO invoice id, since none exists at checkout time), and a
line-faithful ``qbo_payload`` is returned for the route to persist -- the
return-handler books a real Sales Receipt from it once payment is confirmed.

Two layers:
  - unit tests on ``wholesale_checkout.build_order`` directly (mirrors
    tests/test_wholesale_checkout.py's fixture shape) for the guard + redeem
    fidelity + charge math;
  - one route-level test (build_order mocked) confirming the persistence wiring
    in ``api_practitioner_checkout`` -- qbo_lines_json, get_cents, and the order
    keyed on the token with source "wholesale".
"""

import json
import sqlite3

import pytest

import app
from dashboard import orders as O
from dashboard import qbo_billing

PID = "00000000-0000-0000-0000-000000000002"


# ── wallet fake cursor (mirrors tests/test_wholesale_checkout.py) ─────────────

class _FakeCursor:
    def __init__(self, store):
        self.store = store
        self._result = []

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        p = list(params)
        if s.startswith("SELECT wallet_balance_cents FROM practitioners"):
            self._result = [{"wallet_balance_cents": self.store["balances"].get(p[0], 0)}]
        elif s.startswith("UPDATE practitioners SET wallet_balance_cents"):
            self.store["balances"][p[1]] = p[0]
            self._result = []
        elif s.startswith("INSERT INTO wallet_ledger"):
            cols = ["practitioner_id", "entry_type", "amount_cents", "balance_after_cents",
                    "qbo_invoice_id", "module_slug", "earn_period", "note"]
            self.store["ledger"].append(dict(zip(cols, p)))
            self._result = []
        elif "qbo_invoice_id = %s AND entry_type = %s" in s:
            self._result = [{"x": 1}] if any(
                r["qbo_invoice_id"] == p[0] and r["entry_type"] == p[1]
                for r in self.store["ledger"]) else []
        else:
            self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


class _FakeCtx:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return _FakeCursor(self.store)

    def __exit__(self, *a):
        return False


def _seed_box(tmp_path, name, L):
    from dashboard.shipping import init_shipping_schema, add_bottle_type, set_box_capacity
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    bid = add_bottle_type(name, db_path=db)
    set_box_capacity(bid, "L", L, db_path=db)
    return db


@pytest.fixture
def env(tmp_path, monkeypatch):
    import dashboard.wholesale_checkout as wc
    import dashboard.wallet as wallet
    store = {"balances": {}, "ledger": []}
    monkeypatch.setattr(wallet, "_cursor", lambda: _FakeCtx(store))

    def boom(*a, **k):
        raise AssertionError("build_order must not touch QBO invoicing (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "apply_invoice_discount", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", boom)

    db = _seed_box(tmp_path, "dropper", 20)
    catalog = {"x": {"name": "X Formula", "bottle_type": "dropper",
                     "price_cents": 7000, "qbo_item_id": "55"}}
    prac = {"id": PID, "modules_completed": 12, "email": "dr@x.com", "name": "Dr X"}
    return {"wc": wc, "wallet": wallet, "store": store, "db": db,
            "catalog": catalog, "prac": prac}


# ── Guard: no invoice/customer; token-shaped invoice_id ────────────────────────

def test_build_order_creates_no_qbo_invoice_or_customer(env):
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"],
                                db_path=env["db"], catalog=env["catalog"])
    assert out["ok"] is True
    assert out["customer_id"] == ""
    assert out["doc_number"] == ""
    ref = out["invoice_id"]
    assert isinstance(ref, str) and len(ref) == 32
    int(ref, 16)  # valid hex


def test_build_order_qbo_payload_is_line_faithful(env):
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"],
                                db_path=env["db"], catalog=env["catalog"])
    payload = out["qbo_payload"]
    assert payload["lines"][0]["item_id"] == "55"
    assert payload["tax_cents"] == 0
    assert payload["discount_cents"] == out["credit_redeemed_cents"]


def test_build_order_margin_breach_still_creates_nothing(env):
    cat = {"x": {"name": "X", "bottle_type": "dropper", "price_cents": 7000,
                 "qbo_item_id": "55", "cogs_cents": 2000, "fulfillment_cents": 800}}
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"],
                                db_path=env["db"], catalog=cat)
    assert out["ok"] is False
    assert out["error"] == "margin_floor"


# ── Redeem fidelity: resolved BEFORE booking, keyed on the token ───────────────

def test_build_order_redeems_credit_keyed_on_token_not_invoice(env):
    env["wallet"].earn_dropship(PID, 50)  # +$1000 credit
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"],
                                db_path=env["db"], catalog=env["catalog"])
    assert out["ok"] is True
    # certified floor, 40 bottles -> $1000 subtotal; 50% cap -> $500 redeemed
    assert out["subtotal_cents"] == 100000
    assert out["credit_redeemed_cents"] == 50000
    assert out["qbo_payload"]["discount_cents"] == 50000
    assert out["total"] == round((100000 - 50000) / 100.0, 2)
    assert env["wallet"].get_balance_cents(PID) == 50000

    # ledger is keyed on the checkout_ref token, never a QBO invoice id
    ledger_refs = {r["qbo_invoice_id"] for r in env["store"]["ledger"]
                  if r["entry_type"] == "spend_order"}
    assert ledger_refs == {out["invoice_id"]}


def test_build_order_charge_equals_subtotal_minus_redeemed(env):
    env["wallet"].earn_dropship(PID, 10)  # +$200 credit (below the 50% cap)
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"],
                                db_path=env["db"], catalog=env["catalog"])
    assert out["credit_redeemed_cents"] == 20000
    assert out["total"] == round((out["subtotal_cents"] - 20000) / 100.0, 2)


def test_build_order_zelle_fee_free_earn_keyed_on_token(env):
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"], method="zelle",
                                db_path=env["db"], catalog=env["catalog"])
    assert out["fee_free_credit_cents"] == 3000  # 3% of $1000 charged
    ledger_refs = {r["qbo_invoice_id"] for r in env["store"]["ledger"]
                  if r["entry_type"] == "earn_fee_free"}
    assert ledger_refs == {out["invoice_id"]}


def test_build_order_redemption_idempotent_per_token(env):
    """A second call keyed on the SAME token must not redeem twice (mirrors the
    wallet's per-ref idempotency, now keyed on checkout_ref)."""
    env["wallet"].earn_dropship(PID, 50)  # +$1000 credit
    token = "deadbeefdeadbeefdeadbeefdeadbeef"
    first = env["wallet"].redeem_for_order(PID, 100000, token)
    second = env["wallet"].redeem_for_order(PID, 100000, token)
    assert first == 50000
    assert second == 0
    assert env["wallet"].get_balance_cents(PID) == 50000


# ── Route wiring: persist qbo_lines + get_cents + order keyed on the token ────

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


def test_route_persists_qbo_lines_get_cents_and_source(monkeypatch, tmp_path):
    app.app.config["TESTING"] = True
    db = _isolate_db(monkeypatch, tmp_path)
    token = "a" * 32
    fixed_out = {
        "ok": True, "invoice_id": token, "customer_id": "", "doc_number": "",
        "total": 500.0, "subtotal_cents": 100000, "credit_redeemed_cents": 50000,
        "fee_free_credit_cents": 0, "get_cents": 275, "method": "zelle",
        "qbo_payload": {"lines": [{"name": "X Formula", "amount": 25.0, "qty": 40,
                                    "item_id": "55"}],
                       "discount_cents": 50000, "tax_cents": 0},
    }

    def boom(*a, **k):
        raise AssertionError("create_invoice must not be called by the route either")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "apply_invoice_discount", boom)

    monkeypatch.setattr(app, "_practitioner_session_pid", lambda: "pid1")
    monkeypatch.setattr(app._pp, "portal_data", lambda pid: {
        "cart": [{"slug": "x", "qty": 40}], "modules_completed": 12,
        "email": "dr@x.com", "name": "Dr X", "wholesale_unlocked": True,
        "resale_license_number": None,
    })
    monkeypatch.setattr(app._pp, "cart_clear", lambda pid: None)
    monkeypatch.setattr(app._pp, "record_order", lambda *a, **k: None)
    monkeypatch.setattr(app._wc, "build_order", lambda *a, **k: dict(fixed_out))

    r = app.app.test_client().post("/api/practitioner/checkout", json={"method": "zelle"})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body["invoice_id"] == token
    assert body["customer_id"] == ""

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row is not None
    assert row["source"] == "wholesale"
    assert int(row["get_cents"]) == 275

    payload = json.loads(row["qbo_lines_json"])
    assert payload["discount_cents"] == 50000
    assert payload["tax_cents"] == 0
    assert payload["lines"][0]["item_id"] == "55"
