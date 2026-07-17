"""QBO paid-only Stage 4, Task 3: dashboard.dropship_checkout.build_dropship_order --
practitioner-paid drop-ship checkout (ships to the patient).

Converts from "create ONE invoice, redeem credit, apply discount" to paid-only:
a fresh ``checkout_ref`` token is minted, credit is redeemed against that token
(never a QBO invoice id, since none exists at checkout time), and a
line-faithful ``qbo_payload`` is returned for the route to persist -- the
return-handler (already wired for any order carrying qbo_lines_json, Task 1)
books a real Sales Receipt from it once payment is confirmed.

Two layers:
  - unit tests on ``dropship_checkout.build_dropship_order`` directly (mirrors
    tests/test_dropship_checkout.py's fixture shape) for the guard + redeem
    fidelity + charge math;
  - one route-level test (build_dropship_order mocked) confirming the
    persistence wiring in ``api_practitioner_dropship_checkout`` -- qbo_lines_json,
    get_cents, and the order keyed on the token with source "dropship".
"""

import json
import sqlite3

import pytest

import app
from dashboard import orders as O
from dashboard import qbo_billing

PID = "00000000-0000-0000-0000-000000000003"


# ── wallet fake cursor (mirrors tests/test_wholesale_paid_only.py) ───────────

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


@pytest.fixture
def env(monkeypatch):
    import dashboard.dropship_checkout as dc
    import dashboard.wallet as wallet
    import dashboard.tax as tax
    store = {"balances": {}, "ledger": []}
    monkeypatch.setattr(wallet, "_cursor", lambda: _FakeCtx(store))

    def boom(*a, **k):
        raise AssertionError("build_dropship_order must not touch QBO invoicing (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "apply_invoice_discount", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", boom)

    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    monkeypatch.setattr(tax, "compute_get_cents",
                        lambda subtotal, *, channel, ship_to_state, resale_ok=False: 275)

    prac = {"id": PID, "modules_completed": 0, "email": "dr@x.com", "name": "Dr X"}
    ship = {"name": "Pat", "state": "CA", "country": "US", "address1": "1 St"}
    return {"dc": dc, "wallet": wallet, "store": store, "prac": prac, "ship": ship}


def _expected_subtotal(env, qty=40):
    dl = env["dc"].dropship_line_cents(retail_cents=7000, qty=qty,
                                       modules=env["prac"]["modules_completed"],
                                       settings=env["dc"]._settings())
    return dl["unit_cents"] * qty


# ── Guard: no invoice/customer; token-shaped invoice_id ────────────────────────

def test_build_dropship_order_creates_no_qbo_invoice_or_customer(env):
    out = env["dc"].build_dropship_order([{"slug": "x", "qty": 40}], env["prac"],
                                         patient_ship=env["ship"], method="zelle")
    assert out["ok"] is True
    assert out["customer_id"] == ""
    assert out["doc_number"] == ""
    ref = out["invoice_id"]
    assert isinstance(ref, str) and len(ref) == 32
    int(ref, 16)  # valid hex


def test_build_dropship_order_qbo_payload_is_line_faithful(env):
    out = env["dc"].build_dropship_order([{"slug": "x", "qty": 40}], env["prac"],
                                         patient_ship=env["ship"], method="zelle")
    payload = out["qbo_payload"]
    assert payload["lines"][0]["qty"] == 40
    assert payload["tax_cents"] == 0
    assert payload["discount_cents"] == out["credit_redeemed_cents"]
    assert out["ship_to"]["name"] == "Pat"
    assert out["source"] == "dropship"
    assert out["get_cents"] == 275


def test_build_dropship_order_empty_cart_rejected(env):
    assert env["dc"].build_dropship_order([], env["prac"],
                                          patient_ship=env["ship"])["ok"] is False
    assert env["dc"].build_dropship_order([{"slug": "x", "qty": 0}], env["prac"],
                                          patient_ship=env["ship"])["ok"] is False


# ── Redeem fidelity: resolved BEFORE booking, keyed on the token ───────────────

def test_build_dropship_order_redeems_credit_keyed_on_token_not_invoice(env):
    subtotal = _expected_subtotal(env)
    env["wallet"].earn_dropship(PID, 200)  # plenty of credit, well above the 50% cap
    out = env["dc"].build_dropship_order([{"slug": "x", "qty": 40}], env["prac"],
                                         patient_ship=env["ship"], method="zelle")
    assert out["ok"] is True
    assert out["subtotal_cents"] == subtotal
    expected_redeemed = subtotal // 2
    assert out["credit_redeemed_cents"] == expected_redeemed
    assert out["qbo_payload"]["discount_cents"] == expected_redeemed
    assert out["total"] == round((subtotal - expected_redeemed) / 100.0, 2)

    ledger_refs = {r["qbo_invoice_id"] for r in env["store"]["ledger"]
                  if r["entry_type"] == "spend_order"}
    assert ledger_refs == {out["invoice_id"]}


def test_build_dropship_order_charge_equals_subtotal_minus_redeemed(env):
    subtotal = _expected_subtotal(env)
    env["wallet"].earn_dropship(PID, 10)  # a small credit, below the 50% cap
    out = env["dc"].build_dropship_order([{"slug": "x", "qty": 40}], env["prac"],
                                         patient_ship=env["ship"], method="zelle")
    assert out["credit_redeemed_cents"] < subtotal // 2
    assert out["total"] == round((subtotal - out["credit_redeemed_cents"]) / 100.0, 2)


def test_build_dropship_order_zelle_fee_free_earn_keyed_on_token(env):
    out = env["dc"].build_dropship_order([{"slug": "x", "qty": 40}], env["prac"],
                                         patient_ship=env["ship"], method="zelle")
    assert out["fee_free_credit_cents"] > 0
    ledger_refs = {r["qbo_invoice_id"] for r in env["store"]["ledger"]
                  if r["entry_type"] == "earn_fee_free"}
    assert ledger_refs == {out["invoice_id"]}


def test_build_dropship_order_no_fee_free_earn_on_card(env):
    out = env["dc"].build_dropship_order([{"slug": "x", "qty": 40}], env["prac"],
                                         patient_ship=env["ship"], method="card")
    assert out["fee_free_credit_cents"] == 0


def test_build_dropship_order_redemption_idempotent_per_token(env):
    """A second call keyed on the SAME token must not redeem twice (mirrors the
    wallet's per-ref idempotency, now keyed on checkout_ref)."""
    env["wallet"].earn_dropship(PID, 200)
    token = "deadbeefdeadbeefdeadbeefdeadbeef"
    first = env["wallet"].redeem_for_order(PID, 100000, token)
    second = env["wallet"].redeem_for_order(PID, 100000, token)
    assert first == 50000
    assert second == 0


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


def test_route_persists_qbo_lines_get_cents_and_source_dropship(monkeypatch, tmp_path):
    app.app.config["TESTING"] = True
    db = _isolate_db(monkeypatch, tmp_path)
    token = "b" * 32
    fixed_out = {
        "ok": True, "invoice_id": token, "customer_id": "", "doc_number": "",
        "total": 500.0, "subtotal_cents": 100000, "credit_redeemed_cents": 50000,
        "fee_free_credit_cents": 0, "get_cents": 275, "method": "zelle",
        "source": "dropship", "ship_to": {"name": "Pat", "state": "CA"},
        "qbo_payload": {"lines": [{"name": "x", "amount": 25.0, "qty": 40,
                                    "description": "x (drop-ship wholesale)"}],
                       "discount_cents": 50000, "tax_cents": 0},
    }

    def boom(*a, **k):
        raise AssertionError("create_invoice must not be called by the route either")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "apply_invoice_discount", boom)

    monkeypatch.setattr(app, "_practitioner_session_pid", lambda: "pid1")
    monkeypatch.setattr(app._pp, "portal_data", lambda pid: {
        "cart": [{"slug": "x", "qty": 40}], "modules_completed": 0,
        "email": "dr@x.com", "name": "Dr X", "wholesale_unlocked": True,
    })
    monkeypatch.setattr(app._pp, "cart_clear", lambda pid: None)
    monkeypatch.setattr(app._dropship, "build_dropship_order", lambda *a, **k: dict(fixed_out))

    r = app.app.test_client().post(
        "/api/practitioner/dropship/checkout",
        json={"method": "zelle",
              "patient_address": {"name": "Pat", "state": "CA", "country": "US",
                                  "street": "1 Main St", "city": "Los Angeles", "zip": "90001"}})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body["invoice_id"] == token
    assert body["customer_id"] == ""

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row is not None
    assert row["source"] == "dropship"
    assert int(row["get_cents"]) == 275

    payload = json.loads(row["qbo_lines_json"])
    assert payload["discount_cents"] == 50000
    assert payload["tax_cents"] == 0
    assert payload["lines"][0]["qty"] == 40
