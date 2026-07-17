"""QBO paid-only Stage 4, Task 2: dashboard.wholesale_checkout.build_module_order --
certification-module tuition purchase.

Converts from "create ONE invoice, redeem credit, apply discount" to paid-only:
a fresh ``checkout_ref`` token is minted FIRST (no QBO invoice/customer at
checkout time), credit is redeemed via ``wallet.redeem_for_module`` (which keeps
its OWN idempotency -- one module-redemption per calendar month per
practitioner, gated by module_slug + earn_period, not by a ref argument -- its
signature takes no ref/invoice-id param), and a line-faithful ``qbo_payload`` is
returned for a future caller to persist; a real Sales Receipt is booked once
payment is confirmed, mirroring ``build_order`` (Task 1) and the already
paid-only-aware ``quote_module`` just below it.

NOTE: as of this task, ``build_module_order`` has NO caller in app.py (the
prior caller in ``api_practitioner_register`` was already replaced by
``quote_module`` in an earlier paid-only pass -- see task report). These are
therefore unit tests on the function only; there is no route-level persistence
test to write here.
"""

import sqlite3
from datetime import datetime

import pytest

from dashboard import qbo_billing

PID = "00000000-0000-0000-0000-000000000003"


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
        elif "entry_type = 'spend_module' AND earn_period = %s" in s:
            self._result = [{"x": 1}] if any(
                r["practitioner_id"] == p[0] and r["entry_type"] == "spend_module"
                and r["earn_period"] == p[1] for r in self.store["ledger"]) else []
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
    import dashboard.wholesale_checkout as wc
    import dashboard.wallet as wallet
    store = {"balances": {}, "ledger": []}
    monkeypatch.setattr(wallet, "_cursor", lambda: _FakeCtx(store))

    def boom(*a, **k):
        raise AssertionError("build_module_order must not touch QBO invoicing (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "apply_invoice_discount", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", boom)

    prac = {"id": PID, "email": "dr@x.com", "name": "Dr X"}
    return {"wc": wc, "wallet": wallet, "store": store, "prac": prac}


# ── Guard: no invoice/customer/discount; token-shaped invoice_id ───────────────

def test_build_module_order_creates_no_qbo_invoice_or_customer(env):
    out = env["wc"].build_module_order(env["prac"], "module-1", today=datetime(2026, 6, 9))
    assert out["ok"] is True
    assert out["customer_id"] == ""
    assert out["doc_number"] == ""
    ref = out["invoice_id"]
    assert isinstance(ref, str) and len(ref) == 32
    int(ref, 16)  # valid hex


def test_build_module_order_qbo_payload_is_line_faithful(env):
    out = env["wc"].build_module_order(env["prac"], "module-1", today=datetime(2026, 6, 9))
    payload = out["qbo_payload"]
    assert payload["tax_cents"] == 0
    assert payload["discount_cents"] == out["credit_redeemed_cents"]
    line = payload["lines"][0]
    assert line["qty"] == 1
    assert line["amount"] == round(out["tuition_cents"] / 100.0, 2)
    assert "module-1" in line["name"]


# ── Redeem fidelity: resolved BEFORE booking; no QBO discount call ─────────────

def test_build_module_order_redeems_up_to_full_tuition(env):
    env["wallet"].earn_dropship(PID, 50)               # +$1000 credit
    out = env["wc"].build_module_order(env["prac"], "module-1", today=datetime(2026, 6, 9))
    assert out["ok"] is True
    assert out["credit_redeemed_cents"] == 29700        # full $297 covered
    assert out["qbo_payload"]["discount_cents"] == 29700
    assert env["wallet"].get_balance_cents(PID) == 70300


def test_build_module_order_charge_equals_tuition_minus_redeemed(env):
    env["wallet"].earn_dropship(PID, 5)                 # +$100 credit (below full tuition)
    out = env["wc"].build_module_order(env["prac"], "module-1", today=datetime(2026, 6, 9))
    assert out["credit_redeemed_cents"] == 10000
    assert out["total"] == round((out["tuition_cents"] - 10000) / 100.0, 2)


def test_build_module_order_no_credit_charges_full_tuition(env):
    out = env["wc"].build_module_order(env["prac"], "module-1", today=datetime(2026, 6, 9))
    assert out["credit_redeemed_cents"] == 0
    assert out["total"] == round(out["tuition_cents"] / 100.0, 2)


def test_build_module_order_redemption_gated_monthly_not_by_token(env):
    """redeem_for_module keeps its OWN idempotency (module_slug + calendar month),
    not the checkout_ref token -- a second build_module_order call in the same
    month must not redeem twice, even though each call mints a fresh token."""
    env["wallet"].earn_dropship(PID, 50)  # +$1000 credit
    first = env["wc"].build_module_order(env["prac"], "module-1", today=datetime(2026, 6, 9))
    second = env["wc"].build_module_order(env["prac"], "module-1", today=datetime(2026, 6, 20))
    assert first["credit_redeemed_cents"] == 29700
    assert second["credit_redeemed_cents"] == 0
    assert first["invoice_id"] != second["invoice_id"]  # distinct tokens
    assert env["wallet"].get_balance_cents(PID) == 70300
