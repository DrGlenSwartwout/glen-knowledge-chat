"""Tests for dashboard.wallet.earn_dropship_margin (Task 5).

Uses the same FakeCursor / monkeypatched-_cursor seam as test_wallet.py.
"""

import pytest
import dashboard.wallet as wallet


# ── Re-use the same in-memory fake from test_wallet.py ───────────────────────

class _FakeCursor:
    def __init__(self, store):
        self.store = store
        self._result = []

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        p = list(params)
        if s.startswith("SELECT wallet_balance_cents FROM practitioners"):
            pid = p[0]
            self._result = [{"wallet_balance_cents": self.store["balances"].get(pid, 0)}]
        elif s.startswith("UPDATE practitioners SET wallet_balance_cents"):
            self.store["balances"][p[1]] = p[0]
            self._result = []
        elif s.startswith("UPDATE practitioners SET modules_completed"):
            self.store["modules"][p[1]] = p[0]
            self._result = []
        elif s.startswith("INSERT INTO wallet_ledger"):
            cols = ["practitioner_id", "entry_type", "amount_cents",
                    "balance_after_cents", "qbo_invoice_id", "module_slug",
                    "earn_period", "note"]
            self.store["ledger"].append(dict(zip(cols, p)))
            self._result = []
        elif "qbo_invoice_id = %s AND entry_type = %s" in s:
            inv, et = p[0], p[1]
            hit = any(r["qbo_invoice_id"] == inv and r["entry_type"] == et
                      for r in self.store["ledger"])
            self._result = [{"exists": 1}] if hit else []
        elif "entry_type = 'spend_module' AND earn_period = %s" in s:
            pid, period = p[0], p[1]
            hit = any(r["practitioner_id"] == pid
                      and r["entry_type"] == "spend_module"
                      and r["earn_period"] == period
                      for r in self.store["ledger"])
            self._result = [{"exists": 1}] if hit else []
        elif s.startswith("SELECT entry_type, amount_cents") and "wallet_ledger" in s:
            pid = p[0]
            rows = [r for r in self.store["ledger"] if r["practitioner_id"] == pid]
            self._result = list(reversed(rows))
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
def store(monkeypatch):
    st = {"balances": {}, "modules": {}, "ledger": []}
    monkeypatch.setattr(wallet, "_cursor", lambda: _FakeCtx(st))
    return st


# ── Test data ─────────────────────────────────────────────────────────────────

PID = "00000000-0000-0000-0000-000000000002"


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_earn_dropship_margin_credits_exact_margin(store):
    """earn_dropship_margin credits exactly margin_cents; ledger row has correct fields."""
    credited = wallet.earn_dropship_margin(PID, 2204, qbo_invoice_id="INV1")
    assert credited == 2204
    assert wallet.get_balance_cents(PID) == 2204
    assert len(store["ledger"]) == 1
    row = store["ledger"][0]
    assert row["entry_type"] == "earn_dropship_margin"
    assert row["amount_cents"] == 2204
    assert row["balance_after_cents"] == 2204
    assert row["qbo_invoice_id"] == "INV1"


def test_earn_dropship_margin_idempotent_same_invoice(store):
    """Second call with same qbo_invoice_id is a no-op; balance stays put."""
    assert wallet.earn_dropship_margin(PID, 2204, qbo_invoice_id="INV1") == 2204
    assert wallet.earn_dropship_margin(PID, 2204, qbo_invoice_id="INV1") == 0   # replay
    assert wallet.get_balance_cents(PID) == 2204
    assert len(store["ledger"]) == 1   # only one ledger row written


def test_earn_dropship_margin_distinct_invoices_stack(store):
    """Two different invoices each earn and stack correctly."""
    assert wallet.earn_dropship_margin(PID, 2204, qbo_invoice_id="INV1") == 2204
    assert wallet.earn_dropship_margin(PID, 1500, qbo_invoice_id="INV2") == 1500
    assert wallet.get_balance_cents(PID) == 3704
    assert len(store["ledger"]) == 2


def test_earn_dropship_margin_zero_writes_no_ledger_row(store):
    """margin_cents=0 is a zero-delta; _apply returns 0 and writes no ledger row
    (matching the existing earn_order behaviour for zero amounts)."""
    credited = wallet.earn_dropship_margin(PID, 0, qbo_invoice_id="INV-ZERO")
    assert credited == 0
    assert wallet.get_balance_cents(PID) == 0
    assert store["ledger"] == []


def test_earn_dropship_margin_negative_margin_clamps_to_zero(store):
    """Negative margin_cents is clamped to 0; no credit, no ledger row."""
    credited = wallet.earn_dropship_margin(PID, -500, qbo_invoice_id="INV-NEG")
    assert credited == 0
    assert wallet.get_balance_cents(PID) == 0
    assert store["ledger"] == []


def test_earn_dropship_margin_ref_stored_as_note(store):
    """Optional ref= is passed through to the ledger note field."""
    wallet.earn_dropship_margin(PID, 1000, qbo_invoice_id="INV-REF", ref="order-42")
    row = store["ledger"][0]
    assert row["note"] == "order-42"


# ── Regression: old earn_dropship (flat $20/bottle) still works ───────────────

def test_earn_dropship_flat_still_works(store):
    """Regression — the old flat earn_dropship function is unchanged."""
    credited = wallet.earn_dropship(PID, 3, qbo_invoice_id="DS-OLD-1")
    assert credited == 6000   # 3 bottles × $20
    assert wallet.get_balance_cents(PID) == 6000
    row = store["ledger"][0]
    assert row["entry_type"] == "earn_dropship"


def test_earn_dropship_and_margin_idempotency_is_independent(store):
    """earn_dropship and earn_dropship_margin share the same invoice namespace
    in the idempotency guard but use different entry_type values, so the same
    qbo_invoice_id can be used for one of each without blocking the other."""
    wallet.earn_dropship(PID, 1, qbo_invoice_id="SHARED-INV")
    wallet.earn_dropship_margin(PID, 500, qbo_invoice_id="SHARED-INV")
    assert wallet.get_balance_cents(PID) == 2500   # 2000 flat + 500 margin
    assert len(store["ledger"]) == 2
