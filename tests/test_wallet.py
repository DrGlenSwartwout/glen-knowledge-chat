"""Tests for dashboard.wallet — Wellness Credit wallet (Phase 2).

Pure decision logic (caps/gates/amounts) is tested directly. The thin Supabase
wrappers are tested against an in-memory FakeCursor monkeypatched onto the
module's `_cursor` seam (the codebase has no real-Supabase tests; this mirrors
the monkeypatch style used elsewhere).
"""

from datetime import datetime

import pytest


# ── Pure logic ────────────────────────────────────────────────────────────────

def test_wholesale_orders_earn_nothing():
    from dashboard.wallet import earn_amount_order_cents
    assert earn_amount_order_cents(100000) == 0
    assert earn_amount_order_cents(999) == 0


def test_earn_amount_dropship_is_twenty_dollars_per_bottle():
    from dashboard.wallet import earn_amount_dropship_cents
    assert earn_amount_dropship_cents(1) == 2000     # $20/bottle
    assert earn_amount_dropship_cents(10) == 20000


def test_redeem_for_order_capped_at_fifty_percent_and_balance():
    from dashboard.wallet import redeem_amount_for_order_cents
    assert redeem_amount_for_order_cents(80000, 100000) == 50000   # 50% of order
    assert redeem_amount_for_order_cents(20000, 100000) == 20000   # capped by balance


def test_redeem_for_module_up_to_full_tuition_and_balance():
    from dashboard.wallet import redeem_amount_for_module_cents
    assert redeem_amount_for_module_cents(40000) == 29700   # up to 100% of $297
    assert redeem_amount_for_module_cents(10000) == 10000   # capped by balance


def test_period_key_is_year_month():
    from dashboard.wallet import period_key
    assert period_key(datetime(2026, 6, 1)) == "2026-06"
    assert period_key(datetime(2026, 12, 31)) == "2026-12"


# ── Fake Supabase cursor ──────────────────────────────────────────────────────

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
    import dashboard.wallet as wallet
    monkeypatch.setattr(wallet, "_cursor", lambda: _FakeCtx(st))
    return st


# ── Wrappers ──────────────────────────────────────────────────────────────────

PID = "00000000-0000-0000-0000-000000000001"


def test_get_balance_zero_for_untouched_practitioner(store):
    from dashboard.wallet import get_balance_cents
    assert get_balance_cents(PID) == 0


def test_earn_order_is_a_noop(store):
    from dashboard.wallet import earn_order, get_balance_cents
    credited = earn_order(PID, 100000, "inv-1")
    assert credited == 0
    assert get_balance_cents(PID) == 0
    assert store["ledger"] == []               # no row written


def test_earn_dropship_credits_twenty_per_bottle_and_writes_ledger(store):
    from dashboard.wallet import earn_dropship, get_balance_cents
    credited = earn_dropship(PID, 10, ref="ds-1")   # 10 bottles
    assert credited == 20000
    assert get_balance_cents(PID) == 20000
    assert len(store["ledger"]) == 1
    row = store["ledger"][0]
    assert row["entry_type"] == "earn_dropship"
    assert row["amount_cents"] == 20000
    assert row["balance_after_cents"] == 20000


def test_earn_dropship_idempotent_on_invoice(store):
    from dashboard.wallet import earn_dropship, get_balance_cents
    assert earn_dropship(PID, 10, qbo_invoice_id="DINV1") == 20000
    assert earn_dropship(PID, 10, qbo_invoice_id="DINV1") == 0   # same invoice, no double credit
    assert get_balance_cents(PID) == 20000
    assert earn_dropship(PID, 5, qbo_invoice_id="DINV2") == 10000  # distinct invoice stacks
    assert get_balance_cents(PID) == 30000


def test_earn_fee_free_credits_three_percent_idempotent(store):
    from dashboard.wallet import earn_fee_free, get_balance_cents
    assert earn_fee_free(PID, 100000, "FINV1") == 3000   # 3% of a $1000 order
    assert earn_fee_free(PID, 100000, "FINV1") == 0       # same invoice -> idempotent
    assert get_balance_cents(PID) == 3000
    assert earn_fee_free(PID, 33333, "FINV2") == 999      # floor(0.03*33333)
    assert get_balance_cents(PID) == 3999


def test_redeem_for_order_capped_at_half_and_never_negative(store):
    from dashboard.wallet import earn_dropship, redeem_for_order, get_balance_cents
    earn_dropship(PID, 50)                      # +100000 ($1000 credit)
    redeemed = redeem_for_order(PID, 100000, "inv-2")
    assert redeemed == 50000                    # 50% of the $1000 order
    assert get_balance_cents(PID) == 50000
    # nothing more on the same invoice (idempotent)
    assert redeem_for_order(PID, 100000, "inv-2") == 0
    assert get_balance_cents(PID) == 50000


def test_redeem_for_module_full_tuition_and_monthly_gate(store):
    from dashboard.wallet import earn_dropship, redeem_for_module, get_balance_cents
    earn_dropship(PID, 30)                      # +60000
    june = datetime(2026, 6, 10)
    july = datetime(2026, 7, 3)
    first = redeem_for_module(PID, "module-1", today=june)
    assert first == 29700                       # up to 100% of $297
    assert get_balance_cents(PID) == 30300
    # second redemption in the same month is gated
    assert redeem_for_module(PID, "module-2", today=june) == 0
    assert get_balance_cents(PID) == 30300
    # next month allowed again (capped by remaining balance)
    assert redeem_for_module(PID, "module-2", today=july) == 29700
    assert get_balance_cents(PID) == 600


def test_set_modules_completed_clamps(store):
    from dashboard.wallet import set_modules_completed
    assert set_modules_completed(PID, 20) == 12
    assert store["modules"][PID] == 12
    assert set_modules_completed(PID, -3) == 0
    assert store["modules"][PID] == 0
    assert set_modules_completed(PID, 6) == 6
