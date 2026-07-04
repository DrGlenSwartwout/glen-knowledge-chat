"""Tests for dashboard.wallet.earn_care_share / reverse_care_share (Task 3).

Two layers, mirroring sibling wallet tests:
  - Unit/call-shape tests that monkeypatch `_apply` directly (fast, isolates
    the wrapper's own logic: clamping, idempotency key, entry_type).
  - Integration-style tests against a FakeCursor monkeypatched onto `_cursor`
    (same pattern as tests/test_wallet.py) that exercise the REAL `_apply`,
    confirming the debit actually lands in the ledger/balance and that the
    idempotency dedupe (`_already_posted`) really works end-to-end.
"""

import pytest


PID = "00000000-0000-0000-0000-000000000002"


# ── Unit: call-shape via monkeypatched _apply ──────────────────────────────────

def test_earn_care_share_credits_once(monkeypatch):
    from dashboard import wallet
    calls = []

    def fake_apply(pid, kind, fn, *, qbo_invoice_id=None, module_slug=None,
                    earn_period=None, note=None, precheck=None):
        calls.append((pid, kind, qbo_invoice_id, fn(0)))
        return fn(0)

    monkeypatch.setattr(wallet, "_apply", fake_apply)
    amt = wallet.earn_care_share("prac-42", 4950, event_ref="care_share:7:3")
    assert amt == 4950
    assert calls == [("prac-42", "earn_care_share", "care_share:7:3", 4950)]


def test_earn_care_share_clamps_negative(monkeypatch):
    from dashboard import wallet
    monkeypatch.setattr(
        wallet, "_apply",
        lambda pid, kind, fn, *, qbo_invoice_id=None, module_slug=None,
        earn_period=None, note=None, precheck=None: fn(0),
    )
    assert wallet.earn_care_share("prac-42", -10, event_ref="care_share:7:3") == 0


def test_reverse_care_share_debits(monkeypatch):
    from dashboard import wallet
    seen = {}

    def fake_apply(pid, kind, fn, *, qbo_invoice_id=None, module_slug=None,
                    earn_period=None, note=None, precheck=None):
        seen["kind"] = kind
        seen["ref"] = qbo_invoice_id
        seen["note"] = note
        # simulate the real _apply: run precheck against a cursor stand-in
        # that reports the original credit's posted amount (this is what the
        # reversal must reverse — NOT a live-rate recompute).
        if precheck is not None and not precheck(object()):
            return 0
        return fn(1000)

    monkeypatch.setattr(wallet, "_apply", fake_apply)
    monkeypatch.setattr(wallet, "_posted_amount", lambda cur, ref, et: 4950)
    result = wallet.reverse_care_share("prac-42", event_ref="care_share:7:3")
    assert seen["kind"] == "reverse_care_share"
    assert seen["ref"] == "reverse:care_share:7:3"
    assert seen["note"] == "care_share_reversal"
    assert result == 4950  # positive magnitude reversed, mirrors redeem_for_order


# ── Integration: real _apply against a FakeCursor (mirrors test_wallet.py) ────

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
        elif s.startswith("INSERT INTO wallet_ledger"):
            cols = ["practitioner_id", "entry_type", "amount_cents",
                    "balance_after_cents", "qbo_invoice_id", "module_slug",
                    "earn_period", "note"]
            self.store["ledger"].append(dict(zip(cols, p)))
            self._result = []
        elif s.startswith("SELECT amount_cents FROM wallet_ledger"):
            inv, et = p[0], p[1]
            match = next((r for r in self.store["ledger"]
                          if r["qbo_invoice_id"] == inv and r["entry_type"] == et), None)
            self._result = [{"amount_cents": match["amount_cents"]}] if match else []
        elif "qbo_invoice_id = %s AND entry_type = %s" in s:
            inv, et = p[0], p[1]
            hit = any(r["qbo_invoice_id"] == inv and r["entry_type"] == et
                      for r in self.store["ledger"])
            self._result = [{"exists": 1}] if hit else []
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
    st = {"balances": {}, "ledger": []}
    import dashboard.wallet as wallet
    monkeypatch.setattr(wallet, "_cursor", lambda: _FakeCtx(st))
    return st


def test_earn_care_share_real_apply_credits_and_writes_ledger(store):
    from dashboard.wallet import earn_care_share, get_balance_cents
    credited = earn_care_share(PID, 4950, event_ref="care_share:7:3")
    assert credited == 4950
    assert get_balance_cents(PID) == 4950
    assert len(store["ledger"]) == 1
    row = store["ledger"][0]
    assert row["entry_type"] == "earn_care_share"
    assert row["qbo_invoice_id"] == "care_share:7:3"
    assert row["note"] == "care_share"
    assert row["amount_cents"] == 4950


def test_earn_care_share_real_apply_idempotent_per_event_ref(store):
    from dashboard.wallet import earn_care_share, get_balance_cents
    assert earn_care_share(PID, 4950, event_ref="care_share:7:3") == 4950
    assert earn_care_share(PID, 4950, event_ref="care_share:7:3") == 0  # dup, no-op
    assert get_balance_cents(PID) == 4950
    assert len(store["ledger"]) == 1


def test_reverse_care_share_real_apply_debits_balance(store):
    from dashboard.wallet import earn_care_share, reverse_care_share, get_balance_cents
    earn_care_share(PID, 4950, event_ref="care_share:7:3")
    reversed_amt = reverse_care_share(PID, event_ref="care_share:7:3")
    assert reversed_amt == 4950
    assert get_balance_cents(PID) == 0
    assert len(store["ledger"]) == 2
    rev_row = store["ledger"][1]
    assert rev_row["entry_type"] == "reverse_care_share"
    assert rev_row["qbo_invoice_id"] == "reverse:care_share:7:3"
    assert rev_row["note"] == "care_share_reversal"
    assert rev_row["amount_cents"] == -4950
    assert rev_row["balance_after_cents"] == 0


def test_reverse_care_share_real_apply_idempotent_on_reverse_key(store):
    from dashboard.wallet import earn_care_share, reverse_care_share, get_balance_cents
    earn_care_share(PID, 4950, event_ref="care_share:7:3")
    assert reverse_care_share(PID, event_ref="care_share:7:3") == 4950
    assert reverse_care_share(PID, event_ref="care_share:7:3") == 0  # dup reversal, no-op
    assert get_balance_cents(PID) == 0


def test_reverse_care_share_noop_when_original_credit_absent(store):
    """A reversal for an event_ref that was never credited must not debit the
    ledger (guards against reversing a care-share that doesn't exist)."""
    from dashboard.wallet import reverse_care_share, get_balance_cents
    result = reverse_care_share(PID, event_ref="care_share:never-credited")
    assert result == 0
    assert get_balance_cents(PID) == 0
    assert store["ledger"] == []


def test_reverse_care_share_reverses_posted_amount_not_current_rate(store):
    """Regression (money-correctness): the reversal must equal the amount
    ACTUALLY credited for event_ref, never a recompute at the doctor's
    CURRENT modules_completed/cert rate. Credit as if the doctor was at m=12
    (the top band, $49.50 on a $99 charge), then reverse — the new
    reverse_care_share signature takes no rate/modules/share_cents input at
    all, so there is no way for it to consult a since-changed rate; it can
    only read the ledger. Assert the debit is exactly 4950 regardless."""
    from dashboard.wallet import earn_care_share, reverse_care_share, get_balance_cents
    earn_care_share(PID, 4950, event_ref="care_share:9:1")  # posted as if m=12
    reversed_amt = reverse_care_share(PID, event_ref="care_share:9:1")
    assert reversed_amt == 4950
    assert get_balance_cents(PID) == 0
    rev_row = store["ledger"][1]
    assert rev_row["entry_type"] == "reverse_care_share"
    assert rev_row["amount_cents"] == -4950
