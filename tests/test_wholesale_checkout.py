"""Tests for dashboard.wholesale_checkout — the practitioner money path (Phase 3e):
price the cart -> create one QBO invoice -> redeem Wellness Credit -> apply the
discount. QBO and the wallet DB are faked; the orchestration is what's under test.
"""

import sqlite3
from datetime import datetime

import pytest

PID = "00000000-0000-0000-0000-000000000001"


# ── fakes ─────────────────────────────────────────────────────────────────────

class _FakeQB:
    def __init__(self):
        self.created = []
        self.discounts = []

    def find_or_create_customer(self, email, name=""):
        return {"Id": "C1", "DisplayName": name or email}

    def create_invoice(self, customer, lines, **kw):
        # mirror the real input contract: [{name, amount(unit $), qty, item_id, ...}]
        self.created.append({"lines": lines, "kw": kw})
        total = sum(float(l["amount"]) * int(l.get("qty", 1)) for l in lines)
        return {"Id": "INV1", "SyncToken": "0", "DocNumber": "1001", "TotalAmt": total}

    def apply_invoice_discount(self, invoice_id, discount_cents):
        self.discounts.append((invoice_id, discount_cents))
        return {"Id": invoice_id, "SyncToken": "1", "DocNumber": "1001", "TotalAmt": 0}


# reuse the wallet fake cursor shape
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
            self.store["balances"][p[1]] = p[0]; self._result = []
        elif s.startswith("INSERT INTO wallet_ledger"):
            cols = ["practitioner_id", "entry_type", "amount_cents", "balance_after_cents",
                    "qbo_invoice_id", "module_slug", "earn_period", "note"]
            self.store["ledger"].append(dict(zip(cols, p))); self._result = []
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
    def __init__(self, store): self.store = store
    def __enter__(self): return _FakeCursor(self.store)
    def __exit__(self, *a): return False


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
    store = {"balances": {}, "modules": {}, "ledger": []}
    fake_qb = _FakeQB()
    monkeypatch.setattr(wc, "qb", fake_qb)
    monkeypatch.setattr(wallet, "_cursor", lambda: _FakeCtx(store))
    db = _seed_box(tmp_path, "dropper", 20)
    catalog = {"x": {"name": "X Formula", "bottle_type": "dropper",
                     "price_cents": 7000, "qbo_item_id": "55"}}
    prac = {"id": PID, "modules_completed": 12, "email": "dr@x.com", "name": "Dr X"}
    return {"wc": wc, "wallet": wallet, "qb": fake_qb, "store": store,
            "db": db, "catalog": catalog, "prac": prac}


# ── build_order ───────────────────────────────────────────────────────────────

def test_build_order_certified_two_boxes_no_credit(env):
    """Paid-only (Stage 4): build_order creates NO QBO invoice/customer -- it
    returns a checkout_ref token + a line-faithful qbo_payload for the route to
    persist and the return-handler to book once payment is confirmed."""
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"],
                                db_path=env["db"], catalog=env["catalog"])
    assert out["ok"] is True
    assert out["blended_unit_price_cents"] == 2500     # certified floor
    assert out["subtotal_cents"] == 100000
    assert out["credit_redeemed_cents"] == 0
    assert env["qb"].created == []                      # no invoice created
    assert env["qb"].discounts == []                    # no discount applied
    assert out["customer_id"] == ""
    assert isinstance(out["invoice_id"], str) and len(out["invoice_id"]) == 32
    line = out["qbo_payload"]["lines"][0]
    assert line["item_id"] == "55"
    assert out["qbo_payload"]["discount_cents"] == 0


def test_build_order_applies_credit_capped_at_half(env):
    env["wallet"].earn_dropship(PID, 50)               # +$1000 credit
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"],
                                db_path=env["db"], catalog=env["catalog"])
    assert out["ok"] is True
    # 50% of the $1000 order = $500 redeemed (balance had $1000)
    assert out["credit_redeemed_cents"] == 50000
    assert env["qb"].discounts == []                    # no QBO discount call
    assert out["qbo_payload"]["discount_cents"] == 50000
    assert out["total"] == round((100000 - 50000) / 100.0, 2)
    assert env["wallet"].get_balance_cents(PID) == 50000   # halved
    # ledger has the spend_order keyed to the checkout_ref token (not an invoice)
    assert any(r["entry_type"] == "spend_order" and r["qbo_invoice_id"] == out["invoice_id"]
               for r in env["store"]["ledger"])


def test_build_order_zelle_earns_3pct_fee_free(env):
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"], method="zelle",
                                db_path=env["db"], catalog=env["catalog"])
    assert out["ok"] is True
    # certified 40 bottles = $1000 charged (no credit yet) -> 3% = $30 credit
    assert out["fee_free_credit_cents"] == 3000
    assert env["wallet"].get_balance_cents(PID) == 3000


def test_build_order_card_no_fee_free(env):
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"], method="card",
                                db_path=env["db"], catalog=env["catalog"])
    assert out["fee_free_credit_cents"] == 0
    assert env["wallet"].get_balance_cents(PID) == 0


def test_build_order_blocks_when_margin_breached(env):
    cat = {"x": {"name": "X", "bottle_type": "dropper", "price_cents": 7000,
                 "qbo_item_id": "55", "cogs_cents": 2000, "fulfillment_cents": 800}}
    out = env["wc"].build_order([{"slug": "x", "qty": 40}], env["prac"],
                                db_path=env["db"], catalog=cat)
    assert out["ok"] is False
    assert out["error"] == "margin_floor"
    assert env["qb"].created == []                      # no invoice created


def test_build_order_empty_cart(env):
    out = env["wc"].build_order([], env["prac"], db_path=env["db"], catalog=env["catalog"])
    assert out["ok"] is False
    assert out["error"] == "empty_cart"


# ── build_module_order (training-first redemption, up to 100%) ─────────────────

def test_build_module_order_redeems_up_to_full_tuition(env):
    env["wallet"].earn_dropship(PID, 50)               # +$1000 credit
    out = env["wc"].build_module_order(env["prac"], "module-1", today=datetime(2026, 6, 9))
    assert out["ok"] is True
    assert out["credit_redeemed_cents"] == 29700        # full $297 covered
    assert env["qb"].discounts == [("INV1", 29700)]
    assert env["wallet"].get_balance_cents(PID) == 70300


# ── quote_module (paid-only: no QBO invoice at signup) ─────────────────────────

def test_quote_module_is_paid_only_no_invoice(env):
    """A coach's module quote computes what's owed WITHOUT creating a QBO invoice
    or redeeming credit (that happens only when payment is recorded)."""
    out = env["wc"].quote_module(env["prac"], "module-1")
    assert out["ok"] is True
    assert out["tuition_cents"] == 29700
    assert out["amount_due_cents"] == out["tuition_cents"] - out["credit_available_cents"]
    assert out["total"] == round(out["amount_due_cents"] / 100.0, 2)
    assert env["qb"].created == []      # invariant: no A/R invoice minted
    assert env["qb"].discounts == []


def test_quote_module_previews_credit_without_spending(env):
    env["wallet"].earn_dropship(PID, 50)               # +$1000 credit balance
    before = env["wallet"].get_balance_cents(PID)
    out = env["wc"].quote_module(env["prac"], "module-1")
    assert out["credit_available_cents"] > 0           # some credit previewed
    assert out["amount_due_cents"] == 29700 - out["credit_available_cents"]
    assert env["wallet"].get_balance_cents(PID) == before   # NOT redeemed
    assert env["qb"].created == []                     # still no invoice
