# Order Payments Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class, correctable, multi-payment + refund ledger per order that is the app's source of truth and stays in sync with QuickBooks.

**Architecture:** A new `order_payments` table (one row per payment or refund) with all balances derived, never stored. A pure-function module `dashboard/order_payments.py` owns inserts, voids, balance math, and QBO/Stripe sync. Console routes expose add/refund/void/resync; `checkout-return` writes the ledger instead of QBO directly. Three read surfaces render the split. Sync is synchronous with an idempotent resync (Approach C), anchored on a stored `qbo_txn_id` so nothing is ever double-posted.

**Tech Stack:** Python 3 / Flask, sqlite3 (`chat_log.db` = `LOG_DB`), QuickBooks Online (`dashboard/qbo_billing.py`), Stripe (`dashboard/stripe_pay.py`), vanilla-JS static pages.

## Global Constraints

- **DB:** all tables live in `LOG_DB` (`/data/chat_log.db` on prod, mounted only in the web container). Connect with `_sqlite3.connect(LOG_DB)` and `cx.row_factory = _sqlite3.Row`.
- **Money is integer cents.** `amount_cents` is always a positive integer; sign is carried by `kind` (`payment` | `refund`).
- **Auth:** payment/refund/void/resync routes require an actor via `_bos_actor()` returning non-None with role OWNER or OPS (same tier as `finance.record_payment`, which is `MONEY_SEND` / `(OWNER, OPS)`). VA tokens are rejected.
- **Idempotency anchor:** every synced row stores `qbo_txn_id`; a re-push is skipped when it is already set. Auto (Stripe) payment rows are also idempotent on `external_ref` (the PaymentIntent id).
- **QBO/Stripe are called as module functions** (`from dashboard import qbo_billing, stripe_pay`) so tests monkeypatch them — do NOT inject them as parameters.
- **Test isolation:** honor the `PYTEST_CURRENT_TEST` email guard already in the app; run app-importing tests with `doppler -c dev pytest` (bare `pytest` silently skips them).
- **Existing signatures to reuse verbatim:**
  - `qbo_billing.record_payment(customer_id, amount_cents, invoice_id, method=None)` → returns the QBO Payment dict (has `["Id"]`).
  - `qbo_billing.get_invoice(invoice_id)` → dict with `["CustomerRef"]["value"]`, or None.
  - `qbo_billing._post(path, body)` → posts to QBO, returns json.
  - `stripe_pay.refund(payment_intent, amount_cents=None)` → `{id, status, amount}`.
  - `dashboard/orders.py`: `get_order(cx, order_id)` → dict incl. `external_ref`, `total_cents`, `invoice_token`, `pay_status`; `find_order_by_external_ref(cx, ref)`.

---

### Task 1: `order_payments` table + balance math (pure DB, no QBO)

**Files:**
- Create: `dashboard/order_payments.py`
- Test: `tests/test_order_payments.py`

**Interfaces:**
- Produces: `ensure_table(cx)`, `add_payment(cx, order_id, amount_cents, method, *, source="manual", external_ref=None, paid_at=None, note=None, actor=None)` → row dict, `list_payments(cx, order_id)` → list[dict] newest first, `balance(cx, order_id)` → `{"invoice_cents", "paid_cents", "refunded_cents", "balance_cents"}`. In this task `add_payment` does NOT sync to QBO yet (added in Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_order_payments.py
import sqlite3
import pytest
from dashboard import order_payments as op
from dashboard import orders


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    orders.init_orders_table(c)
    op.ensure_table(c)
    # a $412.82 order
    orders.upsert_order(c, source="qbo", external_ref="INV-1",
                        email="dana@example.com", total_cents=41282)
    return c


def _oid(cx):
    return cx.execute("SELECT id FROM orders").fetchone()[0]


def test_add_payment_reduces_balance(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 22291, "Credit card (Stripe)", source="stripe",
                   external_ref="pi_1")
    op.add_payment(cx, oid, 13100, "Zelle")
    b = op.balance(cx, oid)
    assert b["paid_cents"] == 35391
    assert b["refunded_cents"] == 0
    assert b["invoice_cents"] == 41282
    assert b["balance_cents"] == 5891


def test_overpayment_is_negative_balance(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 50000, "Zelle")
    assert op.balance(cx, oid)["balance_cents"] == -8718


def test_stripe_payment_idempotent_on_external_ref(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 22291, "Credit card (Stripe)", source="stripe",
                   external_ref="pi_1")
    op.add_payment(cx, oid, 22291, "Credit card (Stripe)", source="stripe",
                   external_ref="pi_1")
    rows = op.list_payments(cx, oid)
    assert len([r for r in rows if r["kind"] == "payment"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler -c dev run -- pytest tests/test_order_payments.py -v`
Expected: FAIL with `AttributeError: module 'dashboard.order_payments' has no attribute 'ensure_table'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/order_payments.py
"""Multi-payment + refund ledger per order. One row per payment or refund;
balances are always derived, never stored. Functions take a sqlite connection
for testability. QBO/Stripe sync lives in this module (Tasks 2-3) and is called
as module functions so tests can monkeypatch them."""
import sqlite3
from datetime import datetime, timezone

from dashboard import orders

_METHODS = ("Credit card (Stripe)", "eProcessing", "Check", "Cash",
            "Venmo", "PayPal", "Zelle", "Wise")


def _now():
    return datetime.now(timezone.utc).isoformat()


def ensure_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS order_payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            kind TEXT NOT NULL DEFAULT 'payment',
            amount_cents INTEGER NOT NULL,
            method TEXT,
            source TEXT NOT NULL DEFAULT 'manual',
            external_ref TEXT,
            refunds_payment_id INTEGER,
            paid_at TEXT,
            note TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            void_reason TEXT,
            voided_at TEXT,
            qbo_txn_id TEXT,
            qbo_sync TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT,
            created_by TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_order_payments_order "
               "ON order_payments(order_id)")


def _row(cx, pid):
    r = cx.execute("SELECT * FROM order_payments WHERE id=?", (pid,)).fetchone()
    return dict(r) if r else None


def list_payments(cx, order_id):
    rows = cx.execute(
        "SELECT * FROM order_payments WHERE order_id=? ORDER BY id DESC",
        (order_id,)).fetchall()
    return [dict(r) for r in rows]


def _sum(cx, order_id, kind):
    v = cx.execute(
        "SELECT COALESCE(SUM(amount_cents),0) FROM order_payments "
        "WHERE order_id=? AND kind=? AND status='active'",
        (order_id, kind)).fetchone()[0]
    return int(v or 0)


def balance(cx, order_id):
    o = orders.get_order(cx, order_id) or {}
    invoice = int(o.get("total_cents") or 0)
    paid = _sum(cx, order_id, "payment")
    refunded = _sum(cx, order_id, "refund")
    return {"invoice_cents": invoice, "paid_cents": paid,
            "refunded_cents": refunded,
            "balance_cents": invoice - (paid - refunded)}


def _insert(cx, order_id, *, kind, amount_cents, method, source, external_ref,
            refunds_payment_id, paid_at, note, actor):
    now = _now()
    cur = cx.execute(
        "INSERT INTO order_payments (order_id, kind, amount_cents, method, "
        "source, external_ref, refunds_payment_id, paid_at, note, status, "
        "qbo_sync, created_at, updated_at, created_by) "
        "VALUES (?,?,?,?,?,?,?,?,?,'active','pending',?,?,?)",
        (order_id, kind, int(amount_cents), method, source, external_ref,
         refunds_payment_id, paid_at or now, note, now, now, actor))
    cx.commit()
    return _row(cx, cur.lastrowid)


def add_payment(cx, order_id, amount_cents, method, *, source="manual",
                external_ref=None, paid_at=None, note=None, actor=None):
    if int(amount_cents) <= 0:
        raise ValueError("amount_cents must be positive")
    if external_ref:
        dup = cx.execute(
            "SELECT id FROM order_payments WHERE order_id=? AND kind='payment' "
            "AND external_ref=? AND status='active'",
            (order_id, external_ref)).fetchone()
        if dup:
            return _row(cx, dup[0])
    return _insert(cx, order_id, kind="payment", amount_cents=amount_cents,
                   method=method, source=source, external_ref=external_ref,
                   refunds_payment_id=None, paid_at=paid_at, note=note,
                   actor=actor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler -c dev run -- pytest tests/test_order_payments.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/order_payments.py tests/test_order_payments.py
git commit -m "feat: order_payments table + balance math (no sync yet)"
```

---

### Task 2: Void + QBO sync for payments (mocked QBO)

**Files:**
- Modify: `dashboard/order_payments.py`
- Modify: `dashboard/qbo_billing.py` (add `void_payment`)
- Test: `tests/test_order_payments.py`

**Interfaces:**
- Consumes: `qbo_billing.record_payment`, `qbo_billing.get_invoice`, `orders.get_order`.
- Produces: `void(cx, payment_id, reason, *, actor=None)` → row, `resync(cx, payment_id)` → row, and `add_payment` now pushes to QBO (sets `qbo_txn_id`, `qbo_sync='synced'` on success, `'error'` on exception). New `qbo_billing.void_payment(qbo_txn_id)`.

- [ ] **Step 1: Write the failing test**

```python
def test_add_payment_syncs_to_qbo(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "412.82"})
    calls = {}
    monkeypatch.setattr(qbo_billing, "record_payment",
                        lambda cid, amt, iid, method=None: calls.update(
                            cid=cid, amt=amt, iid=iid, method=method) or {"Id": "P9"})
    oid = _oid(cx)
    row = op.add_payment(cx, oid, 13100, "Zelle")
    assert row["qbo_txn_id"] == "P9"
    assert row["qbo_sync"] == "synced"
    assert calls == {"cid": "42", "amt": 13100, "iid": "INV-1", "method": "Zelle"}


def test_void_excludes_from_balance_and_calls_qbo(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment",
                        lambda *a, **k: {"Id": "P9"})
    voided = {}
    monkeypatch.setattr(qbo_billing, "void_payment",
                        lambda txn: voided.update(txn=txn))
    oid = _oid(cx)
    row = op.add_payment(cx, oid, 13100, "Zelle")
    op.void(cx, row["id"], "keyed wrong amount")
    assert op.balance(cx, oid)["paid_cents"] == 0
    assert voided == {"txn": "P9"}
    # voiding again is a no-op (no second QBO call)
    voided.clear()
    op.void(cx, row["id"], "again")
    assert voided == {}


def test_void_null_txn_skips_qbo(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: None)  # push fails
    called = {"n": 0}
    monkeypatch.setattr(qbo_billing, "void_payment",
                        lambda txn: called.__setitem__("n", called["n"] + 1))
    oid = _oid(cx)
    row = op.add_payment(cx, oid, 100, "Cash")   # qbo_sync becomes 'error', no txn
    assert row["qbo_sync"] == "error" and row["qbo_txn_id"] is None
    op.void(cx, row["id"], "typo")
    assert called["n"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler -c dev run -- pytest tests/test_order_payments.py -k "qbo or void" -v`
Expected: FAIL (`record_payment`/`void_payment` not called; `void` missing)

- [ ] **Step 3: Add `void_payment` to qbo_billing**

Append after `record_payment` in `dashboard/qbo_billing.py`:

```python
def void_payment(qbo_txn_id):
    """Delete a QBO Payment (used when a console payment row is voided). QBO
    'delete' operation on the Payment entity. Best-effort; raises on API error."""
    _post("/payment?operation=delete", {"Id": str(qbo_txn_id), "SyncToken": "0"})
```

- [ ] **Step 4: Add QBO push + void/resync to order_payments.py**

Add these helpers and edit `add_payment` to call `_push_payment`:

```python
from dashboard import qbo_billing


def _qbo_ctx(cx, order_id):
    """Resolve (customer_id, qbo_invoice_id) for an order via its stored QBO
    invoice ref (orders.external_ref). Returns (None, None) if unresolvable."""
    o = orders.get_order(cx, order_id) or {}
    inv_id = o.get("external_ref")
    if not inv_id:
        return None, None
    inv = qbo_billing.get_invoice(inv_id)
    if not inv:
        return None, inv_id
    return (inv.get("CustomerRef") or {}).get("value"), inv_id


def _mark_sync(cx, pid, *, qbo_txn_id=None, state):
    cx.execute("UPDATE order_payments SET qbo_txn_id=COALESCE(?, qbo_txn_id), "
               "qbo_sync=?, updated_at=? WHERE id=?",
               (qbo_txn_id, state, _now(), pid))
    cx.commit()


def _push_payment(cx, pid):
    row = _row(cx, pid)
    if row.get("qbo_txn_id"):
        return  # already synced — idempotent
    try:
        cid, inv_id = _qbo_ctx(cx, row["order_id"])
        if not cid or not inv_id:
            raise RuntimeError("no QBO invoice/customer for order")
        res = qbo_billing.record_payment(cid, row["amount_cents"], inv_id,
                                         method=row["method"])
        _mark_sync(cx, pid, qbo_txn_id=(res or {}).get("Id"), state="synced")
    except Exception:
        _mark_sync(cx, pid, state="error")
```

Then, at the end of `add_payment`, replace `return _insert(...)` with:

```python
    row = _insert(cx, order_id, kind="payment", amount_cents=amount_cents,
                  method=method, source=source, external_ref=external_ref,
                  refunds_payment_id=None, paid_at=paid_at, note=note,
                  actor=actor)
    _push_payment(cx, row["id"])
    return _row(cx, row["id"])
```

And add `void` + `resync`:

```python
def void(cx, payment_id, reason, *, actor=None):
    row = _row(cx, payment_id)
    if not row or row["status"] == "void":
        return row
    if row.get("qbo_txn_id"):
        try:
            qbo_billing.void_payment(row["qbo_txn_id"])
        except Exception:
            pass  # keep the app void; row stays flagged for resync/repair
    cx.execute("UPDATE order_payments SET status='void', void_reason=?, "
               "voided_at=?, updated_at=? WHERE id=?",
               (reason, _now(), _now(), payment_id))
    cx.commit()
    return _row(cx, payment_id)


def resync(cx, payment_id):
    row = _row(cx, payment_id)
    if not row or row["status"] == "void":
        return row
    if row["kind"] == "payment":
        _push_payment(cx, payment_id)
    else:
        _push_refund(cx, payment_id)   # defined in Task 3
    return _row(cx, payment_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `doppler -c dev run -- pytest tests/test_order_payments.py -v`
Expected: PASS (all Task 1 + Task 2 tests)

- [ ] **Step 6: Commit**

```bash
git add dashboard/order_payments.py dashboard/qbo_billing.py tests/test_order_payments.py
git commit -m "feat: QBO sync + void + resync for order payments (idempotent on qbo_txn_id)"
```

---

### Task 3: Refunds — record + card auto-refund via Stripe (mocked)

**Files:**
- Modify: `dashboard/order_payments.py`
- Modify: `dashboard/qbo_billing.py` (add `record_refund`)
- Test: `tests/test_order_payments.py`

**Interfaces:**
- Consumes: `stripe_pay.refund`, `qbo_billing.record_refund`.
- Produces: `add_refund(cx, order_id, amount_cents, method, *, refunds_payment_id=None, note=None, actor=None)` → row; `_push_refund(cx, pid)`; `refundable_cents(cx, order_id, refunds_payment_id=None)` → int. New `qbo_billing.record_refund(customer_id, amount_cents, invoice_id, method=None)`.

- [ ] **Step 1: Write the failing test**

```python
def test_refund_reduces_paid_and_guards_overrefund(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    monkeypatch.setattr(qbo_billing, "record_refund", lambda *a, **k: {"Id": "R1"})
    oid = _oid(cx)
    pay = op.add_payment(cx, oid, 13100, "Zelle")
    op.add_refund(cx, oid, 5000, "Zelle", refunds_payment_id=pay["id"])
    b = op.balance(cx, oid)
    assert b["refunded_cents"] == 5000
    assert b["paid_cents"] == 13100
    assert b["balance_cents"] == 41282 - (13100 - 5000)
    # cannot refund more than the payment's un-refunded remainder (8100 left)
    with pytest.raises(ValueError):
        op.add_refund(cx, oid, 9000, "Zelle", refunds_payment_id=pay["id"])


def test_card_refund_calls_stripe_noncard_does_not(cx, monkeypatch):
    from dashboard import qbo_billing, stripe_pay
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    monkeypatch.setattr(qbo_billing, "record_refund", lambda *a, **k: {"Id": "R1"})
    sr = {}
    monkeypatch.setattr(stripe_pay, "refund",
                        lambda pi, amount_cents=None: sr.update(pi=pi, amt=amount_cents)
                        or {"id": "re_1", "status": "succeeded", "amount": amount_cents})
    oid = _oid(cx)
    card = op.add_payment(cx, oid, 22291, "Credit card (Stripe)",
                          source="stripe", external_ref="pi_9")
    zelle = op.add_payment(cx, oid, 13100, "Zelle")
    r1 = op.add_refund(cx, oid, 10000, "Credit card (Stripe)",
                       refunds_payment_id=card["id"])
    assert sr == {"pi": "pi_9", "amt": 10000}
    assert r1["external_ref"] == "re_1"
    sr.clear()
    op.add_refund(cx, oid, 5000, "Zelle", refunds_payment_id=zelle["id"])
    assert sr == {}   # non-card refund did not touch Stripe


def test_void_refund_reapplies_paid(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    monkeypatch.setattr(qbo_billing, "record_refund", lambda *a, **k: {"Id": "R1"})
    monkeypatch.setattr(qbo_billing, "void_payment", lambda txn: None)
    oid = _oid(cx)
    pay = op.add_payment(cx, oid, 13100, "Zelle")
    ref = op.add_refund(cx, oid, 5000, "Zelle", refunds_payment_id=pay["id"])
    op.void(cx, ref["id"], "issued in error")
    assert op.balance(cx, oid)["refunded_cents"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler -c dev run -- pytest tests/test_order_payments.py -k refund -v`
Expected: FAIL (`add_refund` / `record_refund` missing)

- [ ] **Step 3: Add `record_refund` to qbo_billing**

Append after `void_payment` in `dashboard/qbo_billing.py`:

```python
def record_refund(customer_id, amount_cents, invoice_id, method=None):
    """Record money-out against the customer for a refund on an invoice, as a QBO
    RefundReceipt. Amount in cents. `method` is memoed for traceability."""
    amt = round(int(amount_cents) / 100.0, 2)
    body = {
        "CustomerRef": {"value": str(customer_id)},
        "TotalAmt": amt,
        "Line": [{"Amount": amt, "DetailType": "SalesItemLineDetail",
                  "SalesItemLineDetail": {}}],
        "PrivateNote": "Console refund — invoice " + str(invoice_id)
                       + (" — method: " + str(method) if method else ""),
    }
    return _post("/refundreceipt", body).get("RefundReceipt")
```

- [ ] **Step 4: Add refund logic to order_payments.py**

```python
from dashboard import stripe_pay


def refundable_cents(cx, order_id, refunds_payment_id=None):
    """How much can still be refunded. Against a specific payment: that payment's
    amount minus active refunds pointing at it. Otherwise: order net paid."""
    if refunds_payment_id is not None:
        pay = _row(cx, refunds_payment_id)
        if not pay or pay["kind"] != "payment" or pay["status"] != "active":
            return 0
        used = cx.execute(
            "SELECT COALESCE(SUM(amount_cents),0) FROM order_payments "
            "WHERE refunds_payment_id=? AND kind='refund' AND status='active'",
            (refunds_payment_id,)).fetchone()[0]
        return int(pay["amount_cents"]) - int(used or 0)
    b = balance(cx, order_id)
    return b["paid_cents"] - b["refunded_cents"]


def _push_refund(cx, pid):
    row = _row(cx, pid)
    if row.get("qbo_txn_id"):
        return
    try:
        cid, inv_id = _qbo_ctx(cx, row["order_id"])
        if not cid or not inv_id:
            raise RuntimeError("no QBO invoice/customer for order")
        res = qbo_billing.record_refund(cid, row["amount_cents"], inv_id,
                                        method=row["method"])
        _mark_sync(cx, pid, qbo_txn_id=(res or {}).get("Id"), state="synced")
    except Exception:
        _mark_sync(cx, pid, state="error")


def add_refund(cx, order_id, amount_cents, method, *, refunds_payment_id=None,
               note=None, actor=None):
    amt = int(amount_cents)
    if amt <= 0:
        raise ValueError("amount_cents must be positive")
    if amt > refundable_cents(cx, order_id, refunds_payment_id):
        raise ValueError("refund exceeds refundable amount")
    external_ref = None
    src_pay = _row(cx, refunds_payment_id) if refunds_payment_id else None
    if src_pay and src_pay.get("source") == "stripe" and src_pay.get("external_ref"):
        sr = stripe_pay.refund(src_pay["external_ref"], amount_cents=amt)
        external_ref = sr.get("id")
    row = _insert(cx, order_id, kind="refund", amount_cents=amt, method=method,
                  source=("stripe" if external_ref else "manual"),
                  external_ref=external_ref, refunds_payment_id=refunds_payment_id,
                  paid_at=None, note=note, actor=actor)
    _push_refund(cx, row["id"])
    return _row(cx, row["id"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `doppler -c dev run -- pytest tests/test_order_payments.py -v`
Expected: PASS (all ledger tests)

- [ ] **Step 6: Commit**

```bash
git add dashboard/order_payments.py dashboard/qbo_billing.py tests/test_order_payments.py
git commit -m "feat: refunds with card auto-refund via Stripe + over-refund guard"
```

---

### Task 4: Console routes (list / add payment / add refund / void / resync)

**Files:**
- Modify: `app.py` (add routes near the other `/api/orders/...` routes, ~line 37147)
- Test: `tests/test_order_payments_routes.py`

**Interfaces:**
- Consumes: `order_payments.*`, `_bos_actor()`, `LOG_DB`, `_sqlite3`.
- Produces routes:
  - `GET  /api/orders/<int:oid>/payments` → `{"ok":True,"rows":[...],"balance":{...}}`
  - `POST /api/orders/<int:oid>/payments` → body `{amount, method, paid_at?, external_ref?, note?}` (amount in dollars)
  - `POST /api/orders/<int:oid>/refunds` → body `{amount, method, refunds_payment_id?, note?}`
  - `POST /api/orders/payments/<int:pid>/void` → body `{reason}`
  - `POST /api/orders/payments/<int:pid>/resync`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_order_payments_routes.py
import json
import pytest


@pytest.fixture
def client(monkeypatch):
    import app as A
    from dashboard import qbo_billing, orders, order_payments
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "9"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    # seed one order in the app's LOG_DB
    import sqlite3
    cx = sqlite3.connect(A.LOG_DB); cx.row_factory = sqlite3.Row
    orders.init_orders_table(cx); order_payments.ensure_table(cx)
    orders.upsert_order(cx, source="qbo", external_ref="INV-1",
                        email="d@e.com", total_cents=41282)
    cx.close()
    A.app.config["TESTING"] = True
    return A.app.test_client()


def test_add_payment_requires_actor(client, monkeypatch):
    import app as A
    monkeypatch.setattr(A, "_bos_actor", lambda: None)
    r = client.post("/api/orders/1/payments", json={"amount": 131, "method": "Zelle"})
    assert r.status_code == 401


def test_add_payment_and_balance(client, monkeypatch):
    import app as A
    monkeypatch.setattr(A, "_bos_actor", lambda: {"role": "owner"})
    r = client.post("/api/orders/1/payments", json={"amount": 131.00, "method": "Zelle"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    g = client.get("/api/orders/1/payments").get_json()
    assert g["balance"]["paid_cents"] == 13100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler -c dev run -- pytest tests/test_order_payments_routes.py -v`
Expected: FAIL 404 (routes not registered)

- [ ] **Step 3: Add the routes** (insert after the `/api/orders/price-preview` route, ~app.py:37147)

```python
@app.route("/api/orders/<int:oid>/payments", methods=["GET"])
def api_order_payments_list(oid):
    if _bos_actor() is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _op.ensure_table(cx)
        return jsonify({"ok": True, "rows": _op.list_payments(cx, oid),
                        "balance": _op.balance(cx, oid)})
    finally:
        cx.close()


@app.route("/api/orders/<int:oid>/payments", methods=["POST"])
def api_order_payments_add(oid):
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    b = request.get_json(silent=True) or {}
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _op.ensure_table(cx)
        row = _op.add_payment(
            cx, oid, round(float(b.get("amount") or 0) * 100),
            b.get("method") or "", source=b.get("source") or "manual",
            external_ref=b.get("external_ref"), paid_at=b.get("paid_at"),
            note=b.get("note"), actor=str(actor))
        return jsonify({"ok": True, "row": row, "balance": _op.balance(cx, oid)})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        cx.close()


@app.route("/api/orders/<int:oid>/refunds", methods=["POST"])
def api_order_refunds_add(oid):
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    b = request.get_json(silent=True) or {}
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _op.ensure_table(cx)
        row = _op.add_refund(
            cx, oid, round(float(b.get("amount") or 0) * 100),
            b.get("method") or "",
            refunds_payment_id=b.get("refunds_payment_id"),
            note=b.get("note"), actor=str(actor))
        return jsonify({"ok": True, "row": row, "balance": _op.balance(cx, oid)})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        cx.close()


@app.route("/api/orders/payments/<int:pid>/void", methods=["POST"])
def api_order_payment_void(pid):
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    b = request.get_json(silent=True) or {}
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _op.ensure_table(cx)
        row = _op.void(cx, pid, b.get("reason") or "", actor=str(actor))
        oid = row["order_id"] if row else None
        return jsonify({"ok": True, "row": row,
                        "balance": _op.balance(cx, oid) if oid else None})
    finally:
        cx.close()


@app.route("/api/orders/payments/<int:pid>/resync", methods=["POST"])
def api_order_payment_resync(pid):
    if _bos_actor() is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _op.ensure_table(cx)
        row = _op.resync(cx, pid)
        return jsonify({"ok": True, "row": row})
    finally:
        cx.close()
```

- [ ] **Step 4: Add the import** near the other `dashboard` imports at the top of `app.py`:

```python
from dashboard import order_payments as _op
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `doppler -c dev run -- pytest tests/test_order_payments_routes.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_order_payments_routes.py
git commit -m "feat: console routes for order payments + refunds + void + resync"
```

---

### Task 5: `checkout-return` writes the ledger instead of QBO directly

**Files:**
- Modify: `app.py` (~8998, the `_kind == "in-house"` block in `begin_checkout_return`)
- Test: `tests/test_order_payments_routes.py`

**Interfaces:**
- Consumes: `_op.add_payment`, `_bos_orders.find_order_by_external_ref`, `_bos_orders.set_order_stripe_pi`.
- Produces: on a paid in-house return, exactly one `source='stripe'` ledger row keyed on the PaymentIntent id; a second hit adds none.

- [ ] **Step 1: Write the failing test**

```python
def test_checkout_return_creates_one_stripe_row(client, monkeypatch):
    import app as A
    from dashboard import stripe_pay, order_payments
    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "payment_intent": "pi_777",
        "amount_total": 22291,
        "metadata": {"kind": "in-house", "invoice_id": "INV-1", "customer_id": "42"}})
    client.get("/begin/checkout-return?kind=in-house&session_id=cs_1")
    client.get("/begin/checkout-return?kind=in-house&session_id=cs_1")  # retry
    import sqlite3
    cx = sqlite3.connect(A.LOG_DB); cx.row_factory = sqlite3.Row
    order_payments.ensure_table(cx)
    oid = cx.execute("SELECT id FROM orders WHERE external_ref='INV-1'").fetchone()[0]
    rows = [r for r in order_payments.list_payments(cx, oid)
            if r["kind"] == "payment" and r["source"] == "stripe"]
    cx.close()
    assert len(rows) == 1 and rows[0]["amount_cents"] == 22291
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler -c dev run -- pytest tests/test_order_payments_routes.py::test_checkout_return_creates_one_stripe_row -v`
Expected: FAIL (no stripe ledger row; today it calls `set_order_payment`)

- [ ] **Step 3: Replace the recording call** in the `_kind == "in-house" and inv` block. Change the body that currently calls `_bos_orders.set_order_payment(...)` to also write the ledger:

```python
                            if _oih:
                                _op.ensure_table(_cxih)
                                _op.add_payment(
                                    _cxih, _oih["id"],
                                    int(sess.get("amount_total") or 0),
                                    "Credit card (Stripe)", source="stripe",
                                    external_ref=pi_id)
                                _bos_orders.set_order_payment(
                                    _cxih, _oih["id"], method="card",
                                    amount_cents=int(sess.get("amount_total") or 0))
                                if pi_id:
                                    _bos_orders.set_order_stripe_pi(_cxih, _oih["id"], pi_id)
```

(The ledger owns the QBO push now; `set_order_payment` remains only to keep the order's legacy summary/`pay_status` accurate. The prior direct `qbo_billing.record_payment` call in this block, if present, is removed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler -c dev run -- pytest tests/test_order_payments_routes.py::test_checkout_return_creates_one_stripe_row -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_order_payments_routes.py
git commit -m "feat: checkout-return records the ledger (idempotent on PI), not QBO directly"
```

---

### Task 6: Ensure the table at startup + expose order QBO id to the order loader

**Files:**
- Modify: `app.py` (the startup schema-init block that calls the other `init_*`/`ensure_*` table functions)
- Modify: `app.py` (the order-edit loader that `order-new.html` reads — the endpoint returning the order dict for edit mode; add `payments`+`balance` to its JSON)
- Test: `tests/test_order_payments_routes.py`

**Interfaces:**
- Produces: `order_payments` table created on boot; the order-load JSON used by `order-new.html` edit mode includes `"payments": [...]` and `"balance": {...}`.

- [ ] **Step 1: Write the failing test**

```python
def test_order_load_includes_payments(client, monkeypatch):
    import app as A
    monkeypatch.setattr(A, "_bos_actor", lambda: {"role": "owner"})
    client.post("/api/orders/1/payments", json={"amount": 50, "method": "Cash"})
    # the loader the edit page uses (adjust path to the confirmed loader route):
    j = client.get("/api/orders/1/edit-data").get_json()
    assert "payments" in j and j["balance"]["paid_cents"] == 5000
```

> Implementer note: confirm the exact loader route name `order-new.html` calls in edit mode (search `order-new.html` for the `fetch(` that loads the order by id) and use that path in both the test and Step 3.

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler -c dev run -- pytest tests/test_order_payments_routes.py::test_order_load_includes_payments -v`
Expected: FAIL (`payments` absent)

- [ ] **Step 3: Wire it up.** (a) In the startup schema block, add:

```python
    _op.ensure_table(cx)
```

(b) In the order-edit loader route, before returning its JSON, add:

```python
        _op.ensure_table(cx)
        payload["payments"] = _op.list_payments(cx, oid)
        payload["balance"] = _op.balance(cx, oid)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler -c dev run -- pytest tests/test_order_payments_routes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_order_payments_routes.py
git commit -m "feat: create order_payments on boot + include payments in order-edit load"
```

---

### Task 7: Console order card — payments panel UI

**Files:**
- Modify: `static/order-new.html` (edit-mode render, near the `editInvoice()` / prefill block ~line 388-412)

**Interfaces:**
- Consumes: `GET/POST /api/orders/<oid>/payments`, `POST /api/orders/<oid>/refunds`, `POST /api/orders/payments/<pid>/void`, `POST /api/orders/payments/<pid>/resync`, and the `payments`/`balance` now in the edit-load JSON.

- [ ] **Step 1: Add a container** in the edit-mode section of the page (below the pricing/payment block):

```html
<div id="pay-panel" style="display:none;margin-top:18px">
  <h2>Payments</h2>
  <div id="pay-rows"></div>
  <div id="pay-balance" style="margin:10px 0;font-weight:600"></div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:end">
    <div><label>Amount $</label><input id="pay-amt" type="number" step="0.01" style="width:90px"></div>
    <div><label>Method</label>
      <select id="pay-method"><option>Credit card (Stripe)</option><option>eProcessing</option><option>Check</option><option>Cash</option><option>Venmo</option><option>PayPal</option><option>Zelle</option><option>Wise</option></select></div>
    <div><label>Date</label><input id="pay-date" type="date"></div>
    <button class="ghost" onclick="addPayment()">Add payment</button>
  </div>
</div>
```

- [ ] **Step 2: Add the JS** (near the other edit-mode functions):

```javascript
function renderPayments(rows, bal){
  const box=$("pay-rows");
  box.innerHTML = (rows||[]).map(r=>{
    const amt=(r.kind==='refund'?'-':'')+'$'+(r.amount_cents/100).toFixed(2);
    const voided=r.status==='void';
    const badge=r.kind==='refund'?' <span style="color:#c0392b">refund</span>':'';
    const sync=r.qbo_sync!=='synced'?` <button class="mini" onclick="resyncPay(${r.id})">resync</button>`:'';
    const act=voided?` <em>void: ${esc(r.void_reason||'')}</em>`
      :` <button class="mini" onclick="voidPay(${r.id})">void</button>`
       +(r.kind==='payment'?` <button class="mini" onclick="refundPay(${r.id},${r.amount_cents},'${esc(r.method||'')}',${r.source==='stripe'})">refund</button>`:'');
    return `<div style="${voided?'text-decoration:line-through;opacity:.6':''}">`
      +`${esc((r.paid_at||'').slice(0,10))} · ${esc(r.method||'')} · ${amt}${badge} `
      +`<small>[${esc(r.source)}]</small>${sync}${act}</div>`;
  }).join('') || '<em>No payments recorded yet.</em>';
  $("pay-balance").textContent =
    `Paid $${(bal.paid_cents/100).toFixed(2)} · Refunded $${(bal.refunded_cents/100).toFixed(2)} · Balance $${(bal.balance_cents/100).toFixed(2)}`;
}
async function reloadPayments(){
  const r=await fetch(`/api/orders/${EDIT_OID}/payments`,{headers:HEADERS});
  const j=await r.json(); if(j.ok) renderPayments(j.rows,j.balance);
}
async function addPayment(){
  const amt=parseFloat($("pay-amt").value); if(isNaN(amt)||amt<=0){toast("Enter an amount","error");return;}
  const r=await fetch(`/api/orders/${EDIT_OID}/payments`,{method:"POST",headers:HEADERS,
    body:JSON.stringify({amount:amt,method:$("pay-method").value,paid_at:$("pay-date").value||null})});
  const j=await r.json(); if(!j.ok){toast(j.error||"Failed","error");return;}
  $("pay-amt").value=""; renderPayments((await (await fetch(`/api/orders/${EDIT_OID}/payments`,{headers:HEADERS})).json()).rows,j.balance); toast("Payment recorded");
}
async function voidPay(pid){
  const reason=prompt("Reason for voiding this payment?"); if(reason===null)return;
  const r=await fetch(`/api/orders/payments/${pid}/void`,{method:"POST",headers:HEADERS,body:JSON.stringify({reason})});
  if((await r.json()).ok){toast("Voided");reloadPayments();}
}
async function refundPay(pid,amtCents,method,isCard){
  const def=(amtCents/100).toFixed(2);
  const v=prompt((isCard?"Refund to card via Stripe":"Record refund")+" — amount $",def); if(v===null)return;
  const amount=parseFloat(v); if(isNaN(amount)||amount<=0){toast("Bad amount","error");return;}
  const r=await fetch(`/api/orders/${EDIT_OID}/refunds`,{method:"POST",headers:HEADERS,
    body:JSON.stringify({amount,method,refunds_payment_id:pid})});
  const j=await r.json(); if(!j.ok){toast(j.error||"Failed","error");return;}
  toast("Refund recorded"); reloadPayments();
}
async function resyncPay(pid){
  await fetch(`/api/orders/payments/${pid}/resync`,{method:"POST",headers:HEADERS}); reloadPayments();
}
```

- [ ] **Step 3: Show the panel + seed it** inside `editInvoice()` prefill (where `o` is the loaded order, ~line 412):

```javascript
  $("pay-panel").style.display="block";
  if (o.payments) renderPayments(o.payments, o.balance);
  const today=new Date().toISOString().slice(0,10); $("pay-date").value=today;
```

- [ ] **Step 4: Manually verify** (render, don't trust the payload — see the render-verify memory):

Run the app locally (`doppler -c dev run -- python app.py`), open an order in edit mode at `/orders/new?edit=<oid>&key=<CONSOLE_SECRET>`, confirm the Payments panel lists rows, Add records a row, Void strikes it through, Refund on a card row prompts "to card via Stripe". Drive it in a headless/real browser, not just the API.

- [ ] **Step 5: Commit**

```bash
git add static/order-new.html
git commit -m "feat: payments panel on the console order card (add/void/refund/resync)"
```

---

### Task 8: Client invoice — show payments received + balance due

**Files:**
- Modify: `app.py` (the `/api/invoice/<token>` GET handler, ~line 37778) to include ledger data
- Modify: `static/` invoice page template that `/invoice/<token>` renders

**Interfaces:**
- Consumes: `_op.list_payments`, `_op.balance` keyed by the order behind the invoice token.

- [ ] **Step 1: Add ledger to the invoice JSON.** In `/api/invoice/<token>`, after resolving the order id (`oid`), add:

```python
        _op.ensure_table(cx)
        _bal = _op.balance(cx, oid)
        _pays = [p for p in _op.list_payments(cx, oid)
                 if p["status"] == "active"]
        payload["payments"] = [
            {"date": (p["paid_at"] or "")[:10], "method": p["method"],
             "kind": p["kind"], "amount_cents": p["amount_cents"]} for p in _pays]
        payload["balance_due_cents"] = _bal["balance_cents"]
        payload["refunded_cents"] = _bal["refunded_cents"]
```

- [ ] **Step 2: Render it** on the invoice page — add a "Payments received" list (payments) + a "Refunded" line (when `refunded_cents > 0`) + "Balance due $X.XX" from `balance_due_cents`. Voided rows are already excluded server-side.

- [ ] **Step 3: Manually verify** by opening a real `/invoice/<token>` for an order that has two payments: confirm both show, refunds show as a "Refunded" line, and Balance due equals invoice − (paid − refunded).

- [ ] **Step 4: Commit**

```bash
git add app.py static/
git commit -m "feat: client invoice shows payments received + balance due (net of refunds)"
```

---

### Task 9: Board + money view — paid/balance per order and unified payments list

**Files:**
- Modify: `app.py` (`/api/payments` list, ~line 38264) to union manual `order_payments` with the Stripe ledger
- Modify: `static/console-money.html` (render the added rows/columns)

**Interfaces:**
- Consumes: `_op` for a union query; existing `_bos_payments.list_payments`.

- [ ] **Step 1: Union manual payments into `/api/payments`.** After building the Stripe-sourced `rows`, append active non-stripe `order_payments` (payments and refunds) so Zelle/check/cash show alongside card, tagged `source`. Keep it read-only.

- [ ] **Step 2: Show per-order paid-vs-balance** on `/console/orders` (or the money view): for each order, display `balance(cx, oid)` as "Paid $X / Balance $Y".

- [ ] **Step 3: Manually verify** the money view lists a Zelle payment next to Stripe charges and an order shows the correct paid/balance split.

- [ ] **Step 4: Commit**

```bash
git add app.py static/console-money.html
git commit -m "feat: unified payments list + per-order paid/balance on the money view"
```

---

### Task 10: Reconcile Dana Tamraz's real order (runbook, not code)

**Files:** none (operational).

- [ ] **Step 1:** In the console, open Dana's order (velvetdobie@aol.com, $412.82, full six remedies) in edit mode.
- [ ] **Step 2:** Confirm the Stripe card row **$222.91** is present (auto-recorded by `checkout-return`). If it landed on a duplicate/sibling order, void it there.
- [ ] **Step 3:** Add payment **Zelle $131.00**.
- [ ] **Step 4:** Add payment **Zelle $58.91** (the third payment).
- [ ] **Step 5:** Confirm the panel reads Paid $412.82 · Balance $0.00, and `/invoice/<her token>` shows Balance due $0.00.
- [ ] **Step 6:** Delete any leftover duplicate invoices/orders from the earlier back-and-forth so only the single settled order remains.

---

## Self-Review Notes

- **Spec coverage:** data model (T1), void+re-add & QBO sync Approach C (T2), refunds incl. card auto-refund + guard (T3), routes/auth MONEY_SEND (T4), Stripe→ledger via checkout-return (T5), migration/boot + order-load (T6), three surfaces (T7 console, T8 client invoice, T9 board/money), Dana reconcile (T10). Full-history backfill is intentionally out of scope (later slice).
- **Type consistency:** `qbo_txn_id` used throughout (never `qbo_payment_id`); `balance()` always returns `invoice_cents/paid_cents/refunded_cents/balance_cents`; `add_payment`/`add_refund` take positive `amount_cents`, routes convert dollars→cents with `round(x*100)`.
- **Implementer confirmations (two spots):** the exact order-edit loader route name in T6/T8 (search `order-new.html` for its load `fetch`), and whether a direct `qbo_billing.record_payment` call already exists in the `checkout-return` in-house block to remove (T5).
