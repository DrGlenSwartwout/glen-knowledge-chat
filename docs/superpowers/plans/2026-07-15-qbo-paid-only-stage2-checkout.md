# QBO Paid-Only Stage 2 (Biofield + Begin Checkout) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `biofield_checkout` and `begin_checkout` (plus its concierge upsell) from checkout-time unpaid QBO invoices to payment-time paid Sales Receipts, keyed on a checkout token.

**Architecture:** Generate a `checkout_ref` token at checkout that replaces the QBO invoice Id as the order key and correlation ref. Persist the exact QBO line payload (`qbo_lines_json`) on the order so a line-faithful Sales Receipt can be booked when payment confirms, via an idempotent `book_sale_on_payment` helper wired into both the Stripe return handler and the alt-pay `set_order_payment` path. Drop `create_invoice` from both routes and rework `/begin/concierge/add` to append to the pending order instead of a live invoice.

**Tech Stack:** Python, Flask, SQLite, pytest, QuickBooks Online REST API (minor version 75), Stripe, vanilla JS frontend (`static/begin-buy.html`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-15-qbo-paid-only-stage2-checkout-design.md`. Stage 1 is merged: `qbo_billing.create_sales_receipt(customer, lines, *, discount_cents=0, tax_cents=0, email_to=None, bank_account_id=None) -> dict` exists.
- Booking is best-effort: a QBO failure must NEVER break the paying customer or the checkout path — swallow and log, never raise (mirror `_book_membership_qbo`, `app.py:9099`).
- Idempotency is mandatory: exactly one Sales Receipt per order, regardless of how many paid-transition paths fire. Key on the order's `qbo_sales_receipt_id`.
- Line-faithful receipts: the Sales Receipt carries the same line items the invoice would have (from `qbo_lines_json`), NOT a single lossy "total" line.
- Keep response/metadata FIELD names (`invoice_id`, etc.) — their VALUE becomes the `checkout_ref` token — so `static/begin-buy.html` and the Stripe-return `md.get("invoice_id")` read keep working.
- Do NOT touch `qbo_reconcile.py` — it excludes sources `biofield`/`funnel`.
- Run tests via `doppler run -- python3 -m pytest <specific file>` — never bare pytest, never the whole suite (bare pytest silently skips app-import tests; full suite can send real email).
- Amounts on the wire are dollars (2 dp); internal amounts integer cents.
- Zero backfill; new columns default NULL; go-forward only.

---

## Task 1: Order schema + setters for the paid-only join

**Files:**
- Modify: `dashboard/orders.py` — migration list (~line 66-113) and new setters near `set_order_stripe_pi` (~line 421)
- Test: `tests/test_orders_qbo_stage2_columns.py` (create)

**Interfaces:**
- Consumes: existing `_ensure_schema`/migration list, `_now()`, `find_order_by_external_ref`, `get_order` (both `SELECT *`, so new columns surface automatically).
- Produces:
  - `set_order_qbo_lines(cx, external_ref, payload: dict) -> bool` — stores `json.dumps(payload)` into `qbo_lines_json` for the row with that `external_ref`.
  - `set_order_sales_receipt_id(cx, order_id, sales_receipt_id: str) -> bool` — stamps `qbo_sales_receipt_id` (idempotency marker).
  - New columns `qbo_lines_json TEXT`, `qbo_sales_receipt_id TEXT` on `orders`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orders_qbo_stage2_columns.py`:

```python
import json, sqlite3
from dashboard import orders as O


def _fresh_db(tmp_path):
    db = str(tmp_path / "orders.db")
    cx = sqlite3.connect(db)
    O.ensure_schema(cx) if hasattr(O, "ensure_schema") else O._ensure_schema(cx)
    return cx


def _new_order(cx, ref="tok-abc"):
    return O.upsert_order(cx, source="biofield", external_ref=ref, email="a@b.com",
                          total_cents=30000)


def test_new_columns_exist(tmp_path):
    cx = _fresh_db(tmp_path)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(orders)")}
    assert "qbo_lines_json" in cols
    assert "qbo_sales_receipt_id" in cols


def test_set_and_read_qbo_lines_by_ref(tmp_path):
    cx = _fresh_db(tmp_path)
    _new_order(cx, "tok-1")
    payload = {"lines": [{"name": "Biofield", "amount": 300.0, "qty": 1}],
               "discount_cents": 0, "tax_cents": 0}
    assert O.set_order_qbo_lines(cx, "tok-1", payload) is True
    row = O.find_order_by_external_ref(cx, "tok-1")
    assert json.loads(row["qbo_lines_json"]) == payload


def test_set_sales_receipt_id_stamps_order(tmp_path):
    cx = _fresh_db(tmp_path)
    oid = _new_order(cx, "tok-2")
    assert O.set_order_sales_receipt_id(cx, oid, "SR123") is True
    assert O.get_order(cx, oid)["qbo_sales_receipt_id"] == "SR123"


def test_set_qbo_lines_unknown_ref_returns_false(tmp_path):
    cx = _fresh_db(tmp_path)
    assert O.set_order_qbo_lines(cx, "nope", {"lines": []}) is False
```

(If the schema init function is named differently, match the existing name — check the top of `dashboard/orders.py`. The `_fresh_db` helper already tries both `ensure_schema` and `_ensure_schema`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -- python3 -m pytest tests/test_orders_qbo_stage2_columns.py -v`
Expected: FAIL — columns absent / `set_order_qbo_lines` undefined.

- [ ] **Step 3: Add the migration columns**

In `dashboard/orders.py`, append two DDL entries to the `for ddl in (...)` migration tuple (the block ending ~line 113, right after the `invoice_token` entry):

```python
        # Stage 2 (QBO paid-only): the exact QBO line payload captured at checkout
        # ({"lines":[...],"discount_cents":N,"tax_cents":N}) so a line-faithful
        # Sales Receipt can be booked when payment confirms.
        "ALTER TABLE orders ADD COLUMN qbo_lines_json TEXT",
        # The QBO SalesReceipt Id booked for this order (paid-only). NULL until
        # payment books it; presence is the idempotency marker (never re-book).
        "ALTER TABLE orders ADD COLUMN qbo_sales_receipt_id TEXT",
```

- [ ] **Step 4: Add the setters**

In `dashboard/orders.py`, near `set_order_stripe_pi` (~line 421), add:

```python
def set_order_qbo_lines(cx, external_ref, payload):
    """Store the exact QBO line payload for the order with this external_ref, so the
    paid handler can book a line-faithful Sales Receipt. Returns False if no such row."""
    cur = cx.execute("UPDATE orders SET qbo_lines_json=?, updated_at=? WHERE external_ref=?",
                     (json.dumps(payload), _now(), str(external_ref)))
    cx.commit()
    return cur.rowcount > 0


def set_order_sales_receipt_id(cx, order_id, sales_receipt_id):
    """Stamp the booked QBO SalesReceipt Id onto an order (idempotency marker)."""
    cur = cx.execute("UPDATE orders SET qbo_sales_receipt_id=?, updated_at=? WHERE id=?",
                     (str(sales_receipt_id), _now(), order_id))
    cx.commit()
    return cur.rowcount > 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `doppler run -- python3 -m pytest tests/test_orders_qbo_stage2_columns.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add dashboard/orders.py tests/test_orders_qbo_stage2_columns.py
git commit -m "feat(orders): qbo_lines_json + qbo_sales_receipt_id columns and setters (Stage 2)"
```

---

## Task 2: `book_sale_on_payment` — the idempotent paid-only booking helper

**Files:**
- Create: `dashboard/qbo_sale.py`
- Test: `tests/test_book_sale_on_payment.py` (create)

**Interfaces:**
- Consumes: `qbo_billing.find_or_create_customer`, `qbo_billing.create_sales_receipt` (Stage 1), `orders.set_order_sales_receipt_id` (Task 1), `orders.get_order`.
- Produces: `book_sale_on_payment(cx, order: dict) -> str | None` — books one Sales Receipt from the order's `qbo_lines_json`, stamps `qbo_sales_receipt_id`, returns the receipt Id (or the existing one if already booked, or None on best-effort failure). Never raises.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_book_sale_on_payment.py`:

```python
import json, sqlite3, types
import pytest
from dashboard import qbo_sale


class _FakeQB:
    def __init__(self):
        self.receipts = 0
        self.last_lines = None
    def find_or_create_customer(self, email, name=""):
        return {"Id": "C1"}
    def create_sales_receipt(self, cust, lines, *, discount_cents=0, tax_cents=0, email_to=None):
        self.receipts += 1
        self.last_lines = lines
        self.last_discount = discount_cents
        self.last_tax = tax_cents
        return {"Id": f"SR{self.receipts}"}


def _order(**kw):
    base = {"id": 5, "email": "a@b.com", "name": "A", "qbo_sales_receipt_id": None,
            "qbo_lines_json": json.dumps({"lines": [{"name": "Biofield", "amount": 300.0, "qty": 1}],
                                          "discount_cents": 1500, "tax_cents": 0})}
    base.update(kw); return base


def test_books_line_faithful_receipt_and_stamps(monkeypatch):
    qb = _FakeQB()
    stamped = {}
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)
    monkeypatch.setattr(qbo_sale, "orders",
                        types.SimpleNamespace(set_order_sales_receipt_id=
                        lambda cx, oid, sr: stamped.setdefault(oid, sr)))
    sr = qbo_sale.book_sale_on_payment(None, _order())
    assert sr == "SR1"
    assert qb.receipts == 1
    assert qb.last_lines == [{"name": "Biofield", "amount": 300.0, "qty": 1}]
    assert qb.last_discount == 1500
    assert stamped == {5: "SR1"}


def test_idempotent_no_rebook_when_already_booked(monkeypatch):
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)
    monkeypatch.setattr(qbo_sale, "orders",
                        types.SimpleNamespace(set_order_sales_receipt_id=lambda *a, **k: None))
    out = qbo_sale.book_sale_on_payment(None, _order(qbo_sales_receipt_id="SRX"))
    assert out == "SRX"
    assert qb.receipts == 0


def test_best_effort_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("QBO down")
    monkeypatch.setattr(qbo_sale, "qbo_billing",
                        types.SimpleNamespace(find_or_create_customer=boom,
                                              create_sales_receipt=boom))
    monkeypatch.setattr(qbo_sale, "orders",
                        types.SimpleNamespace(set_order_sales_receipt_id=lambda *a, **k: None))
    assert qbo_sale.book_sale_on_payment(None, _order()) is None


def test_missing_lines_json_is_a_noop(monkeypatch):
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)
    monkeypatch.setattr(qbo_sale, "orders",
                        types.SimpleNamespace(set_order_sales_receipt_id=lambda *a, **k: None))
    assert qbo_sale.book_sale_on_payment(None, _order(qbo_lines_json=None)) is None
    assert qb.receipts == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -- python3 -m pytest tests/test_book_sale_on_payment.py -v`
Expected: FAIL — `No module named 'dashboard.qbo_sale'`.

- [ ] **Step 3: Implement `dashboard/qbo_sale.py`**

```python
"""Paid-only QBO booking: create ONE line-faithful Sales Receipt for an order the
moment it is confirmed paid. Idempotent (never re-books) and best-effort (never
raises into the payment path). See docs/superpowers/specs/2026-07-15-qbo-paid-only-
stage2-checkout-design.md."""
import json

from . import qbo_billing
from . import orders


def book_sale_on_payment(cx, order):
    """Book a QBO SalesReceipt for a PAID order from its stored qbo_lines_json.
    Returns the receipt Id (existing one if already booked; None on any failure or
    when there is nothing to book). Never raises."""
    try:
        existing = order.get("qbo_sales_receipt_id")
        if existing:
            return existing
        raw = order.get("qbo_lines_json")
        if not raw:
            return None
        payload = json.loads(raw) if isinstance(raw, str) else raw
        lines = payload.get("lines") or []
        if not lines:
            return None
        cust = qbo_billing.find_or_create_customer(order.get("email") or "",
                                                   order.get("name") or "")
        sr = qbo_billing.create_sales_receipt(
            cust, lines,
            discount_cents=int(payload.get("discount_cents") or 0),
            tax_cents=int(payload.get("tax_cents") or 0),
            email_to=order.get("email") or None)
        sr_id = sr.get("Id")
        if sr_id and order.get("id") is not None:
            orders.set_order_sales_receipt_id(cx, order["id"], sr_id)
        return sr_id
    except Exception as e:  # best-effort — must never break the payment path
        print(f"[qbo-sale] book_sale_on_payment skipped for order "
              f"{order.get('id')!r}: {e!r}", flush=True)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -- python3 -m pytest tests/test_book_sale_on_payment.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/qbo_sale.py tests/test_book_sale_on_payment.py
git commit -m "feat(qbo): book_sale_on_payment — idempotent line-faithful paid-only booking"
```

---

## Task 3: Convert `biofield_checkout` to paid-only

**Files:**
- Modify: `app.py` — `biofield_checkout` (~8302) and the Stripe-return biofield paid branch (~9403-9406 for `kind != "in-house"` / ~9606 biofield block)
- Test: `tests/test_biofield_checkout_paid_only.py` (create)

**Interfaces:**
- Consumes: `orders.set_order_qbo_lines` (Task 1), `qbo_sale.book_sale_on_payment` (Task 2), `_bos_orders.find_order_by_external_ref`.
- Produces: `biofield_checkout` no longer calls `create_invoice`; booking happens on payment.

- [ ] **Step 1: Read the current code**

Read `biofield_checkout` in full (`app.py:8302-8367`) and the Stripe-return handler biofield paths (`app.py:9378-9420` and `app.py:9606-9639`). Confirm where `kind=="biofield"` is marked paid today and where `record_payment` is called.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_biofield_checkout_paid_only.py`:

```python
import json
import app
from dashboard import qbo_billing


def _client():
    return app.app.test_client()


def test_biofield_checkout_creates_no_qbo_invoice(monkeypatch):
    """Guard (mutation-style): checkout must NOT POST an invoice to QBO."""
    def boom(*a, **k):
        raise AssertionError("biofield_checkout must not call create_invoice (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(app, "_biofield_enabled", lambda: True)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    # stub Stripe session creation so no network call
    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "create_checkout_session",
                        lambda *a, **k: {"url": "https://stripe.test/s"})
    r = _client().post("/biofield/checkout", json={"email": "a@b.com", "name": "A"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    # response carries a stable ref token under the invoice_id field (frontend compat)
    assert body.get("invoice_id")


def test_biofield_checkout_persists_qbo_lines(monkeypatch):
    monkeypatch.setattr(qbo_billing, "create_invoice",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no invoice")))
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(app, "_biofield_enabled", lambda: True)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "create_checkout_session", lambda *a, **k: {"url": "u"})
    r = _client().post("/biofield/checkout", json={"email": "b@b.com", "name": "B"})
    ref = r.get_json()["invoice_id"]
    import sqlite3
    from dashboard import orders as O
    cx = sqlite3.connect(app.LOG_DB); cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, ref)
    assert row is not None
    payload = json.loads(row["qbo_lines_json"])
    assert payload["lines"]  # line-faithful payload was stored
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `doppler run -- python3 -m pytest tests/test_biofield_checkout_paid_only.py -v`
Expected: FAIL — current `biofield_checkout` calls `create_invoice` (guard boom) and stores no `qbo_lines_json`.

- [ ] **Step 4: Rewrite the QBO block in `biofield_checkout`**

Replace the invoice block (`app.py:8337-8348`, from `from dashboard import qbo_billing as qb` through the `_ingest_order(...)` call) with:

```python
    checkout_ref = _uuid.uuid4().hex   # stable order/correlation key (no QBO invoice yet)
    qbo_payload = {"lines": pc["qbo_lines"],
                   "discount_cents": pc["discount_cents"] + redeemed, "tax_cents": 0}
    out = {"ok": True, "invoice_id": checkout_ref, "doc_number": "",
           "customer_id": "", "total": round(charged_cents / 100.0, 2)}

    _ingest_order(source="biofield", external_ref=checkout_ref, email=email, name=name,
                  items=pc["items_rec"], total_cents=charged_cents, channel="retail",
                  get_cents=pc["priced"]["get_cents"], discount_cents=pc["discount_cents"],
                  points_redeemed_cents=redeemed, shipping_cents=0)
    try:
        with _sqlite3.connect(LOG_DB) as _lcx:
            _bos_orders.set_order_qbo_lines(_lcx, checkout_ref, qbo_payload)
    except Exception as _e:
        print(f"[biofield] persist qbo_lines failed: {_e!r}", flush=True)
```

Then update the Stripe metadata block just below (`app.py:8353-8357`) so `invoice_id` carries the token and the description no longer needs a DocNumber:

```python
        metadata = {"kind": "biofield", "email": email,
                    "invoice_id": checkout_ref,
                    "customer_id": "",
                    "points_redeemed_cents": str(int(redeemed)),
                    "tier": tier}
        sess = stripe_pay.create_checkout_session(
            charged_cents, customer_email=email,
            description=f"{PROGRAM_TIERS[tier]['name']}",
            metadata=metadata, success_url=success,
            cancel_url=f"{PUBLIC_BASE_URL}/begin")
```

Ensure `import uuid as _uuid` exists at the top of `app.py` (add it if missing). Confirm `_sqlite3` is the module alias already used elsewhere in this file (it is — see `app.py:9379`).

- [ ] **Step 5: Book the Sales Receipt on payment (biofield return branch)**

In the Stripe-return handler, find the `kind=="biofield"` paid handling (`app.py:~9606`). Where the biofield order is resolved by `find_order_by_external_ref(bf_inv)` and marked paid, add a booking call using the resolved order. Replace the `record_payment`-to-invoice behavior for biofield with:

```python
                    from dashboard import qbo_sale as _qsale
                    _bo = _bos_orders.find_order_by_external_ref(_bcxo, bf_inv)
                    if _bo:
                        _qsale.book_sale_on_payment(_bcxo, dict(_bo))
```

Also ensure the generic `if _kind != "in-house": _qb_ret.record_payment(...)` path (`app.py:~9403-9406`) does NOT also fire for `kind=="biofield"` (guard it with `and _kind not in ("biofield",)`), so a paid-only order is never sent to the invoice-apply path. Read those lines and adjust the condition precisely.

- [ ] **Step 6: Run tests to verify they pass**

Run: `doppler run -- python3 -m pytest tests/test_biofield_checkout_paid_only.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Regression — biofield suites**

Run: `doppler run -- python3 -m pytest tests/test_biofield_checkout.py tests/test_biofield_cart.py -v`
Expected: PASS (or any failure also present on `main`).

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_biofield_checkout_paid_only.py
git commit -m "feat(qbo): biofield_checkout paid-only — token key + Sales Receipt on payment"
```

---

## Task 4: Convert `begin_checkout` to paid-only

**Files:**
- Modify: `app.py` — `begin_checkout` (~8371-8470) and the Stripe-return `kind=="funnel"` paid branch + `set_order_payment` alt-pay path
- Test: `tests/test_begin_checkout_paid_only.py` (create)

**Interfaces:**
- Consumes: `orders.set_order_qbo_lines`, `qbo_sale.book_sale_on_payment`, `_bos_orders.set_order_payment`, `_bos_orders.find_order_by_external_ref`.
- Produces: `begin_checkout` no longer calls `create_invoice`; correlation refs (referral/coupon/gift) carry `checkout_ref`; booking on payment (card + alt-pay).

- [ ] **Step 1: Read the current code**

Read `begin_checkout` in full (`app.py:8371-8470`) — note every use of `inv.get("Id")`: `_record_referral_if_any` (~8438), `coupons.mark_redeemed(order_ref=...)` (~8443), `referrals.record_redemption(...)` (~8452), the `out` response (~8457), and `_ingest_order` (~8470). Read the funnel paid path in the Stripe-return handler and `dashboard/orders.py:set_order_payment` (already books nothing today).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_begin_checkout_paid_only.py`:

```python
import json, sqlite3
import app
from dashboard import qbo_billing, orders as O


def _prep(monkeypatch, method="card"):
    def boom(*a, **k):
        raise AssertionError("begin_checkout must not call create_invoice (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "create_checkout_session", lambda *a, **k: {"url": "https://s.test"})


def test_begin_checkout_creates_no_qbo_invoice(monkeypatch):
    _prep(monkeypatch)
    # A valid product slug + minimal payload; adapt slug to a known catalog entry.
    r = app.app.test_client().post("/begin/checkout/PRODUCT_SLUG",
                                   json={"email": "c@b.com", "name": "C", "qty": 1,
                                         "method": "card"})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body.get("invoice_id")  # token under the compat field


def test_begin_checkout_persists_qbo_lines_and_token_ref(monkeypatch):
    _prep(monkeypatch)
    r = app.app.test_client().post("/begin/checkout/PRODUCT_SLUG",
                                   json={"email": "d@b.com", "name": "D", "qty": 1,
                                         "method": "card"})
    ref = r.get_json()["invoice_id"]
    cx = sqlite3.connect(app.LOG_DB); cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, ref)
    assert row is not None and row["source"] == "funnel"
    assert json.loads(row["qbo_lines_json"])["lines"]
```

Before running, replace `PRODUCT_SLUG` with a real active slug — find one via `grep -n 'slug' dashboard/*catalog* ` or an existing begin-checkout test fixture (`tests/test_begin_checkout_engine.py`).

- [ ] **Step 3: Run tests to verify they fail**

Run: `doppler run -- python3 -m pytest tests/test_begin_checkout_paid_only.py -v`
Expected: FAIL — `create_invoice` guard fires.

- [ ] **Step 4: Rewrite `begin_checkout`'s QBO block**

Generate the token before the QBO work and remove `create_invoice`. Replace `app.py:8434-8437` (from `cust = qb.find_or_create_customer(...)` through the `inv = qb.create_invoice(...)` call) with:

```python
    checkout_ref = _uuid.uuid4().hex
    qbo_payload = {
        "lines": pc["qbo_lines"] + _shipping_line(pc["shipping_cents"]),
        "discount_cents": pc["discount_cents"] + pc["points_redeemed_cents"] + _sc_apply,
        "tax_cents": 0}
```

Then rewrite each downstream `inv.get("Id")` to `checkout_ref`:
- `_record_referral_if_any(_ref_ctx, email, checkout_ref)`
- `coupons.mark_redeemed(_ccx, _self_coupon["code"], order_ref=checkout_ref)`
- `coupons.mark_redeemed(_gcx, _gift_coupon["code"], order_ref=checkout_ref)`
- `referrals.record_redemption(_gcx, _gift_coupon["code"], _gift_coupon["email"], email, checkout_ref)`

Replace the `out = {...}` response (`app.py:~8457`) with:

```python
    out = {"ok": True, "invoice_id": checkout_ref, "doc_number": "",
           "total": round(int(pc["priced"]["total_cents"]) / 100.0, 2),
           "method": method, "customer_id": ""}
```

Replace `_ingest_order(source="funnel", external_ref=inv.get("Id"), ...)` with `external_ref=checkout_ref`, and immediately after it persist the payload:

```python
    try:
        with _sqlite3.connect(LOG_DB) as _lcx:
            _bos_orders.set_order_qbo_lines(_lcx, checkout_ref, qbo_payload)
    except Exception as _e:
        print(f"[funnel] persist qbo_lines failed: {_e!r}", flush=True)
```

Update the Stripe metadata for this route to carry `"invoice_id": checkout_ref` (read the metadata dict in this route and set it). Keep `total_cents = pc["priced"]["total_cents"]` as the Stripe charge amount (it already is).

- [ ] **Step 5: Book on payment — funnel branch + alt-pay**

- **Stripe return (`kind=="funnel"`):** in the return handler, find where a funnel order is resolved (`find_order_by_external_ref`) and, replacing the `record_payment`-to-invoice call for `kind != "in-house"` (guard it to also exclude `"funnel"`), add:

```python
                    from dashboard import qbo_sale as _qsale
                    _fo = _bos_orders.find_order_by_external_ref(_cxo, inv)
                    if _fo:
                        _qsale.book_sale_on_payment(_cxo, dict(_fo))
```

- **Alt-pay:** the operator-confirm path calls `orders.set_order_payment`. Add booking right after an order flips paid there. In `dashboard/orders.py:set_order_payment` DO NOT import qbo (keep the DB layer pure) — instead book at the app-level caller of `set_order_payment` for in-house/funnel invoices (the `/api/invoice/<token>` operator confirm route). Read that route, resolve the order, and call `qbo_sale.book_sale_on_payment(cx, dict(order))` after `set_order_payment` returns True. Idempotency (Task 2) makes a double-book impossible if Stripe already booked.

- [ ] **Step 6: Run tests to verify they pass**

Run: `doppler run -- python3 -m pytest tests/test_begin_checkout_paid_only.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Regression — begin/coupon/referral suites**

Run: `doppler run -- python3 -m pytest tests/test_begin_checkout_engine.py tests/test_opens.py -v`
Expected: PASS (or any failure also on `main`). Also grep for coupon/referral test files and include any that touch `order_ref`.

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_begin_checkout_paid_only.py
git commit -m "feat(qbo): begin_checkout paid-only — token key across refs + Sales Receipt on payment"
```

---

## Task 5: Rework `/begin/concierge/add` + frontend response shape

**Files:**
- Modify: `app.py` — `begin_concierge_add` (~9913-9940)
- Modify: `static/begin-buy.html` — confirmation header (~745), `pay_link` fallback (~751)
- Test: `tests/test_concierge_add_paid_only.py` (create)

**Interfaces:**
- Consumes: `orders.find_order_by_external_ref`, `orders.set_order_qbo_lines`, `orders.upsert_order` (re-total).
- Produces: `/begin/concierge/add` appends to the pending order's `qbo_lines_json` + re-totals; NO QBO call pre-payment.

- [ ] **Step 1: Read the current code**

Read `begin_concierge_add` (`app.py:9913-9945`) — note it currently takes `invoice_id` and calls `qbo_billing.add_invoice_line`. Read `static/begin-buy.html:740-760` and `:883-895`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_concierge_add_paid_only.py`:

```python
import json, sqlite3
import app
from dashboard import qbo_billing, orders as O


def test_concierge_add_appends_to_order_no_qbo_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("concierge add must not call QBO pre-payment")
    monkeypatch.setattr(qbo_billing, "add_invoice_line", boom)
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    # seed a pending order with a token ref + one line
    cx = sqlite3.connect(app.LOG_DB); cx.row_factory = sqlite3.Row
    ref = "tok-concierge-1"
    O.upsert_order(cx, source="funnel", external_ref=ref, email="e@b.com", total_cents=8000)
    O.set_order_qbo_lines(cx, ref, {"lines": [{"name": "Base", "amount": 80.0, "qty": 1}],
                                    "discount_cents": 0, "tax_cents": 0})
    r = app.app.test_client().post("/begin/concierge/add",
                                   json={"slug": "ADDON_SLUG", "invoice_id": ref})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json().get("ok") is True
    row = O.find_order_by_external_ref(cx, ref)
    lines = json.loads(row["qbo_lines_json"])["lines"]
    assert len(lines) == 2  # the add-on line was appended
```

Replace `ADDON_SLUG` with a real catalog slug (same source as Task 4).

- [ ] **Step 3: Run test to verify it fails**

Run: `doppler run -- python3 -m pytest tests/test_concierge_add_paid_only.py -v`
Expected: FAIL — current handler calls `add_invoice_line` (guard boom).

- [ ] **Step 4: Rewrite `begin_concierge_add`**

Replace the `add_invoice_line` block. Resolve the order by the posted `invoice_id` (now the checkout_ref), price the add-on the same way the route already computes `unit`/`qty`, append a line to the order's `qbo_lines_json`, and re-total:

```python
    ref = (p.get("invoice_id") or "").strip()
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        order = _bos_orders.find_order_by_external_ref(cx, ref)
        if not order:
            return jsonify({"ok": False, "error": "order not found"}), 404
        payload = json.loads(order["qbo_lines_json"]) if order["qbo_lines_json"] else {"lines": [], "discount_cents": 0, "tax_cents": 0}
        payload["lines"].append({"name": p["name"], "amount": unit, "qty": qty})
        _bos_orders.set_order_qbo_lines(cx, ref, payload)
        new_total = int(order["total_cents"] or 0) + int(round(unit * qty * 100))
        _bos_orders.upsert_order(cx, source=order["source"], external_ref=ref,
                                 total_cents=new_total)
    finally:
        cx.close()
    return jsonify({"ok": True, "total": round(new_total / 100.0, 2)})
```

(Match the existing variable names for `unit`, `qty`, `p["name"]` in the current handler; read them first.)

- [ ] **Step 5: Update `static/begin-buy.html` response-shape**

- Line ~745: the header used `data.doc_number`. Since it's now empty, change the confirmation header to not show a blank invoice number, e.g.:

```javascript
      var head = document.getElementById('cc-head');
      var total = (data.total != null) ? data.total : '';
      head.textContent = 'Order received — $' + total + '.';
```

- Lines ~750-758: remove the `data.pay_link` fallback branch (it is inert — `_QBO_PAYMENTS_ACTIVE` is off). Keep the `data.stripe_url` redirect and the existing `payment_error`/email-note else. Verify by reading the block that card→stripe_url and the alt-pay path are untouched.

- [ ] **Step 6: Run test to verify it passes**

Run: `doppler run -- python3 -m pytest tests/test_concierge_add_paid_only.py -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Commit**

```bash
git add app.py static/begin-buy.html tests/test_concierge_add_paid_only.py
git commit -m "feat(qbo): concierge add appends to pending order; drop live-invoice + pay_link"
```

---

## Full-suite gate (end of plan)

- [ ] Money-path + checkout regression (diff FAILED vs `main` baseline):

Run: `doppler run -- python3 -m pytest tests/test_biofield_checkout.py tests/test_begin_checkout_engine.py tests/test_bos_finance.py tests/test_order_payments_routes.py tests/test_membership_products_fulfill.py -v`
Expected: PASS, or any failure also present on `main`.

- [ ] Manual/live (post-deploy, deployed env): run one real biofield or begin checkout end-to-end, pay, and confirm a **Sales Receipt** (no open balance) appears in QBO with the correct line items and no A/R invoice was created. Local QBO 400s are expected (`reference_qbo_local_token_stale`).

## Notes carried from Stage 1
- `_first_bank_account_id()` has no ORDER BY — non-deterministic with multiple QBO Bank accounts. Still applies (every Sales Receipt uses it). Out of scope here; harden before wide rollout.
