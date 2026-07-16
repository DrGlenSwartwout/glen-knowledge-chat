# QBO Paid-Only (Sales Receipts) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write QBO records only after payment confirms, as Sales Receipts (never unpaid Invoices), starting with the primitive and the one already-post-payment flow (memberships).

**Architecture:** Add `create_sales_receipt` to `dashboard/qbo_billing.py` mirroring `create_invoice`'s line/discount/tax handling but posting a paid `/salesreceipt` (with `DepositToAccountRef`, no A/R). Convert the membership booking (`_book_membership_qbo`) — already a post-payment invoice+payment pair — to a single Sales Receipt. Verify the payload against live QBO via a console-key-gated diagnostic route before trusting it. Deeper flows (begin-checkout re-keying, wholesale/dropship) are staged into follow-on plans (see end).

**Tech Stack:** Python, Flask, SQLite, pytest, QuickBooks Online REST API (minor version 75), `requests`.

## Global Constraints

- QBO API minor version is `75` (`dashboard/qbo_billing.py:MINOR`) — all writes carry it via `_post`.
- A QBO failure in a booking helper must NEVER break the customer-facing transaction — booking is best-effort, logged, never raises to the caller (existing `_book_membership_qbo` contract).
- No change to how customers pay (Stripe / Zelle / Wise unchanged).
- No backfill of historical invoices.
- Amounts on the wire are dollars rounded to 2 dp; internal amounts are integer cents.
- Local QBO tokens 400 (prod rotates the refresh token to `/data`, not Doppler — see `reference_qbo_local_token_stale`). Live verification runs against the **deployed** environment via a gated route, not local pytest.
- Run tests via `doppler run -- python3 -m pytest` (bare pytest silently skips app-importing tests and can flood live email — `reference_deploy_chat_test_doppler_skip`, `feedback_pytest_floods_live_email`).

---

## Task 1: `create_sales_receipt` primitive

**Files:**
- Modify: `dashboard/qbo_billing.py` (add function after `replace_invoice_lines`, ~line 264)
- Test: `tests/test_qbo_sales_receipt.py` (create)

**Interfaces:**
- Consumes: `_build_invoice_lines(lines, discount_cents)`, `find_or_create_item(name, price)`, `_first_bank_account_id()`, `_post(path, body)` — all existing in `dashboard/qbo_billing.py`.
- Produces: `create_sales_receipt(customer: dict, lines: list, *, discount_cents: int = 0, tax_cents: int = 0, email_to: str | None = None, bank_account_id: str | None = None) -> dict` — returns the QBO SalesReceipt dict.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_qbo_sales_receipt.py`:

```python
import pytest
from dashboard import qbo_billing as qb


@pytest.fixture
def captured(monkeypatch):
    """Capture the body posted to _post, stubbing all QBO I/O."""
    sink = {}

    def fake_post(path, body):
        sink["path"] = path
        sink["body"] = body
        return {"SalesReceipt": {"Id": "SR1", "DocNumber": "1001", "TotalAmt": body_total(body)}}

    def body_total(body):
        return round(sum(l["Amount"] for l in body.get("Line", [])), 2)

    monkeypatch.setattr(qb, "_post", fake_post)
    monkeypatch.setattr(qb, "_first_bank_account_id", lambda: "BANK9")
    monkeypatch.setattr(qb, "find_or_create_item", lambda name, price=None: {"Id": "IT1"})
    return sink


def test_posts_to_salesreceipt_with_deposit_account(captured):
    out = qb.create_sales_receipt({"Id": "C1"},
                                  [{"name": "Widget", "amount": 10.0, "qty": 2}])
    assert captured["path"] == "/salesreceipt"
    body = captured["body"]
    assert body["CustomerRef"] == {"value": "C1"}
    assert body["DepositToAccountRef"] == {"value": "BANK9"}
    line = body["Line"][0]
    assert line["DetailType"] == "SalesItemLineDetail"
    assert line["Amount"] == 20.0
    assert line["SalesItemLineDetail"]["ItemRef"] == {"value": "IT1"}
    assert out["Id"] == "SR1"


def test_resolves_provided_item_id_without_lookup(captured, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("find_or_create_item must not be called when item_id given")
    monkeypatch.setattr(qb, "find_or_create_item", boom)
    qb.create_sales_receipt({"Id": "C1"},
                            [{"name": "Widget", "amount": 5.0, "qty": 1, "item_id": "PRE"}])
    assert captured["body"]["Line"][0]["SalesItemLineDetail"]["ItemRef"] == {"value": "PRE"}


def test_tax_cents_stamps_totaltax_override(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 100.0, "qty": 1}],
                            tax_cents=475)
    body = captured["body"]
    assert body["TxnTaxDetail"] == {"TotalTax": 4.75}
    assert body["GlobalTaxCalculation"] == "TaxExcluded"


def test_zero_tax_omits_tax_detail(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 100.0, "qty": 1}])
    assert "TxnTaxDetail" not in captured["body"]


def test_discount_cents_appends_discount_line(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 100.0, "qty": 1}],
                            discount_cents=1500)
    disc = [l for l in captured["body"]["Line"] if l["DetailType"] == "DiscountLineDetail"]
    assert len(disc) == 1
    assert disc[0]["Amount"] == 15.0


def test_email_to_sets_billemail(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 1.0, "qty": 1}],
                            email_to="a@b.com")
    assert captured["body"]["BillEmail"] == {"Address": "a@b.com"}


def test_raises_without_bank_account(monkeypatch):
    monkeypatch.setattr(qb, "_first_bank_account_id", lambda: None)
    monkeypatch.setattr(qb, "find_or_create_item", lambda name, price=None: {"Id": "IT1"})
    with pytest.raises(RuntimeError, match="bank account"):
        qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 1.0, "qty": 1}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -- python3 -m pytest tests/test_qbo_sales_receipt.py -v`
Expected: FAIL — `AttributeError: module 'dashboard.qbo_billing' has no attribute 'create_sales_receipt'`

- [ ] **Step 3: Implement `create_sales_receipt`**

Add to `dashboard/qbo_billing.py` immediately after `replace_invoice_lines` (~line 264):

```python
def create_sales_receipt(customer, lines, *, discount_cents=0, tax_cents=0,
                         email_to=None, bank_account_id=None):
    """Record a PAID sale as a QBO SalesReceipt — booked straight to the deposit
    account, never touching A/R. Mirrors create_invoice's line/discount/tax handling
    so the two agree exactly; the only structural differences are DepositToAccountRef
    and the /salesreceipt endpoint (no AllowOnline* — it is already paid).
    lines: [{name, amount(unit $), qty, description?, item_id?}]. Returns the
    SalesReceipt dict."""
    resolved = []
    for ln in lines:
        item_id = ln.get("item_id")
        if not item_id:
            unit = round(float(ln["amount"]), 2)
            item_id = find_or_create_item(ln.get("name", "RemedyMatch Product"), unit)["Id"]
        resolved.append({**ln, "item_id": item_id})
    if not bank_account_id:
        bank_account_id = _first_bank_account_id()
    if not bank_account_id:
        raise RuntimeError("no QBO bank account found for DepositToAccountRef")
    body = {"CustomerRef": {"value": customer["Id"]},
            "DepositToAccountRef": {"value": str(bank_account_id)},
            "Line": _build_invoice_lines(resolved, discount_cents)}
    if email_to:
        body["BillEmail"] = {"Address": email_to}
    if tax_cents and int(tax_cents) > 0:
        body["TxnTaxDetail"] = {"TotalTax": round(int(tax_cents) / 100.0, 2)}
        body["GlobalTaxCalculation"] = "TaxExcluded"
    return _post("/salesreceipt", body).get("SalesReceipt")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -- python3 -m pytest tests/test_qbo_sales_receipt.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/qbo_billing.py tests/test_qbo_sales_receipt.py
git commit -m "feat(qbo): add create_sales_receipt paid-only primitive"
```

---

## Task 2: Convert membership booking to a Sales Receipt

**Files:**
- Modify: `app.py` — `_book_membership_qbo` (~line 9079)
- Test: `tests/test_membership_qbo_sales_receipt.py` (create)

**Interfaces:**
- Consumes: `qbo_billing.create_sales_receipt` (Task 1), `qbo_billing.find_or_create_customer`.
- Produces: no new signature — `_book_membership_qbo(email, tier)` unchanged externally; internally books one Sales Receipt instead of invoice+payment.

- [ ] **Step 1: Write the failing test**

Create `tests/test_membership_qbo_sales_receipt.py`:

```python
import app
from dashboard import qbo_billing


def test_membership_books_sales_receipt_not_invoice(monkeypatch):
    calls = {"receipt": 0, "invoice": 0, "payment": 0}
    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt",
                        lambda *a, **k: calls.__setitem__("receipt", calls["receipt"] + 1)
                        or {"Id": "SR1"})
    monkeypatch.setattr(qbo_billing, "create_invoice",
                        lambda *a, **k: calls.__setitem__("invoice", calls["invoice"] + 1)
                        or {"Id": "INV1"})
    monkeypatch.setattr(qbo_billing, "record_payment",
                        lambda *a, **k: calls.__setitem__("payment", calls["payment"] + 1))

    app._book_membership_qbo("m@b.com", {"key": "month", "label": "1-Month",
                                         "price_cents": 9900})

    assert calls["receipt"] == 1
    assert calls["invoice"] == 0
    assert calls["payment"] == 0


def test_membership_qbo_failure_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("QBO down")
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", boom)
    # Must swallow and log, not raise.
    app._book_membership_qbo("m@b.com", {"key": "month", "label": "1-Month",
                                         "price_cents": 9900})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -- python3 -m pytest tests/test_membership_qbo_sales_receipt.py -v`
Expected: FAIL — `test_membership_books_sales_receipt_not_invoice` asserts `calls["invoice"] == 0` but current code calls `create_invoice` (so `invoice == 1`).

- [ ] **Step 3: Rewrite `_book_membership_qbo`**

Replace the body of `_book_membership_qbo` in `app.py` (~9079) with:

```python
def _book_membership_qbo(email, tier):
    """Record a paid one-time membership purchase (month / year_prepay) as a QBO
    SalesReceipt — paid-only, no A/R invoice. Best-effort — a QBO failure must never
    break the membership grant, which is already committed by the time this runs.
    Never raises."""
    try:
        from dashboard import qbo_billing as qb
        cust = qb.find_or_create_customer(email, "")
        qb.create_sales_receipt(
            cust,
            [{"name": tier["label"], "amount": tier["price_cents"] / 100.0, "qty": 1}],
            email_to=email)
    except Exception as e:
        print(f"[membership] QBO booking skipped for {email}/{tier.get('key')}: {e!r}",
              flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -- python3 -m pytest tests/test_membership_qbo_sales_receipt.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the membership fulfillment suite for regressions**

Run: `doppler run -- python3 -m pytest tests/test_membership_products_fulfill.py tests/test_membership_products_e2e.py -v`
Expected: PASS (no regressions from the booking swap)

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_membership_qbo_sales_receipt.py
git commit -m "feat(qbo): membership books a Sales Receipt, not an unpaid invoice"
```

---

## Task 3: Live-QBO diagnostic route for Sales Receipts

Proves QBO actually accepts the `/salesreceipt` payload (units can't — see `feedback_verify_against_live_api`). Mirrors the existing `qbo_test_invoice` route (`app.py:5046`), console-key gated.

**Files:**
- Modify: `app.py` — add route near `qbo_test_invoice` (~line 5046)
- Test: `tests/test_qbo_test_salesreceipt_route.py` (create)

**Interfaces:**
- Consumes: `_qbo_auth_ok()` (checks `X-Console-Key`/`?key` against `CONSOLE_SECRET`, `app.py:4955`), `qbo_billing.find_or_create_customer`, `qbo_billing.create_sales_receipt`.
- Produces: `POST /api/qbo/test-sales-receipt` (gated by `_qbo_auth_ok()`, same as `qbo_test_invoice`) → `{"ok": True, "id": ..., "doc_number": ..., "total": ...}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_qbo_test_salesreceipt_route.py`:

```python
import app


def test_route_rejects_without_console_key(monkeypatch):
    # With CONSOLE_SECRET set and no key sent, _qbo_auth_ok() → 401 (mirrors qbo_test_invoice).
    monkeypatch.setattr(app, "CONSOLE_SECRET", "secret-xyz")
    client = app.app.test_client()
    r = client.post("/api/qbo/test-sales-receipt")
    assert r.status_code == 401


def test_route_books_receipt_when_authorized(monkeypatch):
    monkeypatch.setattr(app, "_qbo_auth_ok", lambda: True)
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt",
                        lambda *a, **k: {"Id": "SR9", "DocNumber": "42", "TotalAmt": 1.0})
    client = app.app.test_client()
    r = client.post("/api/qbo/test-sales-receipt")
    assert r.status_code == 200
    assert r.get_json()["id"] == "SR9"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -- python3 -m pytest tests/test_qbo_test_salesreceipt_route.py -v`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Add the route**

Mirror `qbo_test_invoice` (`app.py:5046`) exactly — it gates on `_qbo_auth_ok()` alone (which itself checks the console key) and catches write errors as 500. Add directly beneath it:

```python
@app.route("/api/qbo/test-sales-receipt", methods=["POST"])
def qbo_test_sales_receipt():
    """Create ONE test Sales Receipt for a clearly-named test customer to verify the
    paid-only write layer against live QBO. Void/delete it afterward in QBO."""
    if not _qbo_auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        from dashboard import qbo_billing as qb
        cust = qb.find_or_create_customer("zztest+remedymatch@example.com", "ZZ Test DeleteMe")
        sr = qb.create_sales_receipt(
            cust,
            [{"name": "TEST RemedyMatch Product", "amount": 1.0, "qty": 1,
              "description": "TEST — verifying QBO paid-only write layer, please delete"}])
        return jsonify({"ok": True, "id": sr.get("Id"),
                        "doc_number": sr.get("DocNumber"), "total": sr.get("TotalAmt"),
                        "customer": cust.get("DisplayName")})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -- python3 -m pytest tests/test_qbo_test_salesreceipt_route.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_qbo_test_salesreceipt_route.py
git commit -m "feat(qbo): add console-gated /api/qbo/test-sales-receipt diagnostic"
```

- [ ] **Step 6: Live verification (manual, deployed env — NOT local pytest)**

After merge + deploy, Glen (or an authorized operator) hits the deployed route once:
`POST https://<deployed-host>/api/qbo/test-sales-receipt` with the console key.
Expected: `200 {"ok": true, "id": "...", "total": 1.0}` AND a $1 Sales Receipt visible in QBO with **no** open balance (confirm it did not create A/R). Delete/void the diagnostic receipt afterward. Local runs are expected to 400 on QBO auth (`reference_qbo_local_token_stale`) — that is not a failure of this task.

---

## Full-suite gate (end of plan)

- [ ] Run the money-path suites to confirm no baseline regressions (diff FAILED vs `main`, per `feedback_suite_green_not_task_green`):

Run: `doppler run -- python3 -m pytest tests/test_bos_finance.py tests/test_finance_record_payment.py tests/test_order_payments_routes.py tests/test_membership_products_fulfill.py -v`
Expected: PASS, or any failure also present on `main`.

---

## Staged follow-on plans (out of scope for THIS plan — each gets its own brainstorm + plan)

These flows use the QBO invoice Id as a load-bearing join key and/or mutate the invoice after creation, so they are not mechanical swaps. Each is its own plan:

- **Plan 2 — Begin/biofield checkout re-keying.** `app.py:~8319` returns `inv.Id` as `invoice_id`, uses it as the order `external_ref`, and threads it through Stripe metadata; the Stripe return handler (`app.py:~9358`) then applies a QBO Payment to that invoice. Convert to: create the local order first, key `external_ref` on the order id, drop `create_invoice` at checkout, and book a Sales Receipt in the Stripe-return handler (idempotent on a new `qbo_sales_receipt_id` order column). Requires the schema migration (`ALTER TABLE orders ADD COLUMN qbo_sales_receipt_id TEXT`) and a mutation-style guard test that **no** `/invoice` POST happens at checkout.
- **Plan 3 — Wholesale + dropship.** `dashboard/wholesale_checkout.py:54/87`, `dashboard/dropship_checkout.py:104/285`. Wholesale applies its credit discount *after* creating the invoice (`apply_invoice_discount`); a Sales Receipt is final, so the redeem must be resolved *before* booking and passed as `discount_cents`. Practitioner flows still collect via Stripe/alt-pay, then book a Sales Receipt.
- **Plan 4 — Retire `qbo_reconcile.py` + `record_payment`.** Once legacy open invoices have drained from QBO, remove the poller and the apply-payment-to-invoice path, and delete the dead `_QBO_PAYMENTS_ACTIVE` / `allow_online_pay` branches.
