# Money Inline Actions (Sub-project C2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two money inline actions — Record payment on a QBO receivable (partial/split supported) and Confirm a claimed payment on an order — where each item is shown.

**Architecture:** A new `finance.record_payment` BOS action (MONEY_SEND) over the existing `qbo_billing.record_payment` (extended with an optional method memo); a "Record payment" button on the Money board's Receivables rows; and a frontend-only "Confirm payment (method)" button on claimed orders that reuses the existing `orders.record_payment` with the stored method.

**Tech Stack:** Python (`dashboard/finance.py`, `dashboard/qbo_billing.py`), the BOS action layer, vanilla JS (`static/console-money.html`, `static/console-orders.html`), pytest (mocked QBO) + headless Playwright (mocked) render-verify.

## Global Constraints

- **Reuse, don't reinvent:** B's QBO write (`qbo_billing.record_payment`) and C's order action (`orders.record_payment`) already exist. C2 adds ONE action + TWO buttons + an optional `method` memo. No schema change.
- **Partial/split payments:** `qbo_billing.record_payment(customer_id, amount_cents, invoice_id)` posts a Payment of the GIVEN `amount_cents` and skips only when the invoice balance is already ≤ 0 — so recording a partial reduces the balance and leaves the invoice open for the next (split) payment. Record-payment records the prompted amount, not necessarily the full balance.
- `finance.record_payment` = `risk_tier=MONEY_SEND, permission=(OWNER, OPS)` with a `confirm_summary` (the dispatcher returns `needs_confirmation`; the Receivables `act()` already does the confirm re-POST + reloads). The Orders confirm-claimed reuses `orders.record_payment` (`LOW_WRITE`, `(OWNER,)`) which already defaults `amount_cents` to the order total — so the button passes only `{method: o.pay_method}`, guarded by a client `confirm()`.
- **No JSON-in-onclick of objects.** The Record-payment button passes the invoice id + balance as NUMBERS; the Orders button passes the id + the controlled short `pay_method` string. Prompts gather amount/method at click time.
- **Test env:** the Python tests are pure-module (`import dashboard.finance` / `dashboard.qbo_billing` — clients connect lazily on first call, not at import), so run with **plain `python3 -m pytest`**; if import unexpectedly needs secrets, fall back to `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/<scratch> python3 -m pytest`. Render-verify runs the app via the Doppler form (mkdir scratch first).

---

### Task 1: Backend — `finance.record_payment` action + method memo

**Files:**
- Modify: `dashboard/qbo_billing.py` (`record_payment` gains optional `method`).
- Modify: `dashboard/finance.py` (add `_record_payment_exec`, `_record_payment_confirm_summary`, register the action — next to the existing `finance.refund_order` registration ~line 234).
- Test: `tests/test_finance_record_payment.py` (create).

**Interfaces:**
- Consumes: `qbo_billing.get_invoice(invoice_id)` (returns the QBO invoice incl. `CustomerRef.value`), `qbo_billing.record_payment(customer_id, amount_cents, invoice_id, method=None)`, `qbo_billing._post(path, body)`, `dashboard.actions.action`/`get_action`/`MONEY_SEND`, `dashboard.rbac.OWNER`/`OPS`, `finance._cache`.
- Produces: `finance._record_payment_exec(params, ctx) -> {"ok": bool, "invoice_id"?, "amount"?, "error"?}`; the registered `finance.record_payment` action; `qbo_billing.record_payment(..., method=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_finance_record_payment.py`:

```python
"""C2: finance.record_payment executor (mocked QBO) + the optional method memo."""
import pytest
from dashboard import finance, qbo_billing
from dashboard.actions import get_action, MONEY_SEND
from dashboard.rbac import OWNER, OPS


def test_record_payment_exec_calls_qbo(monkeypatch):
    calls = {}
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "100"})
    def fake_rp(customer_id, amount_cents, invoice_id, method=None):
        calls.update(customer_id=customer_id, amount_cents=amount_cents, invoice_id=invoice_id, method=method)
        return {"Id": "P1"}
    monkeypatch.setattr(qbo_billing, "record_payment", fake_rp)
    res = finance._record_payment_exec({"invoice_id": "55", "amount": 50.0, "method": "Zelle"}, {})
    assert res["ok"] is True
    assert calls == {"customer_id": "42", "amount_cents": 5000, "invoice_id": "55", "method": "Zelle"}


def test_record_payment_exec_missing_invoice(monkeypatch):
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: None)
    res = finance._record_payment_exec({"invoice_id": "99", "amount": 10.0}, {})
    assert res.get("ok") is False


def test_record_payment_exec_rejects_nonpositive(monkeypatch):
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: {"CustomerRef": {"value": "1"}, "Balance": "10"})
    res = finance._record_payment_exec({"invoice_id": "1", "amount": 0}, {})
    assert res.get("ok") is False


def test_action_registered_metadata():
    a = get_action("finance.record_payment")
    assert a is not None and a.risk_tier == MONEY_SEND and a.permission == (OWNER, OPS)


def test_qbo_record_payment_method_memo(monkeypatch):
    cap = {}
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: {"Balance": "100"})
    def fake_post(path, body):
        cap["body"] = body
        return {"Payment": {"Id": "P1"}}
    monkeypatch.setattr(qbo_billing, "_post", fake_post)
    qbo_billing.record_payment("42", 5000, "55", method="Zelle")
    assert "Zelle" in cap["body"].get("PrivateNote", "")
    cap.clear()
    qbo_billing.record_payment("42", 5000, "55")
    assert "PrivateNote" not in cap["body"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_finance_record_payment.py -q`
Expected: FAIL — `finance` has no `_record_payment_exec`; `record_payment()` doesn't accept `method`; no `finance.record_payment` action.

- [ ] **Step 3: Extend `qbo_billing.record_payment` with `method`**

In `dashboard/qbo_billing.py`, change the signature and add the memo. The current body builds `body = {"CustomerRef":..., "TotalAmt": amt, "Line":[...]}` then `return _post("/payment", body).get("Payment")`. Update to:
```python
def record_payment(customer_id, amount_cents, invoice_id, method=None):
    """Record a QBO Payment applied to an invoice. Idempotent: skips when the invoice
    balance is already ≤ 0. `method` (optional) is recorded as a free-text memo (PrivateNote)
    so split payments by different methods are distinguishable."""
    inv = get_invoice(invoice_id)
    if not inv:
        raise RuntimeError(f"invoice {invoice_id} not found")
    try:
        balance = float(inv.get("Balance", inv.get("TotalAmt", 0)) or 0)
    except Exception:
        balance = 0.0
    if balance <= 0:
        return inv
    amt = round(int(amount_cents) / 100.0, 2)
    body = {
        "CustomerRef": {"value": str(customer_id)},
        "TotalAmt": amt,
        "Line": [{"Amount": amt,
                  "LinkedTxn": [{"TxnId": str(invoice_id), "TxnType": "Invoice"}]}],
    }
    if method:
        body["PrivateNote"] = "Console payment — method: " + str(method)
    return _post("/payment", body).get("Payment")
```

- [ ] **Step 4: Add the executor + action to `dashboard/finance.py`**

Near the existing `finance.refund_order` registration (~line 234), add (use whatever alias `finance.py` already imports `qbo_billing` under — confirm at the top of the file; the example uses `qbo_billing`):
```python
def _record_payment_confirm_summary(params):
    amt = float(params.get("amount") or 0)
    m = params.get("method")
    return "Record $%.2f against invoice %s%s?" % (amt, params.get("invoice_id"), (" via " + str(m)) if m else "")


def _record_payment_exec(params, ctx):
    invoice_id = params.get("invoice_id")
    try:
        amount = float(params.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0.0
    if not invoice_id or amount <= 0:
        return {"ok": False, "error": "invoice_id and a positive amount are required"}
    inv = qbo_billing.get_invoice(invoice_id)
    if not inv:
        return {"ok": False, "error": "invoice %s not found" % invoice_id}
    customer_id = (inv.get("CustomerRef") or {}).get("value")
    qbo_billing.record_payment(customer_id, round(amount * 100), invoice_id, method=params.get("method"))
    _cache.clear()
    return {"ok": True, "invoice_id": invoice_id, "amount": amount}


action(key="finance.record_payment", module="money", title="Record payment",
       description="Record a customer payment against a QuickBooks invoice (partial/split supported).",
       risk_tier=MONEY_SEND, permission=(OWNER, OPS),
       confirm_summary=_record_payment_confirm_summary)(_record_payment_exec)
```
(If `finance.py` imports qbo_billing as e.g. `from dashboard import qbo_billing as _qbo`, use `_qbo` throughout; match the file.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_finance_record_payment.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 6: Commit**
```bash
git add dashboard/qbo_billing.py dashboard/finance.py tests/test_finance_record_payment.py
git commit -m "feat(finance): finance.record_payment action (partial/split) + method memo on QBO Payment"
```

---

### Task 2: UI — Record-payment button (Receivables) + Confirm-claimed button (Orders)

**Files:**
- Modify: `static/console-money.html` (`MoneyReceivables`: `recordPayment` + the row button).
- Modify: `static/console-orders.html` (`cardHtml`: the claimed-order confirm button + `confirmClaimed`).
- Verify: headless Playwright (mocked).

**Interfaces:**
- Consumes: `finance.record_payment` (Task 1); `orders.record_payment` (existing); the `MoneyReceivables.act(id, key, params)` confirm-capable poster (console-money.html ~line 327); the orders `act(id, key, params)` (console-orders.html ~line 195).
- Produces: `MoneyReceivables.recordPayment(id, balance)`; `confirmClaimed(id, method)`.

- [ ] **Step 1: Receivables "Record payment" button + `recordPayment`**

In `static/console-money.html`, in the `MoneyReceivables` IIFE: add a `recordPayment` function and expose it (the IIFE returns an object with `load`/`act`/`doRefund` — add `recordPayment`). Then add a button to `rowHtml(r)` (~line 269) next to Send-reminder/Void, passing the invoice id + balance as numbers (no JSON-in-onclick):
```javascript
  // inside rowHtml, in the actions block:
  + '<button onclick="MoneyReceivables.recordPayment(' + iid + ',' + (Number(r.balance) || 0) + ')">Record payment</button>'
```
```javascript
  // add to the IIFE, and include in the returned object:
  function recordPayment(id, balance){
    var amtStr = prompt('Amount to record against invoice ' + id + ':', String(balance || ''));
    if (amtStr === null) return;
    var amount = parseFloat(amtStr);
    if (!(amount > 0)) { alert('Enter a positive amount.'); return; }
    var method = prompt('Method (Zelle / Wise / Card / Check / Cash / Other):', 'Zelle');
    if (method === null) return;
    act(id, 'finance.record_payment', { invoice_id: id, amount: amount, method: method });
  }
  // ... return { load: load, act: act, doRefund: doRefund, recordPayment: recordPayment };
```
`act()` already handles the MONEY_SEND `needs_confirmation` re-POST and reloads the AR list on success — so after a partial/split the row reappears with the reduced balance for the next payment. (Confirm `act(id, actionKey, params)`'s exact signature/confirm behavior at ~line 327 and conform; if `act` expects the action key embedded differently, match it.)

- [ ] **Step 2: Orders "Confirm payment" button + `confirmClaimed`**

In `static/console-orders.html`, in `cardHtml(o)` (~line 97), where the action set is built: when `o.pay_status === 'claimed'`, prepend a dedicated confirm button (the method is the controlled short string `o.pay_method`):
```javascript
  var claimedBtn = (o.pay_status === 'claimed')
    ? '<button onclick="confirmClaimed(' + Number(o.id) + ",'" + String(o.pay_method || '').replace(/'/g, '') + "'" + ')">Confirm payment' + (o.pay_method ? ' (' + esc(o.pay_method) + ')' : '') + '</button> '
    : '';
  // include claimedBtn at the front of `acts` for claimed orders
```
Add the handler near `recordPay` (~line 154):
```javascript
  function confirmClaimed(id, method){
    if (!confirm('Confirm ' + (method || 'this') + ' payment for order #' + id + '?')) return;
    act(id, 'orders.record_payment', { method: method });   // amount_cents defaults to the order total in the executor
  }
```
Leave the generic `recordPay()`/`payBtn()` path for non-claimed orders.

- [ ] **Step 3: Render-verify (headless, mocked)**

`mkdir -p /tmp/money2-test`; start the app on PORT=5097. Save `/tmp/money2-test/mv.py`:
```python
from playwright.sync_api import sync_playwright
import json
posts = []
def route_handler(route):
    u = route.request.url
    if "/api/finance/ar" in u:
        return route.fulfill(status=200, content_type="application/json",
            body=json.dumps({"ok": True, "data": [{"id": "55", "doc": "1001", "customer": "Jo", "email": "j@x.com", "total": 100, "balance": 100, "due_date": "2026-06-01", "days_overdue": 5}], "summary": {}}))
    if "/api/orders" in u and "/api/action" not in u:
        return route.fulfill(status=200, content_type="application/json",
            body=json.dumps({"ok": True, "data": [{"id": 7, "status": "confirmed", "pay_status": "claimed", "pay_method": "zelle", "total_cents": 5000, "email": "j@x.com", "items_json": "[]", "name": "Jo"}]}))
    if "/api/action/finance.record_payment" in u:
        posts.append(("finance", route.request.post_data)); return route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "done", "result": {"ok": True}}))
    if "/api/action/orders.record_payment" in u:
        posts.append(("orders", route.request.post_data)); return route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "done", "result": {"ok": True}}))
    return route.continue_()
with sync_playwright() as p:
    b = p.chromium.launch(); errs = []
    # --- Receivables Record-payment ---
    pg = b.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append("CJS:"+m.text) if (m.type=="error" and "Failed to load resource" not in m.text) else None)
    pg.route("**/api/**", route_handler)
    answers = iter(["50", "Zelle"])              # amount, method prompts
    pg.on("dialog", lambda d: d.accept(next(answers, "")) if d.type == "prompt" else d.accept())
    pg.goto("http://127.0.0.1:5097/console/money?key=test-secret#receivables", wait_until="networkidle"); pg.wait_for_timeout(1200)
    has_rp = pg.evaluate("()=>[...document.querySelectorAll('#panel-receivables button')].some(b=>/Record payment/.test(b.textContent))")
    pg.evaluate("()=>{var b=[...document.querySelectorAll('#panel-receivables button')].find(x=>/Record payment/.test(x.textContent)); if(b) b.click();}")
    pg.wait_for_timeout(800)
    pg.close()
    # --- Orders Confirm-claimed ---
    pg2 = b.new_page(); pg2.on("pageerror", lambda e: errs.append(str(e)))
    pg2.route("**/api/**", route_handler)
    pg2.on("dialog", lambda d: d.accept())
    pg2.goto("http://127.0.0.1:5097/console/orders?key=test-secret", wait_until="networkidle"); pg2.wait_for_timeout(1200)
    has_cc = pg2.evaluate("()=>[...document.querySelectorAll('button')].some(b=>/Confirm payment/.test(b.textContent))")
    pg2.evaluate("()=>{var b=[...document.querySelectorAll('button')].find(x=>/Confirm payment/.test(x.textContent)); if(b) b.click();}")
    pg2.wait_for_timeout(800)
    print("has Record-payment:", has_rp, " has Confirm-payment:", has_cc)
    fin = [json.loads(p) for k, p in posts if k == "finance"]
    ords = [json.loads(p) for k, p in posts if k == "orders"]
    print("finance posts:", fin); print("orders posts:", ords); print("errs:", errs or "NONE")
    assert has_rp and has_cc
    assert fin and fin[0].get("amount") == 50 and fin[0].get("method") == "Zelle" and fin[0].get("invoice_id") == "55"
    assert ords and ords[0].get("method") == "zelle"
    assert not errs, errs
    b.close(); print("OK")
```
Run `python3 /tmp/money2-test/mv.py` → `OK`, `errs: NONE`: the Receivables row has **Record payment**, clicking it (amount 50 / method Zelle) posts `finance.record_payment {invoice_id:'55', amount:50, method:'Zelle'}`; the claimed order has **Confirm payment (zelle)**, clicking it posts `orders.record_payment {method:'zelle'}` (no amount/method prompts). Adapt the mock `/api/finance/ar` + `/api/orders` envelopes to what the real endpoints return (read them first). Kill the server.

- [ ] **Step 4: Commit**
```bash
git add static/console-money.html static/console-orders.html
git commit -m "feat(console): Record-payment on receivables + Confirm-claimed on orders"
```

---

## Verification (whole sub-project)

- `python3 -m pytest tests/test_finance_record_payment.py -q` → 5 pass (executor calls QBO with right customer/cents/method; missing invoice + non-positive amount → error; method memo set/omitted; action MONEY_SEND/(OWNER,OPS)).
- Render-verify: Record-payment button posts `finance.record_payment` with `{invoice_id, amount, method}` through the confirm-capable `act()`; the claimed-order Confirm button posts `orders.record_payment {method}` with no re-prompt; zero JS errors.
- No schema change; refund/void/reminder + the orders fulfillment flow untouched; the failed-charge retry is not implemented (deferred).

## Self-Review Notes

- **Spec coverage:** `finance.record_payment` action + executor (Task 1) ✓; method memo on `qbo_billing.record_payment` (Task 1) ✓; partial/split = balance-driven record (Global Constraints + the executor records the prompted amount) ✓; MONEY_SEND/(OWNER,OPS)+confirm (Task 1 + metadata test) ✓; Receivables button + amount/method prompts (Task 2) ✓; Orders confirm-claimed reusing `orders.record_payment` with stored method, no re-prompt (Task 2) ✓; no JSON-in-onclick of objects (numbers + controlled method string) ✓; retry deferred (not implemented) ✓.
- **Type consistency:** `finance._record_payment_exec(params, ctx)`; `qbo_billing.record_payment(customer_id, amount_cents, invoice_id, method=None)`; JS `MoneyReceivables.recordPayment(id, balance)` / `confirmClaimed(id, method)` — consistent across tasks.
- **YAGNI:** reuses `qbo_billing.record_payment`, `orders.record_payment`, and the existing `act()` confirm flow; no new endpoint, no schema change.
