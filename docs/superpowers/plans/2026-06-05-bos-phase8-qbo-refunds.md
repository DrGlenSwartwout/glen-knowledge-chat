# BOS Phase 8: QBO refunds (the first money_send action)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `finance.refund_order` — issue a QuickBooks RefundReceipt for a customer refund. This is the first real money-OUT action, so it runs at the `money_send` autonomy tier: the owner must confirm, and Shaira/unattended-Justus are queued for approval. It records the refund in QBO (money-out from a bank account); the actual card refund (Stripe) is a follow-on and manual Zelle/Wise refunds are sent by the operator.

**Architecture:** Two new helpers in `dashboard/qbo_billing.py` (`_first_bank_account_id`, `create_refund_receipt`) mirroring the existing write functions. One new action in `dashboard/finance.py` (`finance.refund_order`, `MONEY_SEND`, owner/ops, with a `confirm_summary`). A refund panel on the finance board UI (the board's `act()` already handles the `needs_confirmation` gate). The live RefundReceipt call is production-only (real QBO); the governance, registration, and executor are fully tested with mocks.

**Builds on:** the merged Business OS (Money & Finance module). New branch `sess/ec0e1f15` off main, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Governance (from rbac.py):** money_send -> OWNER = CONFIRM (AUTO only if `OWNER_MONEY_AUTO_THRESHOLD` > 0 and amount < threshold; default 0 = always confirm); OPS = CONFIRM; VA/AGENT = QUEUE. The action passes `amount` (dollars) so the threshold logic applies.

---

## File Structure

- `dashboard/qbo_billing.py` (modify): `_first_bank_account_id` + `create_refund_receipt`.
- `dashboard/finance.py` (modify): `finance.refund_order` action + `_refund_confirm_summary`.
- `tests/test_bos_finance.py` (modify): governance + executor tests (mocked QBO).
- `static/console-finance.html` (modify): a "Issue a refund" panel.

---

## Task 1: QBO RefundReceipt helpers (`dashboard/qbo_billing.py`)

**Files:**
- Modify: `dashboard/qbo_billing.py`

These wrap the existing `_post`/`_query` (real QBO), so they are not unit-tested in isolation; they are exercised via the finance action tests (mocked) and verified live by the operator's first confirmed refund.

- [ ] **Step 1: Add the helpers** (near `_first_income_account_id`, ~qbo_billing.py:72)

```python
def _first_bank_account_id():
    """Return the Id of the first QBO Bank account (DepositToAccountRef source)."""
    rs = _query("SELECT * FROM Account WHERE AccountType = 'Bank'")
    accts = rs.get("QueryResponse", {}).get("Account", [])
    return accts[0]["Id"] if accts else None


def create_refund_receipt(customer_id, amount, *, item_id=None,
                          bank_account_id=None, description="Refund"):
    """Issue a QBO RefundReceipt (records a money-out customer refund). `amount`
    is dollars (float). `bank_account_id` is the DepositToAccountRef -- the account
    the refund comes OUT of (defaults to the first Bank account). Returns the
    RefundReceipt dict. Mirrors the other write functions: a single _post call."""
    amt = round(float(amount), 2)
    if amt <= 0:
        raise ValueError("refund amount must be positive")
    if not item_id:
        item_id = find_or_create_item("Refund", amt)["Id"]
    if not bank_account_id:
        bank_account_id = _first_bank_account_id()
    if not bank_account_id:
        raise RuntimeError("no QBO bank account found for DepositToAccountRef")
    body = {
        "CustomerRef": {"value": str(customer_id)},
        "DepositToAccountRef": {"value": str(bank_account_id)},
        "Line": [{
            "DetailType": "SalesItemLineDetail",
            "Amount": amt,
            "Description": description,
            "SalesItemLineDetail": {
                "ItemRef": {"value": str(item_id)},
                "Qty": 1,
                "UnitPrice": amt,
            },
        }],
    }
    return _post("/refundreceipt", body).get("RefundReceipt")
```

- [ ] **Step 2: Compile**

Run: `python3 -m py_compile dashboard/qbo_billing.py`
Expected: OK.

- [ ] **Step 3: Commit**

```bash
git add dashboard/qbo_billing.py
git commit -m "feat(bos): QBO RefundReceipt helper (create_refund_receipt + first bank account)"
```

---

## Task 2: The refund action (`dashboard/finance.py`)

**Files:**
- Modify: `dashboard/finance.py`
- Test: `tests/test_bos_finance.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bos_finance.py`)

```python
def test_refund_action_registered():
    from dashboard import finance as F, actions as A  # noqa: F401
    a = A.get_action("finance.refund_order")
    assert a is not None
    assert a.module == "money"
    assert a.risk_tier == A.MONEY_SEND
    assert a.permission == ("owner", "ops")
    assert a.confirm_summary is not None


def test_refund_owner_needs_confirmation_no_qbo_call(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB
    called = {"refund": 0}
    monkeypatch.setattr(QB, "create_refund_receipt",
                        lambda *a, **k: called.__setitem__("refund", called["refund"] + 1) or {"Id": "1"})
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 80},
                            R.Actor(role=R.OWNER))
    assert res["status"] == "needs_confirmation"
    assert "80" in res["summary"]
    assert called["refund"] == 0  # nothing happened without confirmation


def test_refund_va_queues_no_qbo_call(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB
    called = {"refund": 0}
    monkeypatch.setattr(QB, "create_refund_receipt",
                        lambda *a, **k: called.__setitem__("refund", called["refund"] + 1) or {"Id": "1"})
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 80}, R.Actor(role=R.VA))
    assert res["status"] == "queued"
    assert called["refund"] == 0
    assert E.get_event(cx, res["event_id"])["status"] == "pending_approval"


def test_refund_executes_when_confirmed(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB
    monkeypatch.setattr(QB, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "C7"}, "DocNumber": "1009"})
    captured = {}
    def _fake_refund(customer_id, amount, **k):
        captured.update({"customer_id": customer_id, "amount": amount})
        return {"Id": "RR1", "DocNumber": "RR-1"}
    monkeypatch.setattr(QB, "create_refund_receipt", _fake_refund)
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 80, "reason": "duplicate"},
                            R.Actor(role=R.OWNER), confirmed=True)
    assert res["status"] == "done"
    assert captured == {"customer_id": "C7", "amount": 80.0}
    assert "80" in res["result"]["message"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_finance.py -k refund -q`
Expected: FAIL (`finance.refund_order` not registered).

- [ ] **Step 3: Append the action to `dashboard/finance.py`**

Add `MONEY_SEND` to the actions import at the top of the QBO-backed section (it currently imports `action, LOW_WRITE, IRREVERSIBLE`):

```python
from dashboard.actions import action, LOW_WRITE, IRREVERSIBLE, MONEY_SEND
```

Then add the action (next to `finance.void_invoice`):

```python
def _refund_confirm_summary(params):
    amt = params.get("amount", "?")
    target = params.get("invoice_id") or f"order #{params.get('order_id', '?')}"
    return (f"Issue a ${amt} refund against invoice {target}. This records a money-out "
            f"refund in QuickBooks (you still send the actual money for Zelle/Wise). Confirm?")


def _refund_order_exec(params, ctx):
    from dashboard import qbo_billing as qb
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    try:
        amount = float(params["amount"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("a positive amount (dollars) is required")
    if amount <= 0:
        raise ValueError("a positive amount is required")
    invoice_id = params.get("invoice_id")
    if not invoice_id and params.get("order_id"):
        from dashboard.orders import get_order
        order = get_order(cx, int(params["order_id"]))
        if not order:
            raise ValueError(f"order #{params['order_id']} not found")
        invoice_id = order.get("external_ref")
    if not invoice_id:
        raise ValueError("invoice_id or order_id required")
    inv = qb.get_invoice(str(invoice_id))
    if not inv:
        raise ValueError(f"invoice {invoice_id} not found")
    customer_id = (inv.get("CustomerRef") or {}).get("value")
    if not customer_id:
        raise ValueError("invoice has no customer")
    description = params.get("reason") or f"Refund for invoice {invoice_id}"
    receipt = qb.create_refund_receipt(customer_id, amount, description=description)
    _cache.clear()
    return {"refund_receipt_id": receipt.get("Id"), "customer_id": customer_id,
            "amount": amount, "invoice_id": invoice_id,
            "message": f"Refund of ${amount:.2f} recorded in QuickBooks "
                       f"(RefundReceipt {receipt.get('DocNumber', receipt.get('Id'))})."}


action(key="finance.refund_order", module="money", title="Refund order",
       description="Record a customer refund in QuickBooks (money-out RefundReceipt).",
       risk_tier=MONEY_SEND, permission=(OWNER, OPS),
       confirm_summary=_refund_confirm_summary)(_refund_order_exec)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_finance.py -q`
Expected: all finance tests pass (the 4 new refund tests + existing).

- [ ] **Step 5: Commit**

```bash
git add dashboard/finance.py tests/test_bos_finance.py
git commit -m "feat(bos): finance.refund_order (money_send, owner-confirm + va-queue governed)"
```

---

## Task 3: Refund panel on the finance board (`static/console-finance.html`)

**Files:**
- Modify: `static/console-finance.html`

The board's existing `act(id, key, params)` already handles `needs_confirmation` (browser confirm -> re-dispatch with `confirmed:true`), so the refund just needs a small input panel that calls it.

- [ ] **Step 1: Add a refund panel** near the top of the finance board body (after the summary strip, before the AR list). Add the markup:

```html
  <div class="panel" style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin:16px 0">
    <h2 style="font-family:'Raleway',sans-serif;font-size:15px;margin-bottom:10px">Issue a refund</h2>
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">
      <input id="r-invoice" placeholder="QBO invoice id" style="flex:1;min-width:140px;padding:8px 10px;background:var(--bg);color:var(--cream);border:1px solid var(--border);border-radius:7px" />
      <input id="r-amount" type="number" step="0.01" placeholder="amount $" style="width:120px;padding:8px 10px;background:var(--bg);color:var(--cream);border:1px solid var(--border);border-radius:7px" />
      <input id="r-reason" placeholder="reason (optional)" style="flex:1;min-width:140px;padding:8px 10px;background:var(--bg);color:var(--cream);border:1px solid var(--border);border-radius:7px" />
      <button onclick="doRefund()" style="border:1px solid var(--red);background:transparent;color:var(--red);border-radius:7px;padding:8px 14px;cursor:pointer">Refund</button>
    </div>
    <div id="r-result" style="margin-top:8px;font-size:13px;color:var(--muted)"></div>
  </div>
```

(Adapt the surrounding markup/IDs to the actual structure of the file; if the file uses CSS classes for `.panel`/`input`/`button`, reuse them instead of inline styles. Keep it consistent with the file's existing style.)

- [ ] **Step 2: Add the `doRefund()` function** to the board's `<script>` (it reuses the existing `act()` which handles the confirm gate):

```javascript
  async function doRefund(){
    var inv = (document.getElementById('r-invoice').value || '').trim();
    var amt = parseFloat(document.getElementById('r-amount').value);
    var reason = (document.getElementById('r-reason').value || '').trim();
    var out = document.getElementById('r-result');
    if (!inv || !(amt > 0)){ out.textContent = 'Enter an invoice id and a positive amount.'; return; }
    out.textContent = 'Working...';
    var params = {invoice_id: inv, amount: amt};
    if (reason) params.reason = reason;
    // /api/action returns needs_confirmation for the money_send tier; confirm then re-send.
    var res = await fetch('/api/action/finance.refund_order', {method:'POST', headers:hdr(), body:JSON.stringify(params)});
    var body = await res.json();
    if (body.status === 'needs_confirmation'){
      if (!confirm(body.summary || 'Confirm this refund?')){ out.textContent = 'Cancelled.'; return; }
      params.confirmed = true;
      body = await (await fetch('/api/action/finance.refund_order', {method:'POST', headers:hdr(), body:JSON.stringify(params)})).json();
    }
    out.textContent = body.status === 'done'
      ? ((body.result && body.result.message) || 'Refund recorded.')
      : ('Error: ' + (body.error || body.status));
    load();
  }
```

(If the finance board uses a shared `hdr()` + `load()`, reuse them; the plan assumes the board already has the console-key header helper and a refresh function, consistent with `console-orders.html`.)

- [ ] **Step 3: Verify it parses**

Run: `python3 -c "import html.parser; html.parser.HTMLParser().feed(open('static/console-finance.html').read()); print('parsed OK')"`
Confirm: no em dashes; no ALL CAPS shouting.

- [ ] **Step 4: Commit**

```bash
git add static/console-finance.html
git commit -m "feat(bos): refund panel on the finance board (money_send confirm gate)"
```

---

## Task 4: Verify under doppler

- [ ] **Step 1: Registration + governance check (no live refund issued)**

```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app, sqlite3
from dashboard import actions as A, dispatch as D, events as E, rbac as R
a = A.get_action("finance.refund_order")
assert a is not None and a.risk_tier == A.MONEY_SEND and a.permission == ("owner","ops")
cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
E.init_event_tables(cx)
# OWNER -> needs confirmation (NO QBO call, no real refund)
r1 = D.dispatch_action(cx, "finance.refund_order", {"invoice_id":"X","amount":5}, R.Actor(role=R.OWNER))
assert r1["status"] == "needs_confirmation", r1
# VA -> queued (NO QBO call)
r2 = D.dispatch_action(cx, "finance.refund_order", {"invoice_id":"X","amount":5}, R.Actor(role=R.VA))
assert r2["status"] == "queued", r2
print("REFUND_GOVERNANCE_OK")
PY'
rm -rf /tmp/bostest
```
Expected: `REFUND_GOVERNANCE_OK` (proves the action is registered + the confirm/queue gates fire WITHOUT issuing a real refund).

Run: `python3 -m pytest tests/test_bos_finance.py tests/test_bos_spine.py -q` (green).
Run: `python3 -m py_compile app.py dashboard/finance.py dashboard/qbo_billing.py` (OK).

- [ ] **Step 2: Commit (if any wiring tweak)**

No new commit expected unless verification surfaced a fix.

---

## Self-Review

**Spec coverage:** `finance.refund_order` issues a QBO RefundReceipt; `money_send` tier so the owner confirms and va/agent queue; `amount` drives the owner threshold; the finance board exposes it behind the confirm gate.

**Honest scope:** records the refund in QuickBooks (money-out from a bank account). It does NOT auto-refund a card (Stripe payment_intent is not stored) -- that is a follow-on; manual Zelle/Wise refunds are sent by the operator, this books them. The DepositToAccountRef defaults to the first QBO bank account (configurable later).

**Safety:** the live RefundReceipt call never runs in tests or the doppler check (all mocked / gated at needs_confirmation/queued). The first real refund is the operator's confirmed click in production.

**Placeholder scan:** none.

**Type consistency:** `create_refund_receipt(customer_id, amount, ...)`, the action key `finance.refund_order`, the `amount`-in-dollars contract, and the `needs_confirmation` UI handling match the dispatch/rbac/board contracts.
```
