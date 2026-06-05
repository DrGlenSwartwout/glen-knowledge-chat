# BOS Phase 9: Stripe card refund

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `finance.refund_order` issue a real Stripe card refund (money actually returns to the customer's card) when a Stripe PaymentIntent is available, in addition to recording the QBO RefundReceipt. Capture the PaymentIntent on the order at the Stripe return so it's automatic for wholesale card orders.

**Money-safety design:** when a Stripe refund applies, issue the **card refund FIRST**; only if it succeeds do we record the QBO RefundReceipt (so the books never claim a refund that did not actually happen). If the card refund fails, the action fails and nothing is booked. Still `money_send` governed (owner confirms, va/agent queue) — unchanged from Phase 8.

**Architecture:** `stripe_pay.refund()` (POST /v1/refunds) + `get_session()` returns the PaymentIntent. The orders table gains a `stripe_payment_intent` column, captured in the existing Stripe return handler. `finance.refund_order` resolves the PaymentIntent (explicit param OR from the order), issues the card refund, then the QBO record. The refund panel gets an optional Stripe-id field.

**Builds on:** the merged Business OS (QBO refunds from Phase 8). New branch `sess/ec0e1f15` off main, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Context:** Stripe is wholesale-only and gated by `STRIPE_ACTIVE` (currently off); the funnel uses QBO online-pay, not Stripe. So card refunds apply to wholesale card orders once Stripe is active. The PaymentIntent is currently discarded by `get_session` and never stored.

---

## File Structure

- `dashboard/stripe_pay.py` (modify): `refund()` + `get_session` returns `payment_intent`.
- `dashboard/orders.py` (modify): `stripe_payment_intent` column + `set_order_stripe_pi` + `find_order_by_external_ref`.
- `dashboard/finance.py` (modify): `finance.refund_order` issues the Stripe refund when a PaymentIntent is available.
- `tests/test_bos_*` (modify): stripe_pay, orders, and finance tests (all mocked, no live Stripe/QBO).
- `app.py` (modify): capture the PaymentIntent in `practitioner_checkout_return`; add the Stripe field to the refund panel.

---

## Task 1: Stripe refund + payment_intent (`dashboard/stripe_pay.py`)

**Files:**
- Modify: `dashboard/stripe_pay.py`
- Test: `tests/test_bos_stripe.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bos_stripe.py`:

```python
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


class _Resp:
    def __init__(self, j): self._j = j
    def raise_for_status(self): pass
    def json(self): return self._j


def test_refund_full_and_partial(monkeypatch):
    from dashboard import stripe_pay as S
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    captured = {}
    def _post(url, data=None, auth=None, timeout=None):
        captured["url"] = url; captured["data"] = data
        return _Resp({"id": "re_1", "status": "succeeded", "amount": data.get("amount", 0)})
    monkeypatch.setattr(S.requests, "post", _post)
    # full refund: no amount
    r = S.refund("pi_123")
    assert captured["url"].endswith("/v1/refunds")
    assert captured["data"]["payment_intent"] == "pi_123" and "amount" not in captured["data"]
    assert r["status"] == "succeeded"
    # partial refund: amount in cents
    S.refund("pi_123", amount_cents=2500)
    assert captured["data"]["amount"] == 2500


def test_get_session_includes_payment_intent(monkeypatch):
    from dashboard import stripe_pay as S
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setattr(S.requests, "get",
                        lambda url, auth=None, timeout=None: _Resp(
                            {"id": "cs_1", "payment_status": "paid", "amount_total": 7000,
                             "metadata": {"invoice_id": "9"}, "payment_intent": "pi_9"}))
    sess = S.get_session("cs_1")
    assert sess["payment_intent"] == "pi_9"
    assert sess["payment_status"] == "paid"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_stripe.py -q`
Expected: FAIL (`refund` not defined / `payment_intent` not in get_session result).

- [ ] **Step 3: Edit `dashboard/stripe_pay.py`**

Add `refund` (near `get_session`):

```python
def refund(payment_intent, amount_cents=None):
    """Issue a Stripe refund against a PaymentIntent. amount_cents=None = full
    refund. Returns {id, status, amount}. Raises on a Stripe error."""
    data = {"payment_intent": str(payment_intent)}
    if amount_cents is not None:
        data["amount"] = int(amount_cents)
    r = requests.post(f"{STRIPE_API}/refunds", data=data, auth=(_key(), ""), timeout=20)
    r.raise_for_status()
    j = r.json()
    return {"id": j.get("id"), "status": j.get("status"), "amount": j.get("amount")}
```

Extend `get_session`'s return dict (the final `return {...}`) to include the PaymentIntent:

```python
    return {"id": j.get("id"), "payment_status": j.get("payment_status"),
            "amount_total": j.get("amount_total"), "metadata": j.get("metadata") or {},
            "payment_intent": j.get("payment_intent")}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_stripe.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/stripe_pay.py tests/test_bos_stripe.py
git commit -m "feat(bos): stripe refund() + get_session returns payment_intent"
```

---

## Task 2: Capture PaymentIntent on the order (`dashboard/orders.py`)

**Files:**
- Modify: `dashboard/orders.py`
- Test: `tests/test_bos_orders.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_bos_orders.py`)

```python
def test_stripe_pi_column_and_lookup():
    from dashboard import orders as O
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="wholesale", external_ref="INV-77", email="w@x.com")
    assert O.set_order_stripe_pi(cx, oid, "pi_77") is True
    assert O.get_order(cx, oid)["stripe_payment_intent"] == "pi_77"
    found = O.find_order_by_external_ref(cx, "INV-77")
    assert found and found["id"] == oid
    assert O.find_order_by_external_ref(cx, "nope") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_orders.py -k stripe_pi -q`
Expected: FAIL.

- [ ] **Step 3: Edit `dashboard/orders.py`**

In `init_orders_table`, after the existing `label_url` ALTER, add a best-effort migration:

```python
    try:
        cx.execute("ALTER TABLE orders ADD COLUMN stripe_payment_intent TEXT")
    except Exception:
        pass
    cx.commit()
```

Add the setter + lookup (near `set_order_tracking`):

```python
def set_order_stripe_pi(cx, order_id, payment_intent):
    cur = cx.execute("UPDATE orders SET stripe_payment_intent=?, updated_at=? WHERE id=?",
                     (payment_intent, _now(), order_id))
    cx.commit()
    return cur.rowcount > 0


def find_order_by_external_ref(cx, external_ref):
    cur = cx.execute("SELECT * FROM orders WHERE external_ref=? ORDER BY id DESC LIMIT 1",
                     (str(external_ref),))
    return _row_to_dict(cur.fetchone())
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_orders.py -q`
Expected: all orders tests pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/orders.py tests/test_bos_orders.py
git commit -m "feat(bos): orders.stripe_payment_intent column + capture/lookup helpers"
```

---

## Task 3: Wire the card refund into `finance.refund_order` (`dashboard/finance.py`)

**Files:**
- Modify: `dashboard/finance.py`
- Test: `tests/test_bos_finance.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bos_finance.py`)

```python
def test_refund_issues_stripe_card_refund_first(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB, stripe_pay as SP
    order = {}
    monkeypatch.setattr(QB, "get_invoice", lambda iid: {"CustomerRef": {"value": "C1"}, "DocNumber": "1"})
    calls = []
    monkeypatch.setattr(SP, "refund", lambda pi, amount_cents=None: calls.append(("stripe", pi, amount_cents)) or {"id": "re_1", "status": "succeeded"})
    monkeypatch.setattr(QB, "create_refund_receipt", lambda cid, amt, **k: calls.append(("qbo", cid, amt)) or {"Id": "RR1", "DocNumber": "RR-1"})
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 40, "stripe_payment_intent": "pi_9"},
                            R.Actor(role=R.OWNER), confirmed=True)
    assert res["status"] == "done"
    # stripe refund runs BEFORE the qbo record, with cents
    assert calls[0] == ("stripe", "pi_9", 4000)
    assert calls[1][0] == "qbo"
    assert "card" in res["result"]["message"].lower()


def test_refund_qbo_only_without_payment_intent(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB, stripe_pay as SP
    monkeypatch.setattr(QB, "get_invoice", lambda iid: {"CustomerRef": {"value": "C1"}})
    monkeypatch.setattr(QB, "create_refund_receipt", lambda cid, amt, **k: {"Id": "RR2", "DocNumber": "RR-2"})
    def _no_stripe(*a, **k): raise AssertionError("stripe.refund must not be called")
    monkeypatch.setattr(SP, "refund", _no_stripe)
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx); 
    from dashboard import orders as O; O.init_orders_table(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV-NONE", "amount": 40},
                            R.Actor(role=R.OWNER), confirmed=True)
    assert res["status"] == "done"
    assert "quickbooks" in res["result"]["message"].lower()


def test_refund_card_failure_blocks_qbo(monkeypatch):
    import sqlite3
    from dashboard import finance as F, dispatch as D, events as E, rbac as R
    from dashboard import qbo_billing as QB, stripe_pay as SP
    monkeypatch.setattr(QB, "get_invoice", lambda iid: {"CustomerRef": {"value": "C1"}})
    monkeypatch.setattr(SP, "refund", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("card declined")))
    qbo_called = {"n": 0}
    monkeypatch.setattr(QB, "create_refund_receipt", lambda *a, **k: qbo_called.__setitem__("n", 1) or {"Id": "RR"})
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    res = D.dispatch_action(cx, "finance.refund_order",
                            {"invoice_id": "INV9", "amount": 40, "stripe_payment_intent": "pi_x"},
                            R.Actor(role=R.OWNER), confirmed=True)
    assert res["status"] == "failed"
    assert qbo_called["n"] == 0  # card failed -> nothing booked
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_finance.py -k "stripe or card or payment_intent" -q`
Expected: FAIL.

- [ ] **Step 3: Edit `_refund_order_exec` in `dashboard/finance.py`**

Replace the body's refund-issuing section (after `customer_id` is resolved, before `receipt = qb.create_refund_receipt(...)`) so it resolves a PaymentIntent and issues the card refund first:

```python
    description = params.get("reason") or f"Refund for invoice {invoice_id}"

    # Resolve a Stripe PaymentIntent: explicit param, else from the captured order.
    pi = (params.get("stripe_payment_intent") or "").strip()
    if not pi and cx is not None:
        try:
            from dashboard.orders import find_order_by_external_ref
            o = find_order_by_external_ref(cx, invoice_id)
            pi = (o or {}).get("stripe_payment_intent") or ""
        except Exception:
            pi = ""

    card_msg = ""
    if pi:
        # Card refund FIRST: only book the QBO refund if real money actually went back.
        from dashboard import stripe_pay
        sr = stripe_pay.refund(pi, int(round(amount * 100)))
        card_msg = f" to the card (Stripe {sr.get('id')})"

    receipt = qb.create_refund_receipt(customer_id, amount, description=description)
    _cache.clear()
    return {"refund_receipt_id": receipt.get("Id"), "customer_id": customer_id,
            "amount": amount, "invoice_id": invoice_id, "stripe_refund": bool(pi),
            "message": f"Refund of ${amount:.2f}{card_msg} recorded in QuickBooks "
                       f"(RefundReceipt {receipt.get('DocNumber', receipt.get('Id'))})."}
```

(Keep everything above `description = ...` — the amount validation, invoice_id/order_id resolution, `get_invoice`, `customer_id` — exactly as is.)

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_finance.py -q`
Expected: all finance tests pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/finance.py tests/test_bos_finance.py
git commit -m "feat(bos): finance.refund_order issues a real Stripe card refund (card first, then QBO)"
```

---

## Task 4: Capture on return + UI field (`app.py`, `static/console-finance.html`)

**Files:**
- Modify: `app.py` (the `practitioner_checkout_return` handler)
- Modify: `static/console-finance.html` (refund panel Stripe field)

- [ ] **Step 1: Capture the PaymentIntent on the order** in `practitioner_checkout_return` (app.py ~5381). After the `qb.record_payment(cid, ..., inv)` call succeeds, add:

```python
                        pi = sess.get("payment_intent")
                        if pi:
                            try:
                                _cxo = _sqlite3.connect(LOG_DB); _cxo.row_factory = _sqlite3.Row
                                _o = _bos_orders.find_order_by_external_ref(_cxo, inv)
                                if _o:
                                    _bos_orders.set_order_stripe_pi(_cxo, _o["id"], pi)
                                _cxo.close()
                            except Exception as _e:
                                print(f"[stripe-return] pi capture: {_e!r}", flush=True)
```

(`_sqlite3`, `LOG_DB`, `_bos_orders` are available from the BOS startup block. Place this inside the `if inv and cid:` block, after the `record_payment` try.)

- [ ] **Step 2: Add the optional Stripe field to the refund panel** in `static/console-finance.html`. Add an input next to the refund inputs:

```html
      <input id="r-stripe" placeholder="Stripe pi_ (optional, refunds the card)" style="flex:1;min-width:160px;padding:8px 10px;background:var(--bg);color:var(--cream);border:1px solid var(--border);border-radius:7px" />
```

And in `doRefund()`, include it in the params when present (before the fetch):

```javascript
    var spi = (document.getElementById('r-stripe').value || '').trim();
    if (spi) params.stripe_payment_intent = spi;
```

(Adapt IDs/styling to the file's real structure, consistent with the existing refund inputs.)

- [ ] **Step 3: Compile + verify under doppler**

Run: `python3 -m py_compile app.py dashboard/finance.py dashboard/stripe_pay.py dashboard/orders.py` (OK).
Run: `python3 -c "import html.parser; html.parser.HTMLParser().feed(open('static/console-finance.html').read()); print('parsed OK')"` (parsed OK).
Run:
```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app, sqlite3
from dashboard import orders as O, stripe_pay as SP
cx = sqlite3.connect(app.LOG_DB); cx.row_factory=sqlite3.Row
cols = [r[1] for r in cx.execute("PRAGMA table_info(orders)")]
assert "stripe_payment_intent" in cols, "column missing"
assert hasattr(SP, "refund"), "stripe refund missing"
print("STRIPE_REFUND_OK")
PY'
rm -rf /tmp/bostest
```
Expected: `STRIPE_REFUND_OK`.

Run: `python3 -m pytest tests/test_bos_stripe.py tests/test_bos_orders.py tests/test_bos_finance.py tests/test_bos_spine.py -q` (green).

- [ ] **Step 4: Commit**

```bash
git add app.py static/console-finance.html
git commit -m "feat(bos): capture Stripe payment_intent on order + refund-panel card field"
```

---

## Self-Review

**Spec coverage:** `finance.refund_order` issues a real Stripe card refund when a PaymentIntent is available (explicit or captured from the order), card-first then QBO; the PaymentIntent is captured at the Stripe return; the refund panel exposes the optional Stripe id.

**Money-safety:** card refund runs before the QBO record; a card-refund failure fails the whole action and books nothing. Governance is unchanged (money_send -> owner confirm, va/agent queue). The live Stripe + QBO calls are mocked in tests and never fire in the doppler check.

**Honest scope:** Stripe is wholesale-only and gated by `STRIPE_ACTIVE`; funnel orders use QBO online-pay / Zelle / Wise (no card to refund). So card refunds apply to wholesale card orders once Stripe is active. No-PaymentIntent orders refund QBO-only (the operator sends manual money).

**Placeholder scan:** none.

**Type consistency:** `stripe_pay.refund(payment_intent, amount_cents)`, `set_order_stripe_pi`, `find_order_by_external_ref`, the `stripe_payment_intent` param, and the amount dollars->cents conversion are consistent across Tasks 1-4.
```
