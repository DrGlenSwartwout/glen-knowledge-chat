# Points Loop — Earning + Redemption — Implementation Plan (Plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the customer points loop — earn 5% on **paid full-price** orders, and let customers **redeem** points at the reorder checkout (the engine already applies them to the floor). Idempotent, behind the existing `PRICING_ENGINE_CHECKOUT` flag (redemption only acts on the engine checkout path).

**Architecture:** A points-earn/redeem **settlement** hook in `/begin/checkout-return` (which already detects `payment_status=="paid"` and looks up the order): on a paid order, earn 5% if it was full-price, and deduct any points the order redeemed — both keyed by the invoice id so a double-hit can't double-credit. Redemption is wired into `/reorder/checkout` (validate against balance → `_price_cart(points_to_redeem_cents=...)` → record `points_redeemed_cents`). A `GET /api/points/balance` + a reorder-cart UI control surface the balance and "apply points."

**Tech Stack:** Python 3.11, Flask, sqlite, `dashboard/points`, `_price_cart`, pytest.

**Depends on (in main):** `dashboard/points` (`earn`, `redeem`, `balance`), the orders `discount_cents`/`points_redeemed_cents`/`shipping_cents` columns, `_price_cart` (takes `points_to_redeem_cents`), `_bos_orders.find_order_by_external_ref`, the `_PRICING_SETTINGS` config (`points_earn_pct`=0.05), the reorder magic-link auth (`_reorder_email_from_cookie`), `/begin/checkout-return` (the paid block ~app.py:2701).

**Run tests:** route tests → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path>`; pure → plain venv pytest. Ignore the 2 known pre-existing failures.

**Key facts:** `/begin/checkout-return` paid block records the QBO payment then `_o = _bos_orders.find_order_by_external_ref(_cxo, inv)`. The order row has `email, total_cents, discount_cents, points_redeemed_cents, shipping_cents`. `points.earn(cx, email, *, full_price_cents, earn_pct, order_ref)`, `points.redeem(cx, email, *, value_cents, order_ref)`, `points.balance(cx, email)`. Earn is "full-price only" → only when `discount_cents == 0`. Product spend = `total_cents - shipping_cents` (GET is absorbed, not in total). Subscription orders never hit this handler (scheduler path) and are discounted, so they correctly don't earn.

---

### Task 1: Idempotency helper + settlement hook (earn + redeem on paid)

**Files:**
- Modify: `dashboard/points.py` (add `has_entry`)
- Modify: `app.py` (`/begin/checkout-return` paid block)
- Test: `tests/test_points_settlement.py`

- [ ] **Step 1: Write the failing test for `has_entry`**

```python
# tests/test_points_settlement.py
import sqlite3
from dashboard import points

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    points.init_points_table(cx); return cx

def test_has_entry_detects_prior_earn():
    cx = _cx()
    assert points.has_entry(cx, order_ref="INV1", reason="earn") is False
    points.earn(cx, "a@x.com", full_price_cents=7000, earn_pct=0.05, order_ref="INV1")
    assert points.has_entry(cx, order_ref="INV1", reason="earn") is True
    assert points.has_entry(cx, order_ref="INV1", reason="redeem") is False
```

- [ ] **Step 2: Run → fail** (`has_entry` missing).

- [ ] **Step 3: Implement `has_entry`**

```python
# add to dashboard/points.py
def has_entry(cx, *, order_ref, reason):
    """True if a ledger row already exists for this order_ref + reason (idempotency guard)."""
    row = cx.execute("SELECT 1 FROM points_ledger WHERE order_ref=? AND reason=? LIMIT 1",
                     (order_ref, reason)).fetchone()
    return row is not None
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Write the failing test for the settlement helper**

```python
import app as appmod

def test_settle_points_earns_on_full_price(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    points.init_points_table(cx); cx.commit()
    order = {"email": "a@x.com", "total_cents": 7265, "shipping_cents": 1265,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_order_points(order, order_ref="INV9")
    # product spend = 7265-1265 = 6000; earn 5% = 300
    assert points.balance(cx, "a@x.com") == 300
    # idempotent: second call does not double-earn
    appmod._settle_order_points(order, order_ref="INV9")
    assert points.balance(cx, "a@x.com") == 300

def test_settle_points_no_earn_when_discounted(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    points.init_points_table(cx); cx.commit()
    order = {"email": "a@x.com", "total_cents": 6000, "shipping_cents": 0,
             "discount_cents": 700, "points_redeemed_cents": 0}
    appmod._settle_order_points(order, order_ref="INV10")
    assert points.balance(cx, "a@x.com") == 0      # discounted → full-price-only rule → no earn

def test_settle_points_redeems_used_points(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    points.init_points_table(cx)
    points.earn(cx, "a@x.com", full_price_cents=20000, earn_pct=0.05, order_ref="seed")  # 1000
    cx.commit()
    order = {"email": "a@x.com", "total_cents": 5800, "shipping_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 200}
    appmod._settle_order_points(order, order_ref="INV11")
    # redeemed 200 deducted; NOT a full-price earn (points_redeemed>0 means a discount was applied)
    assert points.balance(cx, "a@x.com") == 800
```

- [ ] **Step 6: Run → fail** (`_settle_order_points` missing).

- [ ] **Step 7: Implement `_settle_order_points` + call it from `/begin/checkout-return`**

```python
# add to app.py
def _settle_order_points(order, *, order_ref):
    """On a PAID order: deduct redeemed points, and earn 5% if it was a full-price order.
    Idempotent per order_ref. Best-effort — never raises into the return handler."""
    from dashboard import points as _points
    email = (order.get("email") or "").strip().lower()
    if not email:
        return
    earn_pct = float(_PRICING_SETTINGS.get("points_earn_pct", 0.05)) if isinstance(_PRICING_SETTINGS, dict) else 0.05
    redeemed = int(order.get("points_redeemed_cents") or 0)
    discount = int(order.get("discount_cents") or 0)
    product_cents = int(order.get("total_cents") or 0) - int(order.get("shipping_cents") or 0)
    with sqlite3.connect(LOG_DB) as cx:
        _points.init_points_table(cx)
        if redeemed > 0 and not _points.has_entry(cx, order_ref=order_ref, reason="redeem"):
            try:
                _points.redeem(cx, email, value_cents=redeemed, order_ref=order_ref)
            except ValueError:
                pass   # balance already spent elsewhere; don't block the order
        # Earn only on a full-price order (no discount AND no points used) — the "full-price only" rule.
        if discount == 0 and redeemed == 0 and product_cents > 0 \
                and not _points.has_entry(cx, order_ref=order_ref, reason="earn"):
            _points.earn(cx, email, full_price_cents=product_cents, earn_pct=earn_pct,
                         order_ref=order_ref)
```

In `/begin/checkout-return`, inside the `payment_status=="paid"` block, AFTER the existing order lookup (`_o = _bos_orders.find_order_by_external_ref(_cxo, inv)`), add (wrapped so it never breaks the redirect):
```python
                            if _o:
                                try:
                                    _settle_order_points(_o, order_ref=inv)
                                except Exception as _pe:
                                    print(f"[points] settle failed inv={inv}: {_pe!r}", flush=True)
```
(Use the order dict `_o` returned by `find_order_by_external_ref`; if it lacks the new columns, fetch them — confirm the row includes email/total_cents/discount_cents/points_redeemed_cents/shipping_cents.)

- [ ] **Step 8: Run → pass** (all settlement tests).
- [ ] **Step 9: Commit** — `feat(points): settle earn/redeem on paid orders, idempotent per invoice`

---

### Task 2: Points balance API

**Files:** Modify `app.py`; Test `tests/test_points_balance_api.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_points_balance_api.py
import sqlite3, app as appmod
from dashboard import points

def test_balance_requires_auth(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "")
    assert appmod.app.test_client().get("/api/points/balance").status_code == 401

def test_balance_returns_cents_and_dollars(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db"); monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); points.init_points_table(cx)
    points.earn(cx, "a@x.com", full_price_cents=20000, earn_pct=0.05, order_ref="s"); cx.commit()
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    r = appmod.app.test_client().get("/api/points/balance")
    assert r.status_code == 200
    b = r.get_json()
    assert b["balance_cents"] == 1000 and b["balance_dollars"] == "10.00"
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement**

```python
@app.route("/api/points/balance", methods=["GET"])
def api_points_balance():
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"error": "not signed in"}), 401
    from dashboard import points as _points
    with sqlite3.connect(LOG_DB) as cx:
        _points.init_points_table(cx)
        bal = _points.balance(cx, email)
    return jsonify({"balance_cents": bal, "balance_dollars": f"{bal/100:.2f}"})
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(points): GET /api/points/balance`

---

### Task 3: Redemption at the reorder checkout

**Files:** Modify `app.py` (`/reorder/checkout` engine path); Test `tests/test_reorder_redeem.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_reorder_redeem.py
import sqlite3, app as appmod
from dashboard import points

def _setup(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db"); monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); points.init_points_table(cx)
    points.earn(cx, "a@x.com", full_price_cents=40000, earn_pct=0.05, order_ref="s"); cx.commit()  # 2000
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug":s,"name":"Brain Boost","price_cents":7000,"qty_pricing":True,"qbo_item_id":"27"} if s=="brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 0})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id":"C1"})
    cap = {}
    monkeypatch.setattr(appmod.qb, "create_invoice",
        lambda *a, **k: cap.update(k) or {"Id":"INV","TotalAmt":68.0})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: cap.setdefault("order", kw))
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder", lambda *a, **k: "")
    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    return cap

def test_reorder_redeems_points(monkeypatch, tmp_path):
    cap = _setup(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    r = c.post("/reorder/checkout", json={"items":[{"slug":"brain-boost","qty":1}],
               "address":{"state":"CA","country":"US","name":"A"}, "points_to_redeem_cents": 200})
    assert r.status_code == 200
    # 1 bottle $70 full price, 200 pts → line 6800; order records 200 redeemed
    assert cap["order"]["points_redeemed_cents"] == 200

def test_reorder_caps_redemption_to_balance(monkeypatch, tmp_path):
    cap = _setup(monkeypatch, tmp_path)   # balance 2000
    c = appmod.app.test_client()
    r = c.post("/reorder/checkout", json={"items":[{"slug":"brain-boost","qty":1}],
               "address":{"state":"CA","country":"US","name":"A"}, "points_to_redeem_cents": 999999})
    assert r.status_code == 200
    assert cap["order"]["points_redeemed_cents"] <= 2000   # never more than they have
```

- [ ] **Step 2: Run → fail** (redemption not wired).
- [ ] **Step 3: Implement** — in the `/reorder/checkout` engine-flag path, read `points_to_redeem_cents` from the body, cap it at the caller's current `points.balance(email)`, pass it to `_price_cart(..., points_to_redeem_cents=capped)`. `_price_cart` already returns `points_redeemed_cents` (what the engine actually used, ≤ requested due to the floor); that flows to `discount_cents=pc["discount_cents"]+pc["points_redeemed_cents"]` on the invoice and to `_ingest_order(points_redeemed_cents=...)` (already wired in Plan 2). Do NOT deduct the ledger here — the `_settle_order_points` hook (Task 1) deducts on confirmed payment. (For the no-Stripe path where the order is immediately "paid", confirm settlement still runs or call it here — match the existing reorder behavior.)

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(points): redeem points at reorder checkout (capped to balance, settled on payment)`

---

### Task 4: Reorder cart UI — balance + apply points

**Files:** Modify `static/reorder.html`; Test: manual (static)

- [ ] **Step 1:** In `static/reorder.html`, after sign-in, fetch `GET /api/points/balance` and show "You have $X.XX in points." Add an optional "Apply points" input (dollars → cents) included as `points_to_redeem_cents` in the `/reorder/checkout` POST. Show the projected discount from the cart total. Keep styling consistent.
- [ ] **Step 2:** Commit — `feat(points): show balance + apply-points control in the reorder cart`

---

### Task 5: Full suite green + doc

**Files:** Create `docs/points.md`

- [ ] **Step 1:** Run all Plan 1/2/3/4 point + checkout suites; green.
- [ ] **Step 2:** Write `docs/points.md`: earn 5% on **paid full-price** orders (no discount, no points used), settled idempotently per invoice at `/begin/checkout-return`; redeem at the reorder checkout capped to balance + the price floor (43%); balance at `GET /api/points/balance`. Subscriptions don't earn (discounted). **Affiliate first-order suppression + referral crediting + cash-out are Plan 5.**
- [ ] **Step 3:** Commit.

---

## Self-review
- **Spec coverage:** points earn 5% full-price-only (Task 1); redeem at checkout, floor-bounded + balance-capped (Task 3); idempotent settlement on confirmed payment (Task 1); balance surfaced (Tasks 2,4). GET excluded from the earn base (Task 1 product_cents = total − shipping).
- **Deferred to Plan 5:** affiliate-acquired first-order suppression (needs attribution), referral crediting (referrer points/cash by tier), tiers from GHL tags, cash-out review threshold + payout task + W-9.
- **Risk:** redemption changes the charged total — only on the `PRICING_ENGINE_CHECKOUT` path (already flag-gated). Earning is additive (a ledger most surfaces don't read yet). Idempotency prevents double-credit on a re-hit checkout-return.
- **Type consistency:** `points.has_entry(cx, *, order_ref, reason)->bool`, `_settle_order_points(order, *, order_ref)`, `/api/points/balance`, `points_to_redeem_cents` body key used identically across tasks.

## Next
Plan 5 — Rewards tiers + referral attribution + cash-out: read tier from People-hub/GHL tags; on a referred buyer's purchase credit the referrer (points for client-affiliate/doctor, cash-queue for pro influencer); suppress the buyer's points on an affiliate-acquired first order; threshold-triggered payout-review task (Business OS Tasks) + W-9 + the ~70%-of-face cash-out value.
