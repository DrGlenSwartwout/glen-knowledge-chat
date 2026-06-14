# Reorder Checkout on the Pricing Engine + Shipping — Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `dashboard/pricing.compute()` (Plan 1) the live pricer for the reorder cart — volume curve + floors + coupon + GET — and charge actual USA shipping, **behind a feature flag** so it ships dark and Glen flips it on.

**Architecture:** One checkout pricer `_price_cart()` in `app.py` turns a cart into (engine items → `compute()` result + QBO line items at list price + a fixed-amount discount + a shipping line). `/reorder/checkout` calls it under a `PRICING_ENGINE_CHECKOUT` flag; when off, the existing `_qty_unit_cents` path runs unchanged. The orders table gains `discount_cents`, `points_redeemed_cents`, `shipping_cents`. GET stays recorded-not-charged. US-only ship-to is enforced.

**Tech Stack:** Python 3.11, Flask, sqlite (`LOG_DB`), QBO (`dashboard/qbo_billing`), `dashboard/pricing` + `dashboard/tax` + `dashboard/shipping`, pytest.

**Run tests:** routes import `app` (builds clients at import), so:
`doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v` (ignore the 2 known pre-existing failures: `test_pf_playwright_fetch`, `test_bos_routes::test_home_page_served`). Pure-module tests (Tasks 1-3 helpers) can run with plain `~/.venvs/deploy-chat311/bin/python -m pytest`.

**Key facts (from the current-state map):**
- `/reorder/checkout` (app.py ~5656): prices via `_qty_unit_cents`, builds `lines=[{name, amount(unit $), qty, item_id, description}]`, `get = _tax.compute_get_cents(subtotal, channel="retail", ship_to_state=ship["state"])`, `inv = qb.create_invoice(cust, lines, allow_online_pay=True, email_to=email)` (no discount/tax passed), `_ingest_order(... channel="retail", get_cents=get)`, then `_stripe_checkout_url_for_reorder`.
- `qb.create_invoice(customer, lines, *, allow_online_pay=False, email_to=None, discount_cents=0, tax_cents=0)` — `discount_cents` adds a fixed-amount `DiscountLineDetail`; `tax_cents` stamps GET on the invoice. **We pass `discount_cents` but NOT `tax_cents`** (GET stays absorbed/recorded, current behavior).
- `_ingest_order(*, source, external_ref, email, name, phone, items, total_cents, address, channel, get_cents)` → `_bos_orders.upsert_order(...)`. No discount/points/shipping fields yet.
- `dashboard/shipping.quote({bottle_name: qty})` → `{box, shipping_cents,...}`; `pick_box`, `get_current_rates`. Not wired into checkout. Seeded rates S/M/L.
- Products have `name, price_cents, qbo_item_id, qty_pricing`. NO `months_per_unit`/`volume_eligible`/floor fields yet; Pure Powders have no `qty_pricing` and names contain "Pure Powder".

---

### Task 1: Order schema — discount / points / shipping columns

**Files:**
- Modify: `dashboard/orders.py` (the `upsert_order` + the `CREATE TABLE`/migration for the orders table)
- Test: `tests/test_orders_pricing_columns.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orders_pricing_columns.py
import sqlite3
from dashboard import orders as o

def _cx():
    cx = sqlite3.connect(":memory:")
    o.init_orders_schema(cx) if hasattr(o, "init_orders_schema") else o._init(cx)
    return cx

def test_upsert_records_discount_points_shipping():
    cx = _cx()
    o.upsert_order(cx, source="reorder", external_ref="INV1", email="a@x.com",
                   name="A", items=[{"name":"X","qty":1,"desc":"X"}], total_cents=5000,
                   address={"state":"CA"}, channel="retail", get_cents=0,
                   discount_cents=1500, points_redeemed_cents=300, shipping_cents=1265)
    row = cx.execute("SELECT discount_cents, points_redeemed_cents, shipping_cents "
                     "FROM orders WHERE external_ref='INV1'").fetchone()
    assert row == (1500, 300, 1265)

def test_upsert_defaults_new_columns_to_zero():
    cx = _cx()
    o.upsert_order(cx, source="reorder", external_ref="INV2", email="a@x.com",
                   items=[{"name":"X","qty":1,"desc":"X"}], total_cents=5000,
                   address={}, channel="retail", get_cents=0)
    row = cx.execute("SELECT discount_cents, points_redeemed_cents, shipping_cents "
                     "FROM orders WHERE external_ref='INV2'").fetchone()
    assert row == (0, 0, 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_orders_pricing_columns.py -v`
Expected: FAIL (no such column / unexpected kwarg).

- [ ] **Step 3: Implement**

In `dashboard/orders.py`: (a) find the orders `CREATE TABLE` and add three `INTEGER NOT NULL DEFAULT 0` columns `discount_cents`, `points_redeemed_cents`, `shipping_cents`; (b) add an idempotent migration for existing DBs — after the create, run for each new column:
```python
for col in ("discount_cents", "points_redeemed_cents", "shipping_cents"):
    try:
        cx.execute(f"ALTER TABLE orders ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass   # already exists
```
(c) extend `upsert_order` to accept `discount_cents=0, points_redeemed_cents=0, shipping_cents=0` and write them in the INSERT/UPSERT. Match the existing function's exact style (keyword-only args, the existing column list). Read the current `upsert_order` first and mirror its INSERT.

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_orders_pricing_columns.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dashboard/orders.py tests/test_orders_pricing_columns.py
git commit -m "feat(orders): record discount/points/shipping cents on orders"
```

---

### Task 2: `_engine_item` — derive months / eligibility / floors from a product

**Files:**
- Modify: `app.py` (add near `_get_product`/`_qty_unit_cents`, ~app.py:2312)
- Test: `tests/test_engine_item.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_item.py
import app as appmod

def test_engine_item_capsule_defaults():
    p = {"slug": "brain-boost", "name": "Brain Boost", "price_cents": 6997, "qty_pricing": True}
    it = appmod._engine_item(p, 3)
    assert it["unit_cents"] == 6997          # TRUE single-unit list
    assert it["months"] == 3                 # 1 month per unit * qty 3
    assert it["volume_eligible"] is True
    assert "sku_discount_floor_pct" not in it["product"]   # capsule uses global floors

def test_engine_item_pure_powder_excluded_and_floored():
    p = {"slug": "sumac-pure-powder", "name": "Sumac 50:1 Pure Powder", "price_cents": 3997}
    it = appmod._engine_item(p, 2)
    assert it["volume_eligible"] is False     # Pure Powders off the curve
    assert it["product"]["sku_discount_floor_pct"] == 0.75   # ~$30 on $40
    assert it["product"]["sku_points_floor_pct"] == 0.75

def test_engine_item_explicit_fields_win():
    p = {"slug": "x", "name": "X", "price_cents": 7000,
         "volume_eligible": False, "months_per_unit": 6}
    it = appmod._engine_item(p, 2)
    assert it["months"] == 12                 # 6 * 2
    assert it["volume_eligible"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest tests/test_engine_item.py -v` (needs full env)
Expected: FAIL (`_engine_item` missing).

- [ ] **Step 3: Implement**

```python
# add near _get_product in app.py
def _is_pure_powder(p):
    return "pure powder" in (p.get("name") or "").lower() or "pure-powder" in (p.get("slug") or "")

def _engine_item(p, qty):
    """Build a dashboard.pricing.compute() item from a product dict + quantity.
    Derives months (months_per_unit*qty, default 1/unit), volume eligibility
    (all Functional Formulations; Pure Powders + info_only excluded), and the
    per-SKU floor override for Pure Powders (0.75 -> ~$30)."""
    qty = max(1, int(qty))
    months_per_unit = int(p.get("months_per_unit", 1))
    if "volume_eligible" in p:
        eligible = bool(p["volume_eligible"])
    else:
        eligible = not (_is_pure_powder(p) or p.get("info_only"))
    prod = dict(p)
    if _is_pure_powder(p) and "sku_discount_floor_pct" not in prod:
        prod["sku_discount_floor_pct"] = 0.75
        prod["sku_points_floor_pct"] = 0.75
    return {
        "slug": p.get("slug"), "name": p.get("name", p.get("slug")), "qty": qty,
        "product": prod, "unit_cents": int(p.get("price_cents") or 0),
        "months": months_per_unit * qty, "volume_eligible": eligible,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest tests/test_engine_item.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_engine_item.py
git commit -m "feat(checkout): _engine_item derives months/eligibility/floors from product"
```

---

### Task 3: `_price_cart` — engine pricing + shipping + US-only, returns QBO-ready parts

**Files:**
- Modify: `app.py`
- Test: `tests/test_price_cart.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_price_cart.py
import pytest, app as appmod

def _stub_products(monkeypatch):
    cat = {"brain-boost": {"slug":"brain-boost","name":"Brain Boost","price_cents":7000,
                           "qty_pricing":True,"qbo_item_id":"27"}}
    monkeypatch.setattr(appmod, "_get_product", lambda s: cat.get(s))

def test_price_cart_volume_and_shipping(monkeypatch):
    _stub_products(monkeypatch)
    # 6 months total -> 29% volume off each 7000 = 4970
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295, "box": "M"})
    out = appmod._price_cart([{"slug":"brain-boost","qty":6}], ship={"state":"CA","country":"US"})
    assert out["priced"]["lines"][0]["line_total_cents"] == 4970
    assert out["discount_cents"] == 6 * (7000 - 4970)         # engine discount, list - net
    assert out["shipping_cents"] == 2295
    # QBO lines carry LIST price (qty applied by QBO), discount is separate
    assert out["qbo_lines"][0]["amount"] == 70.0 and out["qbo_lines"][0]["qty"] == 6

def test_price_cart_rejects_non_us(monkeypatch):
    _stub_products(monkeypatch)
    with pytest.raises(appmod.CheckoutError):
        appmod._price_cart([{"slug":"brain-boost","qty":1}], ship={"state":"ON","country":"CA"})

def test_price_cart_skips_unknown(monkeypatch):
    _stub_products(monkeypatch)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 0})
    out = appmod._price_cart([{"slug":"nope","qty":1}], ship={"state":"CA","country":"US"})
    assert out["qbo_lines"] == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest tests/test_price_cart.py -v`
Expected: FAIL (`_price_cart` / `CheckoutError` / `_shipping` missing).

- [ ] **Step 3: Implement**

```python
# add to app.py (near the reorder checkout). Local imports per this codebase's convention.
from dashboard import shipping as _shipping   # module-level is fine for a leaf util; if it
                                              # imports heavy deps, move inside _price_cart.

class CheckoutError(Exception):
    """Raised for a checkout the customer must fix (e.g. non-US ship-to)."""

def _price_cart(cart, *, ship, coupon_pct=None, subscriber_tier_pct=None, channel="retail"):
    """Price a reorder/checkout cart through the pricing engine + shipping.
    Returns {priced, qbo_lines, discount_cents, points_redeemed_cents, shipping_cents,
    items_rec, subtotal_list_cents}. Raises CheckoutError for non-US ship-to."""
    from dashboard import pricing as _pricing, tax as _tax
    country = (ship.get("country") or "US").strip().upper()
    if country not in ("US", "USA", ""):
        raise CheckoutError("We ship to US addresses only — please use a US forwarding address.")
    settings = _pricing.load_settings(_PRICING_SETTINGS)
    items, qbo_lines, items_rec, box_counts, subtotal_list = [], [], [], {}, 0
    for c in (cart or []):
        p = _get_product((c.get("slug") or "").strip())
        if not p:
            continue
        qty = max(1, min(int(c.get("qty", 1) or 1), 99))
        it = _engine_item(p, qty)
        items.append(it)
        subtotal_list += it["unit_cents"] * qty
        qbo_lines.append({"name": p["name"], "amount": round(it["unit_cents"] / 100.0, 2),
                          "qty": qty, "item_id": p.get("qbo_item_id"), "description": p["name"]})
        items_rec.append({"name": p["name"], "qty": qty, "desc": p["name"]})
        box_counts[p["name"]] = box_counts.get(p["name"], 0) + qty
    priced = _pricing.compute(items, settings=settings, coupon_pct=coupon_pct,
                              subscriber_tier_pct=subscriber_tier_pct, channel=channel,
                              ship_to_state=ship.get("state", ""),
                              tax_fn=_tax.compute_get_cents)
    shipping_cents = int(_shipping.quote(box_counts).get("shipping_cents", 0)) if box_counts else 0
    return {
        "priced": priced, "qbo_lines": qbo_lines, "items_rec": items_rec,
        "subtotal_list_cents": subtotal_list,
        "discount_cents": priced["discount_cents"],
        "points_redeemed_cents": priced["points_redeemed_cents"],
        "shipping_cents": shipping_cents,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest tests/test_price_cart.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_price_cart.py
git commit -m "feat(checkout): _price_cart — engine pricing + shipping + US-only guard"
```

---

### Task 4: Wire `/reorder/checkout` behind `PRICING_ENGINE_CHECKOUT`

**Files:**
- Modify: `app.py` (the `/reorder/checkout` route ~5656)
- Test: `tests/test_reorder_checkout_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reorder_checkout_engine.py
import app as appmod

def _setup(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug":s,"name":"Brain Boost","price_cents":7000,"qty_pricing":True,"qbo_item_id":"27"} if s=="brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295})
    captured = {}
    def fake_invoice(cust, lines, **kw):
        captured["lines"] = lines; captured["kw"] = kw
        return {"Id": "INV9", "TotalAmt": 100.0}
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", fake_invoice)
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: captured.setdefault("order", kw))
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder", lambda *a, **k: "https://stripe/x")
    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    return captured

def test_reorder_checkout_uses_engine_discount(monkeypatch):
    captured = _setup(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/reorder/checkout", json={"items":[{"slug":"brain-boost","qty":6}],
                                          "address":{"state":"CA","country":"US","name":"A"}})
    assert r.status_code == 200
    # engine discount (6*(7000-4970)=12180) passed to QBO as discount_cents
    assert captured["kw"]["discount_cents"] == 12180
    assert captured["order"]["discount_cents"] == 12180
    assert captured["order"]["shipping_cents"] == 2295
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest tests/test_reorder_checkout_engine.py -v`
Expected: FAIL (flag path not implemented / discount_cents not passed).

- [ ] **Step 3: Implement**

Read the current `/reorder/checkout` body. Wrap the new path in the flag; keep the old path intact for `else`:
```python
if os.environ.get("PRICING_ENGINE_CHECKOUT", "").strip().lower() in ("1","true","yes","on"):
    try:
        pc = _price_cart(cart, ship=ship, coupon_pct=_active_coupon_pct())
    except CheckoutError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    cust = qb.find_or_create_customer(email, ship.get("name",""))
    inv = qb.create_invoice(cust, pc["qbo_lines"] + _shipping_line(pc["shipping_cents"]),
                            allow_online_pay=True, email_to=email,
                            discount_cents=pc["discount_cents"] + pc["points_redeemed_cents"])
    _ingest_order(source="reorder", external_ref=inv.get("Id"), email=email,
                  name=ship.get("name",""), items=pc["items_rec"],
                  total_cents=int(round(float(inv.get("TotalAmt") or 0)*100)),
                  address=ship, channel="retail",
                  get_cents=pc["priced"]["get_cents"],
                  discount_cents=pc["discount_cents"],
                  points_redeemed_cents=pc["points_redeemed_cents"],
                  shipping_cents=pc["shipping_cents"])
    out = {"invoice_id": inv.get("Id"), "doc_number": inv.get("DocNumber"),
           "total": inv.get("TotalAmt")}
    stripe_url = _stripe_checkout_url_for_reorder(out, email) if _STRIPE_ACTIVE else ""
    return jsonify({"ok": True, "stripe_url": stripe_url, **out})
# else: existing path unchanged
```
Add the helpers:
```python
def _shipping_line(shipping_cents):
    if not shipping_cents:
        return []
    return [{"name": "Shipping (USPS)", "amount": round(int(shipping_cents)/100.0, 2),
             "qty": 1, "description": "USPS shipping"}]

def _active_coupon_pct():
    # reorder one-time orders may honor the daily coupon; subscriber tier is N/A here.
    return _COUPONS.get("_discount_percent", 0) if get_today_coupon_code() else None
```

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest tests/test_reorder_checkout_engine.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reorder_checkout_engine.py
git commit -m "feat(checkout): reorder checkout prices via engine + shipping behind PRICING_ENGINE_CHECKOUT flag"
```

---

### Task 5: Full suite green + flag doc

**Files:**
- Create: `docs/checkout-pricing-engine.md`

- [ ] **Step 1: Run all new + the existing reorder tests**

Run: `... -m pytest tests/test_orders_pricing_columns.py tests/test_engine_item.py tests/test_price_cart.py tests/test_reorder_checkout_engine.py tests/test_reorder_cart.py -v`
Expected: PASS (old reorder tests still green under the flag-off default).

- [ ] **Step 2: Write the doc**

```markdown
# Checkout on the pricing engine
`/reorder/checkout` prices via `dashboard.pricing.compute()` ONLY when
`PRICING_ENGINE_CHECKOUT` is truthy (default off → legacy `_qty_unit_cents` path).
When on: list-price QBO lines + a fixed-amount discount line (engine discount + redeemed
points) + a USPS shipping line; GET stays recorded-not-charged; ship-to must be US.
Orders record discount_cents / points_redeemed_cents / shipping_cents.
To go live: set PRICING_ENGINE_CHECKOUT=true in Doppler remedy-match/prd + Render.
Begin-funnel checkout, points earning at payment-return, and the Products-console floor
UI are later plans.
```

- [ ] **Step 3: Commit**

```bash
git add docs/checkout-pricing-engine.md
git commit -m "docs(checkout): pricing-engine checkout flag + behavior"
```

---

## Self-review
- **Spec coverage:** engine live on reorder (Tasks 3-4), volume/floors via `_engine_item` + `compute` (Tasks 2-3), Pure Powder exclusion + $30 floor (Task 2), shipping always-charged + US-only (Tasks 3-4), GET recorded-not-charged preserved (Task 4 passes get_cents to ingest, not tax_cents to invoice), order records discount/points/shipping (Task 1). Flagged + reversible (Task 4).
- **Deferred to later plans:** begin-funnel `/begin/checkout` wiring (same `_price_cart`, next); **points EARNING** at payment-return + first-affiliate-order suppression (Plan 4 rewards); Products-console per-SKU floor + `months_per_unit`/`volume_eligible` edit UI; **subscriptions** (Plan 3); cash-out review. Until the console UI exists, `_engine_item` derives eligibility/floors from product name/flags, so no data migration is needed to ship.
- **Risk:** changes live prices → shipped behind `PRICING_ENGINE_CHECKOUT` (default off); legacy path untouched when off; existing `test_reorder_cart.py` must stay green.
- **Type consistency:** `_engine_item(p, qty) -> item dict`, `_price_cart(cart, *, ship, coupon_pct, subscriber_tier_pct, channel) -> dict`, `CheckoutError`, `_shipping_line(cents) -> list` used identically across Tasks 2-4.

## Next
Plan 3 — Subscriptions ("Subscribe & Grow"): `subscriptions` table, Stripe vault setup at reorder checkout (`setup_future_usage`), the daily scheduler (charge/skip/heads-up/dunning) pricing each cycle through `_price_cart(subscriber_tier_pct=tier)`, and the manage-plan portal.
