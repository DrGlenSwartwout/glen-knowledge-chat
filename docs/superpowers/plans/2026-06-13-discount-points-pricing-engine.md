# Discount & Points Pricing Engine — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one pure pricing function — the single source of truth for what a cart costs after the one allowed % discount, points, and the wholesale/points floors — plus a points ledger, so the cart UI, checkout, and the (Plan 2) subscription scheduler can never disagree on a price.

**Architecture:** A new `dashboard/pricing.py` (settings + floor math + `compute()`), a new `dashboard/points.py` (sqlite ledger), a `pricing-settings.json` config (global defaults, overridable like the existing `coupons.json`), and per-SKU floor fields read off the existing product dict. GET tax is injected (the existing `dashboard.tax.compute_get_cents`) so the engine is testable without env. A thin `/api/pricing/preview` endpoint proves it end-to-end. Live-checkout rewiring + subscriptions are Plan 2.

**Tech Stack:** Python 3.11, Flask, sqlite (`LOG_DB`), pytest. Config via the existing `_load_json(DATA_DIR / "...json", default)` pattern.

**Decisions (from the spec, 2026-06-13):** one % discount only (subscriber tier exclusive of coupons); discount floor = 57% of list (wholesale), points floor = 43% (both per-SKU overridable, console-editable); points earn 5% on full-price spend only; points = a price discount (reduces GET base). Floors are percent-of-list and cover all Functional Formulations™ + Pure Powders; high-cost SKUs get a per-SKU higher floor.

**Run tests with:** `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v` (ignore the 2 known pre-existing failures elsewhere: `test_pf_playwright_fetch`, `test_bos_routes::test_home_page_served`).

---

### Task 1: Pricing settings loader

**Files:**
- Create: `dashboard/pricing.py`
- Test: `tests/test_pricing_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pricing_engine.py
from dashboard import pricing

def test_defaults_present():
    s = pricing.load_settings({})
    assert s["discount_floor_pct"] == 0.57
    assert s["points_floor_pct"] == 0.43
    assert s["points_earn_pct"] == 0.05
    assert s["points_redeem_per_point_cents"] == 5
    assert s["subscribe_tiers"] == [5, 10, 15]
    assert s["cadences"] == [1, 2, 3]
    assert s["volume_anchors"] == [[1, 0], [3, 14], [6, 29], [12, 43]]

def test_overrides_merge_over_defaults():
    s = pricing.load_settings({"discount_floor_pct": 0.70})
    assert s["discount_floor_pct"] == 0.70   # overridden
    assert s["points_floor_pct"] == 0.43     # default retained
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_pricing_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.pricing'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/pricing.py
"""Single source of truth for cart pricing: one % discount, points, and the
wholesale (57%) / points (43%) floors. Pure + injectable for testing."""

DEFAULTS = {
    "discount_floor_pct": 0.57,           # all % discounts clamp up to list * this (= wholesale)
    "points_floor_pct": 0.43,             # points clamp up to list * this
    "points_earn_pct": 0.05,              # earn 5% of full-price spend, as redemption-value cents
    "points_redeem_per_point_cents": 5,   # 1 point = 5 cents (20 points = $1)
    "subscribe_tiers": [5, 10, 15],       # % by completed-order count (1st,2nd,3rd+)
    "cadences": [1, 2, 3],                # months
    # volume curve: [total_months, pct_off] knots, ascending; linear interp; flat beyond last
    "volume_anchors": [[1, 0], [3, 14], [6, 29], [12, 43]],
}

def load_settings(overrides):
    """DEFAULTS merged with a dict of overrides (e.g. from pricing-settings.json)."""
    s = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if v is not None:
            s[k] = v
    return s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_pricing_engine.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/pricing.py tests/test_pricing_engine.py
git commit -m "feat(pricing): settings loader with global defaults"
```

---

### Task 2: Per-SKU floor resolution

**Files:**
- Modify: `dashboard/pricing.py`
- Test: `tests/test_pricing_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_floor_uses_global_pct_when_no_override():
    s = pricing.load_settings({})
    p = {"slug": "neuro-mag", "price_cents": 7000}
    assert pricing.unit_floor_cents(p, 7000, s, "discount") == 3990   # round(7000*0.57)
    assert pricing.unit_floor_cents(p, 7000, s, "points") == 3010     # round(7000*0.43)

def test_floor_uses_per_sku_pct_override():
    s = pricing.load_settings({})
    p = {"slug": "costly", "price_cents": 9000,
         "sku_discount_floor_pct": 0.70, "sku_points_floor_pct": 0.60}
    assert pricing.unit_floor_cents(p, 9000, s, "discount") == 6300   # 9000*0.70
    assert pricing.unit_floor_cents(p, 9000, s, "points") == 5400     # 9000*0.60

def test_floor_uses_absolute_wholesale_override():
    s = pricing.load_settings({})
    # absolute wholesale wins for the discount floor; points floor = wholesale - allowance,
    # allowance defaults to list*(discount_pct - points_pct) = 7000*0.14 = 980
    p = {"slug": "fixed", "price_cents": 7000, "wholesale_cents": 4200}
    assert pricing.unit_floor_cents(p, 7000, s, "discount") == 4200
    assert pricing.unit_floor_cents(p, 7000, s, "points") == 3220     # 4200 - 980
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_pricing_engine.py -k floor -v`
Expected: FAIL with `AttributeError: module 'dashboard.pricing' has no attribute 'unit_floor_cents'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to dashboard/pricing.py
def unit_floor_cents(product, list_cents, settings, kind):
    """Per-unit floor in cents. kind in ('discount','points').
    Precedence: absolute wholesale_cents > per-SKU pct > global pct."""
    list_cents = int(list_cents)
    whole = product.get("wholesale_cents")
    if kind == "discount":
        if whole:
            return int(whole)
        pct = product.get("sku_discount_floor_pct", settings["discount_floor_pct"])
        return int(round(list_cents * pct))
    # points
    if whole:
        allowance = int(round(list_cents * (settings["discount_floor_pct"]
                                            - settings["points_floor_pct"])))
        return int(whole) - allowance
    pct = product.get("sku_points_floor_pct", settings["points_floor_pct"])
    return int(round(list_cents * pct))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_pricing_engine.py -k floor -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/pricing.py tests/test_pricing_engine.py
git commit -m "feat(pricing): per-SKU floor override resolution"
```

---

### Task 3: Apply the one % discount, clamped to the discount floor

**Files:**
- Modify: `dashboard/pricing.py`
- Test: `tests/test_pricing_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_discount_applied_above_floor():
    # 15% off 7000 = 5950, above the 3990 floor → 5950
    assert pricing.apply_discount(7000, 15, 3990) == 5950

def test_discount_clamped_to_floor():
    # 50% off 7000 = 3500, below the 3990 floor → clamp to 3990
    assert pricing.apply_discount(7000, 50, 3990) == 3990

def test_zero_discount_is_list():
    assert pricing.apply_discount(7000, 0, 3990) == 7000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_pricing_engine.py -k discount -v`
Expected: FAIL with `AttributeError: ... 'apply_discount'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to dashboard/pricing.py
def apply_discount(list_cents, pct, floor_cents):
    """Apply a single percentage discount, never below floor_cents."""
    discounted = int(round(int(list_cents) * (1 - (pct or 0) / 100.0)))
    return max(discounted, int(floor_cents))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_pricing_engine.py -k discount -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/pricing.py tests/test_pricing_engine.py
git commit -m "feat(pricing): single % discount clamped to floor"
```

---

### Task 4: Apply points, clamped to the points floor

**Files:**
- Modify: `dashboard/pricing.py`
- Test: `tests/test_pricing_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_points_reduce_above_floor():
    # price 5950, points 1000, floor 3010 → 4950, used 1000
    assert pricing.apply_points(5950, 1000, 3010) == (4950, 1000)

def test_points_clamped_at_floor_partial_use():
    # price 4000, want 2000 off, floor 3010 → only 990 usable → 3010, used 990
    assert pricing.apply_points(4000, 2000, 3010) == (3010, 990)

def test_points_none_requested():
    assert pricing.apply_points(5950, 0, 3010) == (5950, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_pricing_engine.py -k points -v`
Expected: FAIL with `AttributeError: ... 'apply_points'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to dashboard/pricing.py
def apply_points(price_cents, points_cents, floor_cents):
    """Subtract points (in redemption-value cents) but never below floor_cents.
    Returns (new_price_cents, points_actually_used_cents)."""
    price_cents = int(price_cents)
    reducible = max(0, price_cents - int(floor_cents))
    used = min(max(0, int(points_cents)), reducible)
    return price_cents - used, used
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_pricing_engine.py -k points -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/pricing.py tests/test_pricing_engine.py
git commit -m "feat(pricing): points redemption clamped to points floor"
```

---

### Task 5: Volume curve + `compute()` — price a whole cart (the public API)

**Files:**
- Modify: `dashboard/pricing.py`
- Test: `tests/test_pricing_engine.py`

- [ ] **Step 1: Write the failing test for `volume_pct`**

```python
def test_volume_pct_at_anchors():
    s = pricing.load_settings({})
    assert pricing.volume_pct(1, s) == 0
    assert pricing.volume_pct(3, s) == 14
    assert pricing.volume_pct(6, s) == 29
    assert pricing.volume_pct(12, s) == 43

def test_volume_pct_interpolates_and_caps():
    s = pricing.load_settings({})
    assert pricing.volume_pct(2, s) == 7            # halfway 0->14
    assert pricing.volume_pct(9, s) == 36           # halfway 29->43
    assert pricing.volume_pct(24, s) == 43          # flat beyond the last knot
    assert pricing.volume_pct(0, s) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_pricing_engine.py -k volume_pct -v`
Expected: FAIL with `AttributeError: ... 'volume_pct'`

- [ ] **Step 3: Implement `volume_pct`**

```python
# add to dashboard/pricing.py
def volume_pct(months, settings):
    """Percentage discount for total cart months, linear-interpolated through the
    console anchor table (ascending [months, pct_off] pairs); flat beyond the last knot."""
    anchors = settings["volume_anchors"]
    m = max(0, int(months or 0))
    if m <= anchors[0][0]:
        return float(anchors[0][1])
    for (m0, p0), (m1, p1) in zip(anchors, anchors[1:]):
        if m <= m1:
            return p0 + (p1 - p0) * (m - m0) / (m1 - m0)
    return float(anchors[-1][1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_pricing_engine.py -k volume_pct -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the failing test for `compute`**

```python
def _fake_tax(subtotal_cents, *, channel, ship_to_state, resale_ok=False):
    return int(round(subtotal_cents * 0.04)) if ship_to_state == "HI" else 0

def test_compute_one_line_subscriber_tier_and_points():
    s = pricing.load_settings({})
    items = [{"slug": "neuro-mag", "name": "Neuro Mag", "qty": 1,
              "product": {"slug": "neuro-mag", "price_cents": 7000},
              "unit_cents": 7000, "months": 1, "volume_eligible": True}]
    r = pricing.compute(items, settings=s, subscriber_tier_pct=15,
                        points_to_redeem_cents=1000, channel="retail",
                        ship_to_state="HI", tax_fn=_fake_tax)
    # M=1 -> volume 0%; best-of(0,15)=15%. 15% off 7000 = 5950; points 1000 -> 4950
    assert r["lines"][0]["line_total_cents"] == 4950
    assert r["discount_cents"] == 1050
    assert r["points_redeemed_cents"] == 1000
    assert r["get_cents"] == 198            # round(4950*0.04)

def test_compute_subscriber_tier_beats_coupon_no_stack():
    s = pricing.load_settings({})
    items = [{"slug": "x", "name": "X", "qty": 1,
              "product": {"slug": "x", "price_cents": 7000},
              "unit_cents": 7000, "months": 1, "volume_eligible": True}]
    r = pricing.compute(items, settings=s, subscriber_tier_pct=5, coupon_pct=40,
                        channel="retail", ship_to_state="HI", tax_fn=_fake_tax)
    assert r["lines"][0]["line_total_cents"] == 6650   # 5% (sub), coupon ignored

def test_compute_volume_mix_and_match_beats_subscriber():
    s = pricing.load_settings({})
    # two different SKUs, 3 months each = 6 total -> volume 29% beats 15% tier
    items = [
        {"slug": "a", "name": "A", "qty": 1, "product": {"slug": "a", "price_cents": 7000},
         "unit_cents": 7000, "months": 3, "volume_eligible": True},
        {"slug": "b", "name": "B", "qty": 1, "product": {"slug": "b", "price_cents": 7000},
         "unit_cents": 7000, "months": 3, "volume_eligible": True},
    ]
    r = pricing.compute(items, settings=s, subscriber_tier_pct=15,
                        channel="retail", ship_to_state="CA", tax_fn=_fake_tax)
    assert r["lines"][0]["line_total_cents"] == 4970   # 29% off 7000
    assert r["lines"][1]["line_total_cents"] == 4970

def test_compute_pure_powder_excluded_from_volume_floored_at_30():
    s = pricing.load_settings({})
    # Pure Powder: NOT volume_eligible (months ignored), per-SKU floor 75% of 4000 = 3000
    items = [{"slug": "pp", "name": "Pure Powder", "qty": 1,
              "product": {"slug": "pp", "price_cents": 4000,
                          "sku_discount_floor_pct": 0.75, "sku_points_floor_pct": 0.75},
              "unit_cents": 4000, "months": 12, "volume_eligible": False}]
    r = pricing.compute(items, settings=s, subscriber_tier_pct=15,
                        points_to_redeem_cents=1000, channel="retail",
                        ship_to_state="CA", tax_fn=_fake_tax)
    # volume 0 (excluded); 15% off 4000 = 3400 (floor 3000); points -> 3000 (only 400 used)
    assert r["lines"][0]["line_total_cents"] == 3000
    assert r["points_redeemed_cents"] == 400
```

- [ ] **Step 6: Run test to verify it fails**

Run: `... -m pytest tests/test_pricing_engine.py -k compute -v`
Expected: FAIL with `AttributeError: ... 'compute'`

- [ ] **Step 7: Implement `compute`**

```python
# add to dashboard/pricing.py
def compute(items, *, settings, subscriber_tier_pct=None, coupon_pct=None,
            points_to_redeem_cents=0, channel="retail", ship_to_state=None,
            resale_ok=False, tax_fn=None):
    """Price a cart. The single % discount per line = max(volume_pct, sub-or-coupon).
    Subscriber tier and coupon never stack (subscriber wins if present). Points apply
    after, then GET tax on the discounted subtotal. Base is the TRUE single-unit list, so
    floors always anchor to list.

    items: [{"slug","name","qty","product","unit_cents","months","volume_eligible"}]
    Returns a dict with per-line breakdown + order totals.
    """
    base_pct = subscriber_tier_pct if subscriber_tier_pct else (coupon_pct or 0)
    total_months = sum(int(it.get("months") or 0) for it in items if it.get("volume_eligible"))
    vpct = volume_pct(total_months, settings)
    points_left = max(0, int(points_to_redeem_cents or 0))
    lines, subtotal, total_discount, total_points = [], 0, 0, 0

    for it in items:
        p = it["product"]
        qty = int(it["qty"])
        unit_list = int(it["unit_cents"])
        line_list = unit_list * qty
        line_pct = max(vpct if it.get("volume_eligible") else 0, base_pct)
        disc_floor = unit_floor_cents(p, unit_list, settings, "discount") * qty
        pts_floor = unit_floor_cents(p, unit_list, settings, "points") * qty

        after_disc = apply_discount(line_list, line_pct, disc_floor)
        after_pts, used = apply_points(after_disc, points_left, pts_floor)
        points_left -= used

        lines.append({
            "slug": it["slug"], "name": it["name"], "qty": qty,
            "list_cents": line_list, "discount_cents": line_list - after_disc,
            "points_cents": used, "line_total_cents": after_pts, "pct_applied": line_pct,
        })
        subtotal += after_pts
        total_discount += (line_list - after_disc)
        total_points += used

    get_cents = tax_fn(subtotal, channel=channel, ship_to_state=ship_to_state,
                       resale_ok=resale_ok) if tax_fn else 0
    return {
        "lines": lines,
        "subtotal_cents": subtotal,
        "discount_cents": total_discount,
        "points_redeemed_cents": total_points,
        "volume_months": total_months,
        "volume_pct": vpct,
        "get_cents": get_cents,
        "total_cents": subtotal + get_cents,
    }
```

- [ ] **Step 8: Run test to verify it passes**

Run: `... -m pytest tests/test_pricing_engine.py -k "compute or volume_pct" -v`
Expected: PASS (6 passed)

- [ ] **Step 9: Commit**

```bash
git add dashboard/pricing.py tests/test_pricing_engine.py
git commit -m "feat(pricing): volume curve + compute() best-of-one with floors and tax"
```

---

### Task 6: Points ledger (earn full-price-only, redeem, balance)

**Files:**
- Create: `dashboard/points.py`
- Test: `tests/test_points_ledger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_points_ledger.py
import sqlite3
from dashboard import points

def _cx():
    cx = sqlite3.connect(":memory:")
    points.init_points_table(cx)
    return cx

def test_earn_then_balance():
    cx = _cx()
    # earn 5% of a $70 full-price order = 350 cents of value
    bal = points.earn(cx, "a@x.com", full_price_cents=7000, earn_pct=0.05, order_ref="o1")
    assert bal == 350
    assert points.balance(cx, "a@x.com") == 350

def test_redeem_decrements_balance():
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=7000, earn_pct=0.05, order_ref="o1")
    bal = points.redeem(cx, "a@x.com", value_cents=200, order_ref="o2")
    assert bal == 150
    assert points.balance(cx, "a@x.com") == 150

def test_cannot_redeem_more_than_balance():
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=2000, earn_pct=0.05, order_ref="o1")  # 100
    import pytest
    with pytest.raises(ValueError):
        points.redeem(cx, "a@x.com", value_cents=500, order_ref="o2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_points_ledger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.points'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/points.py
"""Loyalty points ledger. Values stored in redemption-value CENTS (1 point = 5c).
Earn is on full-price spend only (caller decides eligibility); redeem is bounded
by balance here and by the price floor in dashboard.pricing.compute()."""

def init_points_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS points_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            delta_cents INTEGER NOT NULL,
            reason TEXT,
            order_ref TEXT,
            balance_after INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_points_email ON points_ledger(email)")
    cx.commit()

def balance(cx, email):
    row = cx.execute("SELECT COALESCE(SUM(delta_cents),0) FROM points_ledger WHERE email=?",
                     (email,)).fetchone()
    return int(row[0] or 0)

def _add(cx, email, delta_cents, reason, order_ref):
    bal = balance(cx, email) + int(delta_cents)
    cx.execute("""INSERT INTO points_ledger(email,delta_cents,reason,order_ref,balance_after)
                  VALUES (?,?,?,?,?)""", (email, int(delta_cents), reason, order_ref, bal))
    cx.commit()
    return bal

def earn(cx, email, *, full_price_cents, earn_pct, order_ref):
    delta = int(round(int(full_price_cents) * float(earn_pct)))
    return _add(cx, email, delta, "earn", order_ref)

def redeem(cx, email, *, value_cents, order_ref):
    value_cents = int(value_cents)
    if value_cents > balance(cx, email):
        raise ValueError("redeem exceeds balance")
    return _add(cx, email, -value_cents, "redeem", order_ref)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_points_ledger.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/points.py tests/test_points_ledger.py
git commit -m "feat(points): ledger with earn/redeem/balance integrity"
```

---

### Task 7: Wire config + a `/api/pricing/preview` endpoint (proves it end-to-end)

**Files:**
- Modify: `app.py` (config load near the other `_load_json` calls ~line 94; route near other `/api` routes)
- Test: `tests/test_pricing_preview_route.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pricing_preview_route.py
import json, app as appmod

def test_preview_prices_a_known_product(monkeypatch):
    client = appmod.app.test_client()
    # stub product lookup + tax so the test is deterministic
    monkeypatch.setattr(appmod, "_get_product",
                        lambda slug: {"slug": slug, "price_cents": 7000} if slug == "neuro-mag" else None)
    monkeypatch.setattr(appmod, "_qty_unit_cents", lambda p, qty: 7000)
    r = client.post("/api/pricing/preview", json={
        "items": [{"slug": "neuro-mag", "qty": 1}],
        "subscriber_tier_pct": 15, "ship_to_state": "CA"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["lines"][0]["line_total_cents"] == 5950   # 15% off 7000
    assert body["discount_cents"] == 1050

def test_preview_skips_unknown_product(monkeypatch):
    client = appmod.app.test_client()
    monkeypatch.setattr(appmod, "_get_product", lambda slug: None)
    r = client.post("/api/pricing/preview", json={"items": [{"slug": "nope", "qty": 1}]})
    assert r.status_code == 200
    assert r.get_json()["lines"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_pricing_preview_route.py -v`
Expected: FAIL (404 from the missing route)

- [ ] **Step 3: Write minimal implementation**

Add the config load next to the other `_load_json` calls (near app.py:94):

```python
_PRICING_SETTINGS = _load_json(DATA_DIR / "pricing-settings.json", default={})
```

Add the import near the top with the other `from dashboard import ...`:

```python
from dashboard import pricing as _pricing, tax as _tax
```

Add the route (near the other `/api/...` routes):

```python
@app.route("/api/pricing/preview", methods=["POST"])
def api_pricing_preview():
    data = request.get_json(silent=True) or {}
    settings = _pricing.load_settings(_PRICING_SETTINGS)
    items = []
    for it in (data.get("items") or []):
        p = _get_product(it.get("slug"))
        if not p:
            continue                      # unavailable/inactive → skip, never silently mis-price
        qty = max(1, int(it.get("qty") or 1))
        # base = TRUE single-unit list (volume is now a discount candidate, not a base price)
        items.append({
            "slug": p["slug"], "name": p.get("name", p["slug"]), "qty": qty, "product": p,
            "unit_cents": int(p.get("price_cents") or 0),
            "months": qty * int(p.get("months_per_unit", 1)),   # 30-cap bottle = 1 month
            "volume_eligible": bool(p.get("volume_eligible", True)),  # Pure Powders set False
        })
    result = _pricing.compute(
        items, settings=settings,
        subscriber_tier_pct=data.get("subscriber_tier_pct"),
        coupon_pct=data.get("coupon_pct"),
        points_to_redeem_cents=data.get("points_to_redeem_cents") or 0,
        channel=data.get("channel", "retail"),
        ship_to_state=data.get("ship_to_state"),
        tax_fn=_tax.compute_get_cents)
    return jsonify(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_pricing_preview_route.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_pricing_preview_route.py
git commit -m "feat(pricing): /api/pricing/preview endpoint over the engine"
```

---

### Task 8: Full-suite green + settings doc

**Files:**
- Create: `docs/pricing-engine.md`

- [ ] **Step 1: Run the whole new suite**

Run: `... -m pytest tests/test_pricing_engine.py tests/test_points_ledger.py tests/test_pricing_preview_route.py -v`
Expected: PASS (all)

- [ ] **Step 2: Write the operator doc**

```markdown
# Pricing engine (dashboard/pricing.py)
Single source of truth for cart pricing. The one % discount per line = max(volume, subscriber
tier or coupon) — best-of-one; subscriber tier never stacks with a coupon. Volume is a smooth
months-based curve (`volume_anchors`, mix-and-match across all Functional Formulations;
Pure Powders and `info_only` excluded via `volume_eligible=false`; 30-cap bottle = 1 month,
larger formats via `months_per_unit`). Base is the true single-unit list, so floors anchor to
list: discount floor 57% (wholesale), points floor 43%. Override globally in
`pricing-settings.json` (DATA_DIR); override a single SKU with `sku_discount_floor_pct` /
`sku_points_floor_pct` or absolute `wholesale_cents` (Pure Powders use 0.75 → $30). Points
(dashboard/points.py) earn 5% of full-price spend only, redeem above the floor, reduce the GET
tax base. Shipping is added by the caller at checkout (always charged, actual USA cost, US
ship-to only), never by the engine. Preview: POST /api/pricing/preview.
```

- [ ] **Step 3: Commit**

```bash
git add docs/pricing-engine.md
git commit -m "docs(pricing): operator notes for the pricing engine"
```

---

## Self-review

- **Spec coverage:** §A.2 best-of-one (volume vs subscriber vs coupon) → Task 5. §A.3 floors + per-SKU override (incl. Pure Powder 0.75/$30) → Tasks 2,5,7. §A.4 order of operations → Task 5. §A.5 points earn/redeem → Tasks 4,6. §A.6 single function → Task 5. §A.7 volume curve (months, mix-and-match, eligibility) → Tasks 1,5,7. GET-tax-on-discounted-base → Task 5. Console-editable per-SKU floors → fields read in Task 2/7 (the Products-console *edit UI* is a Plan 2 task). §A.8 shipping is a checkout concern (Plan 2), not the engine.
- **Deferred to Plan 2 (recurring orders):** the Products-console edit UI for per-SKU floor fields; the console editor for `pricing-settings.json`; wiring `compute()` into live `/reorder/checkout` + `/begin/checkout`; subscriptions table, Stripe vault setup, scheduler, dunning, manage portal, emails; earning points on full-price orders at checkout time.
- **Placeholders:** none — every code step is complete.
- **Type consistency:** `unit_floor_cents(product, list_cents, settings, kind)`, `apply_discount(list_cents, pct, floor_cents)`, `apply_points(price_cents, points_cents, floor_cents) -> (new, used)`, `compute(...) -> dict` are used identically across Tasks 2–7.

## Next
Plan 2 — Recurring Orders ("Subscribe & Grow"): subscriptions table, Stripe vault setup flow, daily scheduler (charge/skip/heads-up/dunning), manage-plan portal, per-SKU floor edit UI in the Products console, and checkout-time points earning — all pricing through this engine.
