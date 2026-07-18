# Membership-on-Invoice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a customer add a group-coaching membership as an invoice line item so the order's products price at the member rate provisionally, with the real membership grant firing only when the invoice is paid in full.

**Architecture:** A membership tier becomes a reserved order line (`slug="membership:<tier>"`, `kind="membership"`). Its mere presence in the cart makes the pricing engine treat the buyer as a member (computed, never persisted). On the order's fully-paid transition, a settlement effect grants the real membership (idempotent, keyed on the order). An "add membership & save" control on both the staff editor and the customer invoice toggles the line and shows gross / net-add / net-savings.

**Tech Stack:** Python 3 / Flask (single `app.py`), SQLite (`LOG_DB` / `chat_log.db`), vanilla-JS static pages (`static/*.html`), pytest. Pricing helpers in `dashboard/pricing.py`; order persistence in `dashboard/orders.py`; membership tiers in `dashboard/membership_products.py`; paid-order effects in `dashboard/order_settlement.py`.

## Global Constraints

- Member pricing while a membership line is on an unpaid invoice is **computed only** — no `memberships` row is written until payment. (Root-cause fix: enroll-before-payment is what this replaces.)
- The membership line is **excluded** from FF / volume-discount math and from `total_ff_qty`.
- **At most one** membership line per order; adding replaces any existing one.
- The real grant fires **only when the order reaches fully-paid** (`orders.set_order_payment` / `mark_order_paid_keep_status`), never on add, preview, or partial payment.
- Grant is **idempotent**, keyed on the order (`order_membership_grants(order_ref)` claim, `INSERT … ON CONFLICT DO NOTHING`), mirroring `group_bundle_grants`.
- If the buyer is **already a paid member** (`membership_products.owns_group`), the offer is hidden and no grant is written.
- Explicit per-line overrides (`rec["override"]=True`) do **not** auto-flip with the toggle — overrides stay sticky.
- Offered tiers are **configurable**; default `["month"]` (the $99 one-time, grant-only tier). Tier data: `dashboard/membership_products.py` `TIERS`.
- **Copy rules (Glen):** no em dashes, no ALL-CAPS words, no "Hook:" labels in any user-facing button/offer copy.
- Tests that must `import app` guard with `app = pytest.importorskip("app")` and are run via `doppler run --project remedy-match --config dev -- python3 -m pytest <path>`.

---

### Task 1: Membership-line helpers + tier config

Pure functions with no Flask/DB dependency, so they unit-test in isolation and every later task consumes them.

**Files:**
- Modify: `dashboard/membership_products.py` (append helpers; `TIERS`, `get_tier`, `grant_days`, `owns_group` already exist)
- Test: `tests/test_membership_invoice_line.py` (create)

**Interfaces:**
- Produces:
  - `line_slug(tier_key: str) -> str` → `"membership:<tier_key>"`
  - `line_for(tier_key: str) -> dict | None` → stored line dict, or None for unknown tier
  - `tier_of_line(line: dict) -> str | None` → tier key if `line` is a membership line, else None
  - `cart_has_membership_tier(lines: list[dict]) -> str | None` → first membership tier key in the cart, else None
  - `invoice_offer_tiers() -> list[str]` → configured tier keys, default `["month"]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membership_invoice_line.py
import os
from dashboard import membership_products as mp


def test_line_slug_and_for():
    assert mp.line_slug("month") == "membership:month"
    line = mp.line_for("month")
    assert line["slug"] == "membership:month"
    assert line["kind"] == "membership"
    assert line["tier"] == "month"
    assert line["qty"] == 1
    assert line["unit_cents"] == mp.get_tier("month")["price_cents"] == 9900
    assert line["line_cents"] == 9900
    assert line["name"] == mp.get_tier("month")["label"]


def test_line_for_unknown_tier_is_none():
    assert mp.line_for("nope") is None


def test_tier_of_line_detects_by_kind_and_slug():
    assert mp.tier_of_line({"kind": "membership", "tier": "month"}) == "month"
    assert mp.tier_of_line({"slug": "membership:year_prepay"}) == "year_prepay"
    assert mp.tier_of_line({"slug": "paracleanse", "qty": 1}) is None
    assert mp.tier_of_line({}) is None


def test_cart_has_membership_tier():
    cart = [{"slug": "paracleanse", "qty": 1}, {"slug": "membership:month", "kind": "membership"}]
    assert mp.cart_has_membership_tier(cart) == "month"
    assert mp.cart_has_membership_tier([{"slug": "paracleanse"}]) is None


def test_invoice_offer_tiers_default_and_env(monkeypatch):
    monkeypatch.delenv("MEMBERSHIP_INVOICE_TIERS", raising=False)
    assert mp.invoice_offer_tiers() == ["month"]
    monkeypatch.setenv("MEMBERSHIP_INVOICE_TIERS", "month, year_prepay , bogus")
    # keeps order, drops unknown tiers
    assert mp.invoice_offer_tiers() == ["month", "year_prepay"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_membership_invoice_line.py -v`
Expected: FAIL with `AttributeError: module 'dashboard.membership_products' has no attribute 'line_slug'`

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/membership_products.py`:

```python
import os

_MEMBERSHIP_LINE_PREFIX = "membership:"


def line_slug(tier_key):
    return f"{_MEMBERSHIP_LINE_PREFIX}{tier_key}"


def line_for(tier_key):
    """The stored order-line dict for a membership tier, or None if the tier is unknown.
    Carries kind='membership' + tier so pricing/rendering can recognize it without a
    product-catalog lookup (the slug is intentionally NOT a catalog product)."""
    t = TIERS.get(tier_key)
    if not t:
        return None
    return {"slug": line_slug(tier_key), "name": t["label"], "qty": 1,
            "unit_cents": t["price_cents"], "line_cents": t["price_cents"],
            "kind": "membership", "tier": tier_key}


def tier_of_line(line):
    """Tier key if `line` is a membership line (by kind marker or slug prefix), else None."""
    if not isinstance(line, dict):
        return None
    if line.get("kind") == "membership":
        tk = line.get("tier") or (line.get("slug") or "")[len(_MEMBERSHIP_LINE_PREFIX):]
        return tk if tk in TIERS else None
    slug = (line.get("slug") or "")
    if slug.startswith(_MEMBERSHIP_LINE_PREFIX):
        tk = slug[len(_MEMBERSHIP_LINE_PREFIX):]
        return tk if tk in TIERS else None
    return None


def cart_has_membership_tier(lines):
    """First membership tier key present in `lines`, else None."""
    for ln in (lines or []):
        tk = tier_of_line(ln)
        if tk:
            return tk
    return None


def invoice_offer_tiers():
    """Tier keys offered by the on-invoice membership control. Configurable via the
    MEMBERSHIP_INVOICE_TIERS env var (comma-separated); unknown tiers are dropped;
    default ['month']."""
    raw = (os.environ.get("MEMBERSHIP_INVOICE_TIERS") or "").strip()
    if not raw:
        return ["month"]
    out = [k.strip() for k in raw.split(",") if k.strip() in TIERS]
    return out or ["month"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_membership_invoice_line.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/membership_products.py tests/test_membership_invoice_line.py
git commit -m "feat: membership-line helpers + configurable invoice offer tiers"
```

---

### Task 2: Provisional member pricing + membership-line pricing

Make the two in-house pricing paths (a) price the membership line at its fixed tier price and skip it in product/FF logic, and (b) treat the buyer as a member when a membership line is in the cart.

**Files:**
- Modify: `app.py` — `_price_inhouse_invoice` loop (`app.py:38476` member gate, `app.py:38510` loop top) and `api_orders_price_preview` (`app.py:38993` member gate, `app.py:39009` loop top)
- Test: `tests/test_membership_invoice_pricing.py` (create)

**Interfaces:**
- Consumes: `membership_products.tier_of_line`, `.cart_has_membership_tier`, `.get_tier` (Task 1)
- Produces: no new symbols; behavior change to `POST /api/orders/price-preview` and `_price_inhouse_invoice`

- [ ] **Step 1: Write the failing test** (integration via the Flask test client; needs the product catalog, so it importorskips app)

```python
# tests/test_membership_invoice_pricing.py
import pytest
app_mod = pytest.importorskip("app")


@pytest.fixture
def client():
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def _preview(client, lines, email=""):
    r = client.post("/api/orders/price-preview",
                    json={"email": email, "lines": lines},
                    headers={"X-Console-Key": app_mod.CONSOLE_SECRET})
    assert r.status_code == 200, r.data
    return r.get_json()


def test_membership_line_flips_products_to_member(client):
    # A non-member cart of 6 different FFs prices at LIST without a membership line...
    ffs = [{"slug": s, "qty": 1} for s in
           ["paracleanse", "nerve-repair", "neuroceramides",
            "microbiome", "oxygen-cleanse", "macular-wellness-lycopene"]]
    base = _preview(client, ffs, email="nonmember-test@example.com")
    ff_lines = [l for l in base["lines"] if l["slug"] != "membership:month"]
    assert all(l["effective_unit_cents"] == l["list_cents"] for l in ff_lines)

    # ...but adding the membership line flips them to member pricing (savings > 0)
    withmem = _preview(client, ffs + [{"slug": "membership:month", "qty": 1}],
                       email="nonmember-test@example.com")
    ff_lines2 = [l for l in withmem["lines"] if l["slug"] != "membership:month"]
    assert all(l["effective_unit_cents"] < l["list_cents"] for l in ff_lines2)
    # the membership line itself is priced at the tier price and is not FF
    mem = [l for l in withmem["lines"] if l["slug"] == "membership:month"][0]
    assert mem["effective_unit_cents"] == 9900
    assert mem["is_ff"] is False
    assert mem["savings_cents"] == 0
    # subtotal includes the $99 membership on top of the (now discounted) products
    assert withmem["subtotal_cents"] == sum(l["line_cents"] for l in withmem["lines"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_membership_invoice_pricing.py -v`
Expected: FAIL — without the change the `membership:month` line is dropped (`if not p: continue`) and the FFs stay at list, so `effective_unit_cents < list_cents` assertions fail.

- [ ] **Step 3: Write minimal implementation**

3a. In `_price_inhouse_invoice`, change the member gate at `app.py:38476`:

```python
    total_ff_qty = _inhouse_total_ff_qty(lines_in)
    program_member = _is_paid_member(email) or bool(_mp.cart_has_membership_tier(lines_in))
```

(ensure `from dashboard import membership_products as _mp` is imported at module top; it already is elsewhere — reuse the existing alias if present.)

3b. Handle the membership line at the very top of the loop, before `_get_product`, at `app.py:38510`:

```python
    for ln in lines_in:
        slug = (ln.get("slug") or "").strip()
        _mtier = _mp.tier_of_line(ln)
        if _mtier:
            t = _mp.get_tier(_mtier)
            _mrec = {"slug": _mp.line_slug(_mtier), "name": t["label"], "qty": 1,
                     "unit_cents": t["price_cents"], "line_cents": t["price_cents"],
                     "kind": "membership", "tier": _mtier}
            items_rec.append(_mrec)
            subtotal_list += t["price_cents"]
            continue
        p = _get_product(slug)
        if not p:
            continue
```

3c. Apply the identical two changes to `api_orders_price_preview`:
- member gate at `app.py:38993`:
```python
    _ppm = _is_paid_member(_pemail) or bool(_mp.cart_has_membership_tier(lines_in))
```
- membership-line branch at the top of the loop (`app.py:39009`), emitting the preview line shape:
```python
    for ln in lines_in:
        slug = (ln.get("slug") or "").strip()
        _mtier = _mp.tier_of_line(ln)
        if _mtier:
            t = _mp.get_tier(_mtier)
            out_lines.append({"slug": _mp.line_slug(_mtier), "qty": 1, "is_ff": False,
                              "list_cents": t["price_cents"], "effective_unit_cents": t["price_cents"],
                              "line_cents": t["price_cents"], "vol_pct": 0, "savings_cents": 0})
            subtotal += t["price_cents"]
            continue
```

3d. Confirm `_inhouse_total_ff_qty` (`app.py:5519`) already excludes the membership line: its slug resolves to no product so `_qty_eligible` is false. No change needed, but add a defensive `if _mp.tier_of_line(ln): continue` at its loop top if it does not already null-guard `_get_product`.

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_membership_invoice_pricing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_membership_invoice_pricing.py
git commit -m "feat: membership line prices products as member (provisional, computed)"
```

---

### Task 3: Membership-offer math endpoint

One endpoint the UIs call to render gross / net-add / net-savings for a cart + tier.

**Files:**
- Modify: `app.py` (add route near `api_orders_price_preview`, ~`app.py:39061`)
- Test: `tests/test_membership_offer_endpoint.py` (create)

**Interfaces:**
- Consumes: `membership_products.get_tier`, `.invoice_offer_tiers`; the existing `/api/orders/price-preview` pricing (Task 2)
- Produces: `POST /api/orders/membership-offer` → `{"ok":true, "tier", "gross_cents", "savings_cents", "net_add_cents", "net_savings_cents", "offered_tiers":[...]}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membership_offer_endpoint.py
import pytest
app_mod = pytest.importorskip("app")


@pytest.fixture
def client():
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def test_offer_math_for_six_ffs(client):
    ffs = [{"slug": s, "qty": 1} for s in
           ["paracleanse", "nerve-repair", "neuroceramides",
            "microbiome", "oxygen-cleanse", "macular-wellness-lycopene"]]
    r = client.post("/api/orders/membership-offer",
                    json={"email": "nonmember-test@example.com", "tier": "month", "lines": ffs},
                    headers={"X-Console-Key": app_mod.CONSOLE_SECRET})
    assert r.status_code == 200, r.data
    j = r.get_json()
    assert j["gross_cents"] == 9900
    assert j["savings_cents"] > 0                       # membership unlocks FF savings
    assert j["net_add_cents"] == max(0, 9900 - j["savings_cents"])
    assert j["net_savings_cents"] == j["savings_cents"]
    assert "month" in j["offered_tiers"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_membership_offer_endpoint.py -v`
Expected: FAIL 404 (route not defined)

- [ ] **Step 3: Write minimal implementation** (add after `api_orders_price_preview`)

```python
@app.route("/api/orders/membership-offer", methods=["POST"])
def api_orders_membership_offer():
    """Given a product cart + a membership tier, return the offer economics: gross fee,
    the product savings the membership unlocks on THIS order, and the net add. Pure read;
    no persistence. Used by the staff editor and the customer invoice controls."""
    _body = request.get_json(silent=True) or {}
    tier_key = (_body.get("tier") or "month").strip()
    tier = _mp.get_tier(tier_key)
    if not tier:
        return jsonify({"ok": False, "error": "unknown tier"}), 400
    lines_in = _body.get("lines") or []
    email = (_body.get("email") or "").strip().lower()
    # Price the product lines twice: as-is (list for a non-member) and with the membership
    # line present (member). Savings = the delta the membership unlocks.
    def _subtotal(lines):
        with app.test_request_context(
                "/api/orders/price-preview", method="POST",
                json={"email": email, "lines": lines}):
            resp = api_orders_price_preview()
        data = resp.get_json() if hasattr(resp, "get_json") else resp[0].get_json()
        prod = [l for l in data["lines"] if not str(l["slug"]).startswith("membership:")]
        return sum(l["line_cents"] for l in prod), prod
    base_sub, _ = _subtotal(lines_in)
    mem_sub, _ = _subtotal(lines_in + [{"slug": _mp.line_slug(tier_key), "qty": 1}])
    savings = max(0, base_sub - mem_sub)
    gross = int(tier["price_cents"])
    return jsonify({"ok": True, "tier": tier_key, "gross_cents": gross,
                    "savings_cents": savings, "net_add_cents": max(0, gross - savings),
                    "net_savings_cents": savings, "offered_tiers": _mp.invoice_offer_tiers()})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_membership_offer_endpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_membership_offer_endpoint.py
git commit -m "feat: /api/orders/membership-offer returns gross/net-add/net-savings"
```

---

### Task 4: Membership line renders on the customer invoice

Pass the membership marker through the invoice line-view so the line shows as a labelled membership, not an unknown product.

**Files:**
- Modify: `app.py` `_invoice_line_view` (`app.py:39605`)
- Modify: `static/invoice.html` line-map (`invoice.html:196`)
- Test: `tests/test_invoice_line_view_membership.py` (create)

**Interfaces:**
- Consumes: membership line shape from Task 1 (`kind`, `tier`)
- Produces: `_invoice_line_view` output carries `kind` and `tier` for membership lines

- [ ] **Step 1: Write the failing test**

```python
# tests/test_invoice_line_view_membership.py
import pytest
app_mod = pytest.importorskip("app")


def test_membership_line_view_passthrough():
    line = {"slug": "membership:month", "name": "Monthly Membership",
            "qty": 1, "unit_cents": 9900, "line_cents": 9900,
            "kind": "membership", "tier": "month"}
    out = app_mod._invoice_line_view(line)
    assert out["kind"] == "membership"
    assert out["tier"] == "month"
    assert out["name"] == "Monthly Membership"
    assert out["line_cents"] == 9900
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_invoice_line_view_membership.py -v`
Expected: FAIL — `out` has no `kind` key (KeyError)

- [ ] **Step 3: Write minimal implementation**

In `_invoice_line_view` (`app.py:39605`), after the base `out` dict is built and before the `_get_product` product-anchoring block, add:

```python
    if l.get("kind") == "membership":
        out["kind"] = "membership"
        out["tier"] = l.get("tier")
        return out
```

In `static/invoice.html:196`, extend the copied fields so the marker survives to the renderer:

```javascript
  return {slug:l.slug, name:l.name, qty:l.qty, unit_cents:l.unit_cents,
          line_cents:l.line_cents, kind:l.kind, tier:l.tier,
          service:l.service, srp_cents:l.srp_cents, regular_cents:l.regular_cents};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_invoice_line_view_membership.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py static/invoice.html tests/test_invoice_line_view_membership.py
git commit -m "feat: render membership line on customer invoice"
```

---

### Task 5: Grant membership when the order is paid in full

A settlement effect that grants the real membership on the paid transition, idempotent per order, wired into both the card path (settlement hub) and the alt-pay path (`_record_payment_exec`).

**Files:**
- Modify: `app.py` — new `_grant_membership_line_on_paid(cx, order)`; register claim table; wire into `_SETTLEMENT_DEPS` (`app.py:6441`) and `_record_payment_exec` (`dashboard/orders.py:906` calls back into an app dep — see 5c)
- Modify: `dashboard/order_settlement.py` `settle_paid_order_effects` (`order_settlement.py:23`) — new dispatch branch
- Test: `tests/test_membership_grant_on_paid.py` (create)

**Interfaces:**
- Consumes: `membership_products.cart_has_membership_tier`, `.get_tier`, `.grant_days`, `.owns_group` (Task 1); `_grant_membership` (`app.py:11837`)
- Produces:
  - `_grant_membership_line_on_paid(cx, order) -> str` returning `"granted"`, `"already"`, `"none"`, or `"member"`
  - claim table `order_membership_grants(order_ref TEXT PRIMARY KEY, email TEXT, tier TEXT, created_at TEXT)`
  - `_SETTLEMENT_DEPS.grant_membership_line` dep + a `grant_membership_line(md)` branch in `settle_paid_order_effects`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membership_grant_on_paid.py
import sqlite3, pytest
app_mod = pytest.importorskip("app")
from dashboard import orders


@pytest.fixture
def cx(tmp_path, monkeypatch):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    orders.init_orders_table(c)
    app_mod.init_membership_tables(c)
    return c


def _order_with_membership(cx, email="grant-test@example.com"):
    orders.upsert_order(cx, source="inhouse", external_ref="INH-TESTMEM",
                        email=email, total_cents=9900,
                        items=[{"slug": "membership:month", "name": "Monthly Membership",
                                "qty": 1, "unit_cents": 9900, "line_cents": 9900,
                                "kind": "membership", "tier": "month"}])
    return orders.get_order_by_ref(cx, "INH-TESTMEM")


def test_grants_once_and_is_idempotent(cx):
    o = _order_with_membership(cx)
    assert app_mod._grant_membership_line_on_paid(cx, o) == "granted"
    assert app_mod._is_paid_member("grant-test@example.com") is True
    # second call on the same order does not double-grant
    assert app_mod._grant_membership_line_on_paid(cx, o) == "already"


def test_no_membership_line_is_noop(cx):
    orders.upsert_order(cx, source="inhouse", external_ref="INH-NOMEM",
                        email="x@example.com", total_cents=6997,
                        items=[{"slug": "paracleanse", "name": "ParaCleanse",
                                "qty": 1, "unit_cents": 6997, "line_cents": 6997}])
    o = orders.get_order_by_ref(cx, "INH-NOMEM")
    assert app_mod._grant_membership_line_on_paid(cx, o) == "none"


def test_already_member_does_not_regrant(cx, monkeypatch):
    monkeypatch.setattr(app_mod._mp, "owns_group", lambda _cx, _e: True)
    o = _order_with_membership(cx, email="already@example.com")
    assert app_mod._grant_membership_line_on_paid(cx, o) == "member"
```

(If `orders.get_order_by_ref` is not the exact accessor name, use the ref-lookup that exists in `dashboard/orders.py`; confirm during implementation.)

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_membership_grant_on_paid.py -v`
Expected: FAIL — `_grant_membership_line_on_paid` undefined

- [ ] **Step 3: Write minimal implementation**

3a. Add the claim table to `init_membership_tables` (`app.py:11593`):

```python
    cx.execute("""CREATE TABLE IF NOT EXISTS order_membership_grants (
        order_ref TEXT PRIMARY KEY, email TEXT, tier TEXT, created_at TEXT)""")
```

3b. Add the grant function near `_grant_membership` (`app.py:11837`):

```python
def _grant_membership_line_on_paid(cx, order):
    """On a fully-paid order that carries a membership line, write the real membership
    grant. Idempotent per order_ref (claim row). Returns granted|already|none|member."""
    if not order:
        return "none"
    tier_key = _mp.cart_has_membership_tier(order.get("items") or [])
    if not tier_key:
        return "none"
    email = (order.get("email") or "").strip().lower()
    ref = order.get("external_ref") or f"id:{order.get('id')}"
    if _mp.owns_group(cx, email):
        return "member"
    claim = cx.execute(
        "INSERT INTO order_membership_grants (order_ref, email, tier, created_at) "
        "VALUES (?,?,?,?) ON CONFLICT(order_ref) DO NOTHING",
        (ref, email, tier_key, datetime.utcnow().isoformat() + "Z"))
    if not claim.rowcount:
        return "already"
    days = _mp.grant_days(tier_key, _now_utc().date())
    _grant_membership(cx, email, days, _mp.get_tier(tier_key)["source"])
    cx.commit()
    return "granted"
```

3c. Wire into the card/checkout hub. In `order_settlement.py` `settle_paid_order_effects` (`order_settlement.py:23`) add a branch that calls `deps.grant_membership_line(md)`; in `app.py:6441` add the dep:

```python
    grant_membership_line=lambda _md: _grant_membership_line_on_paid(
        _sqlite3.connect(LOG_DB), _md.get("order")) ,
```

(Follow the existing dep bundle's connection/commit convention exactly — mirror `_grant_group_bundle`.)

3d. Wire into alt-pay. In `dashboard/orders.py` `_record_payment_exec` (`orders.py:906`), immediately after `set_order_payment(...)` and `book_sale_on_payment`, invoke the same grant via the app hook the module already uses for settlement side-effects (mirror how `_qsale.book_sale_on_payment` is injected). If `_record_payment_exec` has no app callback seam, add the call in the app-layer action registration at `orders.py:959` where `record_payment` is wired, so alt-pay (Zelle/check) on a membership line also grants.

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_membership_grant_on_paid.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py dashboard/order_settlement.py dashboard/orders.py tests/test_membership_grant_on_paid.py
git commit -m "feat: grant membership on fully-paid order carrying a membership line"
```

---

### Task 6: Staff editor control (order-new.html)

An "Add membership" control in the pricing section that toggles the membership line into `LINES`, reprices via preview, and shows gross / net-add / net-savings.

**Files:**
- Modify: `static/order-new.html` (pricing section near `app.py`-served markup ~line 105; `LINES`/`linesPayload`/`refreshPreview` at `order-new.html:166,318,321`)
- Test: manual + `tests/test_membership_offer_endpoint.py` already covers the math; UI verified via headless browser at review.

**Interfaces:**
- Consumes: `POST /api/orders/membership-offer` (Task 3); membership line shape (Task 1); the existing `linesPayload()`/`price-preview` flow
- Produces: a membership line in `LINES` when toggled on (`{slug:"membership:"+tier, qty:1, kind:"membership", tier}`)

- [ ] **Step 1: Add markup** — a control block after the Payments/pricing section:

```html
<div id="membership-offer" class="section" style="display:none">
  <h2>Group coaching membership</h2>
  <div id="membership-offer-copy" class="muted"></div>
  <button class="ghost" id="membership-toggle" onclick="toggleMembership()">Add membership</button>
</div>
```

- [ ] **Step 2: Add the toggle + offer refresh JS**

```javascript
function membershipTier(){ return (window.MEMBERSHIP_TIER || 'month'); }
function hasMembershipLine(){ return LINES.some(l => (l.slug||'').startsWith('membership:')); }
function toggleMembership(){
  const slug = 'membership:' + membershipTier();
  if (hasMembershipLine()) LINES = LINES.filter(l => !(l.slug||'').startsWith('membership:'));
  else LINES.push({slug, name:'Membership', qty:1, unit_cents:0, edited:false, kind:'membership', tier:membershipTier()});
  renderLines(); refreshPreview(); refreshMembershipOffer();
}
async function refreshMembershipOffer(){
  const prod = LINES.filter(l => !(l.slug||'').startsWith('membership:'))
                    .map(l => ({slug:l.slug, qty:l.qty}));
  if (!prod.length){ $('membership-offer').style.display='none'; return; }
  const r = await fetch('/api/orders/membership-offer', {method:'POST', headers:HEADERS,
    body: JSON.stringify({email:$('c-email').value.trim(), tier:membershipTier(), lines:prod})});
  const j = await r.json(); if(!j.ok) return;
  $('membership-offer').style.display='';
  $('membership-toggle').textContent = hasMembershipLine() ? 'Remove membership' : 'Add membership';
  const g=dollars(j.gross_cents), s=dollars(j.net_savings_cents), n=dollars(j.net_add_cents);
  $('membership-offer-copy').innerHTML = j.net_add_cents > 0
    ? `${g} membership. Saves ${s} on this order, so just ${n} more.`
    : `${g} membership. Saves ${s} on this order, effectively free.`;
}
```

Call `refreshMembershipOffer()` at the end of `refreshPreview()` and once after edit-mode load.

- [ ] **Step 3: Verify in a real browser**

Load `/orders/new?edit_order=<a test order>&key=…`, click Add membership, confirm the FF lines drop to member prices, the membership line appears at $99, the total rises by the net, and the copy shows gross/net-savings/net-add. Toggle off and confirm it reverts.

- [ ] **Step 4: Commit**

```bash
git add static/order-new.html
git commit -m "feat: staff editor add-membership control with net-savings copy"
```

---

### Task 7: Customer invoice control + endpoint

A token-authed endpoint to add/remove the membership line on an order, plus the offer card on `invoice.html`, so the customer self-serves it before paying.

**Files:**
- Modify: `app.py` — new `POST /api/invoice/<token>/membership` near `api_invoice_get` (`app.py:39730`)
- Modify: `static/invoice.html` — offer card + toggle + reprice
- Test: `tests/test_invoice_membership_toggle.py` (create)

**Interfaces:**
- Consumes: `_invoice_order_for_token` (existing), `_reprice_and_persist_invoice` (`app.py:38634`), `membership_products` helpers (Task 1), `owns_group`
- Produces: `POST /api/invoice/<token>/membership` `{action:"add"|"remove", tier?}` → updated invoice summary (same shape as `api_invoice_get`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_invoice_membership_toggle.py
import sqlite3, pytest
app_mod = pytest.importorskip("app")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "LOG_DB", str(tmp_path / "chat_log.db"))
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def test_add_then_remove_membership_reprices(client, monkeypatch):
    token = _seed_unpaid_order_with_token(client, monkeypatch)   # helper: 6 FFs, non-member
    add = client.post(f"/api/invoice/{token}/membership", json={"action": "add", "tier": "month"})
    assert add.status_code == 200, add.data
    j = add.get_json()["order"]
    assert any(l["slug"] == "membership:month" for l in j["lines"])
    ff = [l for l in j["lines"] if l["slug"] != "membership:month"]
    assert all(l["unit_cents"] < 6997 for l in ff)      # products now member-priced
    rem = client.post(f"/api/invoice/{token}/membership", json={"action": "remove"})
    j2 = rem.get_json()["order"]
    assert not any(l["slug"] == "membership:month" for l in j2["lines"])
    ff2 = [l for l in j2["lines"] if l["slug"] != "membership:month"]
    assert all(l["unit_cents"] == 6997 for l in ff2)     # reverted to list
```

(`_seed_unpaid_order_with_token` builds an unpaid in-house order via `orders.upsert_order` with a fresh `invoice_token`; mirror the seeding in `tests/test_client_order.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_invoice_membership_toggle.py -v`
Expected: FAIL 404 (route undefined)

- [ ] **Step 3: Write minimal implementation**

```python
@app.route("/api/invoice/<token>/membership", methods=["POST"])
def api_invoice_membership(token):
    """Customer-facing: add or remove the membership line on their own (unpaid) invoice
    and reprice. Guarded: only unpaid orders; hidden/blocked if already a paid member."""
    order = _invoice_order_for_token(token)
    if not order:
        return jsonify({"ok": False, "error": "invalid or expired invoice"}), 404
    if order.get("pay_status") == "paid":
        return jsonify({"ok": False, "error": "invoice already paid"}), 409
    body = request.get_json(silent=True) or {}
    action = (body.get("action") or "").strip()
    lines = [l for l in (order.get("items") or [])
             if not str(l.get("slug") or "").startswith("membership:")]
    if action == "add":
        tier_key = (body.get("tier") or "month").strip()
        if tier_key not in _mp.invoice_offer_tiers():
            return jsonify({"ok": False, "error": "tier not offered"}), 400
        with _sqlite3.connect(LOG_DB) as _mc:
            if _mp.owns_group(_mc, (order.get("email") or "").lower()):
                return jsonify({"ok": False, "error": "already a member"}), 409
        lines.append({"slug": _mp.line_slug(tier_key), "qty": 1})
    elif action != "remove":
        return jsonify({"ok": False, "error": "action must be add or remove"}), 400
    payload = [{"slug": l["slug"], "qty": int(l.get("qty") or 1),
                **({"unit_cents": l["unit_cents"]} if l.get("override") else {})}
               for l in lines]
    cx = _sqlite3.connect(LOG_DB); cx.row_factory = _sqlite3.Row
    try:
        _reprice_and_persist_invoice(cx, order, payload, pickup=(order.get("channel") == "pickup"))
    finally:
        cx.close()
    return api_invoice_get(token)     # return the fresh summary
```

Guards match the constraints: unpaid only, offered tier only, not-already-member, membership line stripped-then-re-added so add is idempotent and remove is clean.

3b. In `static/invoice.html`, add an offer card (hidden when paid or already a member) that calls the endpoint and re-renders from the returned summary:

```javascript
async function toggleInvoiceMembership(action){
  const r = await fetch(API + '/membership', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action, tier: (ORDER.offer_tier||'month')})});
  const j = await r.json(); if(!j.ok){ alert(j.error||'Could not update'); return; }
  ORDER = j.order; renderInvoice(); renderPayments();
}
```

Populate the offer copy from a call to `/api/orders/membership-offer` (Task 3) on load, using the same no-em-dash copy rule as Task 6.

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_invoice_membership_toggle.py -v`
Expected: PASS

- [ ] **Step 5: Verify in a real browser** on a seeded unpaid invoice token, then **Commit**

```bash
git add app.py static/invoice.html tests/test_invoice_membership_toggle.py
git commit -m "feat: customer invoice add/remove membership with live reprice"
```

---

### Task 8: Dana #66 rollout (operational — no TDD)

Apply the feature to Dana's real order and clear the two prerequisites from the spec. Do this only after Tasks 1-7 are deployed.

- [ ] **Step 1: Clear the redundant overrides on #66** so the order floats (and can flip to member if she takes the offer). Re-save #66 through the staff editor sending the 6 FF lines WITHOUT `unit_cents` (she is now a revoked non-member, so they recompute to $69.97 as floats). Verify `override=None` on all 6 and total unchanged at $432.82.

- [ ] **Step 2: Add the product she wants** via the staff editor; confirm it prices at non-member list.

- [ ] **Step 3: Payments-on-invoice depends on the QBO reconcile.** Do NOT record her card in the ledger until Rae deletes duplicate invoice 24457 (else double-count). After that delete, record card $222.91 + Zelle $131 + $88 in the ledger so the customer invoice shows payments + balance.

- [ ] **Step 4: Send the invoice** with the membership offer visible; confirm the net-savings copy renders and that adding membership reprices correctly for her.

---

## Self-Review

**Spec coverage:**
- Both surfaces → Tasks 6 (staff) + 7 (customer). ✓
- Configurable tiers → Task 1 `invoice_offer_tiers`. ✓
- Auto-flip pricing (add/remove), member-guarded → Tasks 2 (engine) + 6/7 (toggle) + `owns_group` guard. ✓
- Grant on fully-paid, idempotent, both pay paths → Task 5. ✓
- Gross + net-add + net-savings → Task 3 + 6/7 copy. ✓
- Customer invoice shows payments + balance → already present (`api_invoice_get`); membership line renders via Task 4. ✓
- Membership line excluded from FF math → Task 2 (branch before `_get_product`; `_inhouse_total_ff_qty` guard). ✓
- Override lines don't auto-flip → preserved by `_price_inhouse_invoice` override precedence (unchanged). ✓
- Dana prerequisites → Task 8. ✓

**Placeholder scan:** No TBD/TODO. Two named accessors flagged for confirmation at implementation (`orders.get_order_by_ref`, the `_record_payment_exec` app-callback seam) with explicit fallback instructions — not silent placeholders.

**Type consistency:** membership line shape `{slug, name, qty, unit_cents, line_cents, kind, tier}` is identical across Tasks 1, 2, 4, 5. Helper names (`tier_of_line`, `cart_has_membership_tier`, `line_slug`, `line_for`, `invoice_offer_tiers`, `grant_days`) match across tasks. Endpoint response keys (`gross_cents`, `net_add_cents`, `net_savings_cents`) match Tasks 3, 6, 7.
