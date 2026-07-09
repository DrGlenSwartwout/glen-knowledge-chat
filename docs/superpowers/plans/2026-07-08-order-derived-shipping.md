# Derived Shipping + Per-Client Pickup Default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop charging shipping for things that cannot be shipped, and let an operator mark a client as "picks up by default" without that decision leaking into orders it shouldn't touch.

**Architecture:** "This order has nothing to ship" becomes a *derived* fact — one predicate, `is_shippable(product)`, consulted where box counts are built — so it is never stored and never needs unsetting. `pickup` goes back to meaning a human's fulfillment choice about a physical order, and that choice alone is what a new `client_prefs` table remembers, written only by an explicit operator toggle.

**Tech Stack:** Python 3, Flask, sqlite3, vanilla-JS static console pages, pytest.

**Spec:** `docs/superpowers/specs/2026-07-08-order-derived-shipping-design.md`

## Global Constraints

- **Repo:** work inside the worktree `/tmp/wt-deploy-chat-75cbcf9d` on branch `sess/75cbcf9d`. Never `cd ~/deploy-chat` to edit — another session shares that checkout.
- **Tests that import `app`** need real secrets and a writable `DATA_DIR`. They run under the Doppler harness. Because this command contains a `--` separator, write it to a script rather than pasting it:
  ```bash
  printf '#!/bin/bash\ndoppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 -m pytest "$@"\n' > /tmp/dt.sh && chmod +x /tmp/dt.sh
  ```
  Then run: `/tmp/dt.sh tests/test_x.py -v`
- **Tests that do NOT import `app`** (pure `dashboard/` modules) run with plain `python3 -m pytest tests/test_x.py -v`.
- **Forward-only.** No data migration. Orders already carrying `channel='pickup'` keep it.
- **`dashboard/biofield_invoice.py` keeps sending `pickup: True`.** That is a deliberate courtesy absorbed into the $300 analysis fee, not a bug. Do not remove it.
- **Nothing writes `client_prefs` except the explicit console endpoint.** Creating or saving an order must never touch it.
- Match the surrounding code's comment density and idiom. These files explain *why*, not *what*.

---

## File Structure

| File | Responsibility |
|---|---|
| `dashboard/shipping.py` | **Modify.** Owns all box/rate logic. Gains `is_shippable(product)` — the single answer to "is there a physical thing here?" |
| `app.py` `_price_cart` (~5350-5390) | **Modify.** Skip non-shippable lines when building `box_counts`/`total_bottles`; gate the non-US country check on the cart having something to ship. |
| `dashboard/products.py` `catalog()` | **Modify.** Emit `shippable` per product so the browser can see the same fact the server does. |
| `dashboard/client_prefs.py` | **Create.** Per-client fulfillment preference. Pure functions over a sqlite connection. Mirrors `client_prices.py`. |
| `app.py` `/api/console/client-prefs` | **Create.** The *only* writer of `client_prefs`. Owner-gated. |
| `static/order-new.html` | **Modify.** Disable Pickup when nothing is shippable; add the per-client default checkbox; pre-check Pickup from the preference on new orders. |
| `dashboard/biofield_invoice.py` | **Modify.** Comment only. |

---

### Task 1: `is_shippable` — the single predicate

**Files:**
- Modify: `dashboard/shipping.py` (add after `resolve_bottle_type`, which ends at line 566)
- Test: `tests/test_shipping.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `is_shippable(product: dict | None) -> bool`. Later tasks import it as `from dashboard.shipping import is_shippable` or call it as `_shipping.is_shippable(p)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_shipping.py`:

```python
# ── Shippability ──────────────────────────────────────────────────────────────

def test_is_shippable_false_for_services_and_info_only():
    """A service or info-only SKU has nothing to put in a box."""
    from dashboard.shipping import is_shippable
    assert is_shippable({"name": "Biofield Analysis", "service": True, "info_only": True}) is False
    assert is_shippable({"name": "EVOX Session", "service": True, "info_only": True}) is False
    assert is_shippable({"name": "EMF", "info_only": True}) is False


def test_is_shippable_true_for_a_normal_product():
    from dashboard.shipping import is_shippable
    assert is_shippable({"name": "Neuro Magnesium", "price_cents": 6997}) is True
    assert is_shippable({"name": "Hand Cradle", "bottle_type": "default"}) is True


def test_is_shippable_handles_missing_product():
    from dashboard.shipping import is_shippable
    # No caller passes None — _price_cart skips falsy products at app.py:5356 —
    # but the predicate must not raise on one. It treats it as an empty product.
    assert is_shippable(None) is True
    assert is_shippable({}) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_shipping.py -k is_shippable -v`
Expected: FAIL with `ImportError: cannot import name 'is_shippable'`

- [ ] **Step 3: Write minimal implementation**

In `dashboard/shipping.py`, immediately after `resolve_bottle_type` (after line 566):

```python
def is_shippable(product) -> bool:
    """False when a product has no physical thing to put in a box.

    The ONE place this question is asked. Services (Biofield Analysis, EVOX) and
    info-only SKUs (EMF, an affiliate link) carry no bottle and must not inflate
    a box count — four bottles plus a Biofield Analysis is still four bottles.
    A future digital SKU is taught here and nowhere else.
    """
    p = product or {}
    return not (p.get("service") or p.get("info_only"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_shipping.py -k is_shippable -v`
Expected: PASS (3 tests)

Then confirm nothing else broke: `python3 -m pytest tests/test_shipping.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/shipping.py tests/test_shipping.py
git commit -m "feat(shipping): add is_shippable predicate"
```

---

### Task 2: `_price_cart` stops boxing services

This is the money fix. A cart of four bottles plus a Biofield Analysis currently counts **five** items and can select a **Medium** flat-rate box where a **Small** would do.

**Files:**
- Modify: `app.py` lines 5364-5367 (inside `_price_cart`'s cart loop)
- Test: `tests/test_price_cart_shippable.py` (create)

**Interfaces:**
- Consumes: `dashboard.shipping.is_shippable` from Task 1. In `app.py` the module is already imported as `_shipping`, so call `_shipping.is_shippable(p)`.
- Produces: no new symbols. `_price_cart`'s return shape is unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_price_cart_shippable.py`:

```python
"""A non-physical line must not add to the box count.

Imports app (needs real secrets + writable DATA_DIR), so it's skipped under plain
pytest and runs under the Doppler harness:
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 -m pytest tests/test_price_cart_shippable.py
"""
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import app
except Exception as _e:  # pragma: no cover - exercised only under plain pytest
    pytest.skip(f"app import requires real secrets: {_e}", allow_module_level=True)

US = {"name": "T", "state": "HI", "country": "US"}


def test_a_service_line_does_not_change_shipping():
    """Four bottles ship for the same price with or without a Biofield Analysis
    added. Before this fix the service counted as a fifth bottle and could push
    the order up a box size."""
    bottles_only = app._price_inhouse_invoice(
        [{"slug": "neuro-magnesium", "qty": 4}],
        email="", pickup=False, ship=US)
    bottles_plus_service = app._price_inhouse_invoice(
        [{"slug": "neuro-magnesium", "qty": 4},
         {"slug": "biofield-analysis", "qty": 1}],
        email="", pickup=False, ship=US)
    assert bottles_only["shipping_cents"] > 0            # a real box was quoted
    assert bottles_plus_service["shipping_cents"] == bottles_only["shipping_cents"]


def test_services_only_cart_has_no_shipping_without_pickup():
    """The derived rule: nothing physical -> no shipping, no flag required."""
    res = app._price_inhouse_invoice(
        [{"slug": "biofield-analysis", "qty": 1}],
        email="", pickup=False, ship=US)
    assert res["shipping_cents"] == 0


def test_bottles_still_ship_when_not_pickup():
    """Guard against the fix over-reaching: real products still cost shipping."""
    res = app._price_inhouse_invoice(
        [{"slug": "neuro-magnesium", "qty": 1}],
        email="", pickup=False, ship=US)
    assert res["shipping_cents"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
printf '#!/bin/bash\ndoppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 -m pytest "$@"\n' > /tmp/dt.sh && chmod +x /tmp/dt.sh
/tmp/dt.sh tests/test_price_cart_shippable.py -v
```

Expected: `test_services_only_cart_has_no_shipping_without_pickup` FAILS — it quotes a Small Flat Rate box (a nonzero `shipping_cents`) for a service.

`test_a_service_line_does_not_change_shipping` may pass or fail depending on where 4 vs 5 bottles fall in the current box-capacity table. **Both outcomes are informative — record which you saw in the commit message.** If it passes today, it is still the regression guard that keeps this fix honest.

If ALL THREE tests are skipped, the Doppler harness is not working. Stop and fix that before continuing; a skipped test proves nothing.

- [ ] **Step 3: Write minimal implementation**

In `app.py`, inside `_price_cart`'s cart loop. Replace lines 5364-5367:

```python
        bt = _shipping.resolve_bottle_type(slug, p)
        box_counts[bt] = box_counts.get(bt, 0) + qty
        total_bottles += qty
```

with:

```python
        # A service / info-only line has nothing to put in a box, so it must not
        # inflate the box count: 4 bottles + a Biofield Analysis is still a Small.
        # A cart of ONLY such lines leaves box_counts empty, and _shipping_for_cart
        # already returns 0 for that — no flag, no stored state.
        if _shipping.is_shippable(p):
            bt = _shipping.resolve_bottle_type(slug, p)
            box_counts[bt] = box_counts.get(bt, 0) + qty
            total_bottles += qty
```

Leave the comment above it (`# shipping.pick_box keys by BOTTLE TYPE...`) in place.

- [ ] **Step 4: Run test to verify it passes**

Run: `/tmp/dt.sh tests/test_price_cart_shippable.py -v`
Expected: PASS (3 tests)

Then confirm no regression in the existing pricer tests:
Run: `/tmp/dt.sh tests/test_invoice_edit.py tests/test_evox_products.py -v`
Expected: all PASS. In particular `test_price_inhouse_honors_overrides_service_and_pickup` (`tests/test_invoice_edit.py:22`) still expects `shipping_cents == 0` for a pickup — unchanged, since `effective_shipping_cents` is untouched.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_price_cart_shippable.py
git commit -m "fix(shipping): a service line no longer counts toward the box size"
```

---

### Task 3: The country check follows shippability

An overseas client buying only a Biofield Analysis currently gets "We ship to US addresses only" — on a service. That error exists because the check runs before we know whether anything ships.

**Files:**
- Modify: `app.py` lines 5350-5352 (the country guard in `_price_cart`)
- Test: `tests/test_price_cart_shippable.py` (append)

**Interfaces:**
- Consumes: `box_counts` as built in Task 2.
- Produces: no new symbols. `CheckoutError` is still raised for a shippable overseas cart.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_price_cart_shippable.py`:

```python
def test_overseas_services_only_cart_prices_without_error():
    """A service has no shipment, so a non-US address is irrelevant to it."""
    res = app._price_inhouse_invoice(
        [{"slug": "biofield-analysis", "qty": 1}],
        email="", pickup=False, ship={"name": "T", "country": "AU"})
    assert res is not None
    assert res["shipping_cents"] == 0


def test_overseas_cart_with_a_bottle_still_raises():
    """Guard: the US-only rule still holds for anything we actually mail."""
    with pytest.raises(app.CheckoutError):
        app._price_cart([{"slug": "neuro-magnesium", "qty": 1}],
                        ship={"name": "T", "country": "AU"}, channel="retail")
```

The second test calls `_price_cart` directly rather than `_price_inhouse_invoice`, because `_price_inhouse_invoice` catches and re-raises `CheckoutError` only after neutralizing the country for pickups (`app.py:32688`) — going straight to `_price_cart` tests the guard itself.

- [ ] **Step 2: Run test to verify it fails**

Run: `/tmp/dt.sh tests/test_price_cart_shippable.py -k overseas -v`
Expected: `test_overseas_services_only_cart_prices_without_error` FAILS with `CheckoutError: We ship to US addresses only`.
`test_overseas_cart_with_a_bottle_still_raises` PASSES already — it is the guard, not the change.

- [ ] **Step 3: Write minimal implementation**

In `app.py` `_price_cart`, currently:

```python
    country = (ship.get("country") or "US").strip().upper()
    if country not in ("US", "USA", ""):
        raise CheckoutError("We ship to US addresses only — please use a US forwarding address.")
    settings = _pricing.load_settings(_pricing_settings())
```

Keep the assignment where it is; move only the `raise` to after the cart loop. So:

```python
    country = (ship.get("country") or "US").strip().upper()
    settings = _pricing.load_settings(_pricing_settings())
```

and then, immediately after the `for c in (cart or []):` loop ends and **before** `priced = _pricing.compute(...)`:

```python
    # US-only shipping — but only a cart with something to ship has an opinion
    # about the address. An overseas client buying a service prices fine.
    if box_counts and country not in ("US", "USA", ""):
        raise CheckoutError("We ship to US addresses only — please use a US forwarding address.")
```

**Before editing, verify `country` is not read between its assignment and the loop's end.** Run:

```bash
sed -n '5348,5392p' app.py | grep -n country
```

Expected: exactly two hits — the assignment, and (after your edit) the new guard. If you see a third, stop and report it.

- [ ] **Step 4: Run test to verify it passes**

Run: `/tmp/dt.sh tests/test_price_cart_shippable.py -v`
Expected: PASS (5 tests)

Run: `/tmp/dt.sh tests/test_invoice_edit.py -v`
Expected: all PASS — including `test_price_inhouse_honors_overrides_service_and_pickup`, whose pickup path neutralizes country at `app.py:32688`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_price_cart_shippable.py
git commit -m "fix(shipping): US-only check applies only to carts with something to ship"
```

---

### Task 4: The catalog tells the browser what is shippable

**Files:**
- Modify: `dashboard/products.py` `catalog()` (lines 54-69)
- Test: `tests/test_products_catalog_shippable.py` (create)

**Interfaces:**
- Consumes: `dashboard.shipping.is_shippable` from Task 1.
- Produces: every dict returned by `catalog()` gains `"shippable": bool`. Task 7's JavaScript reads `p.shippable` off `/api/products?all=1`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_products_catalog_shippable.py`:

```python
"""catalog() exposes the same shippability answer the pricer uses."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_catalog_marks_services_unshippable():
    from dashboard.products import catalog
    rows = {r["slug"]: r for r in catalog(with_ingredients_only=False, include_inactive=True)}
    assert rows["biofield-analysis"]["shippable"] is False
    assert rows["evox-session"]["shippable"] is False


def test_catalog_marks_a_real_product_shippable():
    from dashboard.products import catalog
    rows = {r["slug"]: r for r in catalog(with_ingredients_only=False, include_inactive=True)}
    assert rows["neuro-magnesium"]["shippable"] is True


def test_every_catalog_row_has_shippable():
    from dashboard.products import catalog
    rows = catalog(with_ingredients_only=False, include_inactive=True)
    assert rows, "catalog must not be empty"
    assert all(isinstance(r.get("shippable"), bool) for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_products_catalog_shippable.py -v`
Expected: FAIL with `KeyError: 'shippable'` (or `assert None is False`)

- [ ] **Step 3: Write minimal implementation**

At the top of `dashboard/products.py`, add to the imports:

```python
from dashboard.shipping import is_shippable
```

(`dashboard/shipping.py` imports only `sqlite3`/`typing` at module scope, so there is no import cycle.)

Then in `catalog()`, inside the `out.append({...})` dict, after the `"service"` key:

```python
                    "service": bool(p.get("service")),
                    # The browser needs the same shippability answer the pricer uses,
                    # so the order builder can disable Pickup when nothing ships.
                    "shippable": is_shippable(p),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_products_catalog_shippable.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/products.py tests/test_products_catalog_shippable.py
git commit -m "feat(products): expose shippable on the catalog"
```

---

### Task 5: `client_prefs` — the per-client pickup default

**Files:**
- Create: `dashboard/client_prefs.py`
- Test: `tests/test_client_prefs.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `init_table(cx) -> None`
  - `set_pickup_default(cx, email: str, value: bool) -> None` — raises `ValueError` on empty email
  - `get_pickup_default(cx, email: str) -> bool` — `False` for unknown or empty email

  Task 6's endpoint imports this as `from dashboard import client_prefs as _cpf`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_client_prefs.py`:

```python
"""Per-client fulfillment preference. Mirrors the client_prices test shape."""
import sqlite3
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _cx():
    from dashboard import client_prefs as C
    cx = sqlite3.connect(":memory:")
    C.init_table(cx)
    return C, cx


def test_unset_client_defaults_to_no_pickup():
    C, cx = _cx()
    assert C.get_pickup_default(cx, "nobody@x.com") is False


def test_set_get_round_trip_and_is_case_insensitive():
    C, cx = _cx()
    C.set_pickup_default(cx, "Bobbi@X.com", True)
    assert C.get_pickup_default(cx, "bobbi@x.com") is True
    assert C.get_pickup_default(cx, "  BOBBI@x.com  ") is True


def test_set_is_idempotent_and_reversible():
    C, cx = _cx()
    C.set_pickup_default(cx, "bobbi@x.com", True)
    C.set_pickup_default(cx, "bobbi@x.com", True)     # upsert, not a second row
    assert cx.execute("SELECT COUNT(*) FROM client_prefs").fetchone()[0] == 1
    C.set_pickup_default(cx, "bobbi@x.com", False)    # explicit flip back
    assert C.get_pickup_default(cx, "bobbi@x.com") is False


def test_scoped_to_client():
    C, cx = _cx()
    C.set_pickup_default(cx, "bobbi@x.com", True)
    assert C.get_pickup_default(cx, "other@x.com") is False


def test_empty_email_is_rejected_on_write_and_false_on_read():
    C, cx = _cx()
    assert C.get_pickup_default(cx, "") is False
    assert C.get_pickup_default(cx, None) is False
    with pytest.raises(ValueError):
        C.set_pickup_default(cx, "  ", True)


def test_init_table_is_idempotent():
    C, cx = _cx()
    C.init_table(cx)  # second call must not raise
    C.set_pickup_default(cx, "a@x.com", True)
    assert C.get_pickup_default(cx, "a@x.com") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_client_prefs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.client_prefs'`

- [ ] **Step 3: Write minimal implementation**

Create `dashboard/client_prefs.py`:

```python
"""Persistent per-client fulfillment preferences. Today there is exactly one:
whether this client (by email) collects in person, so the order builder can
pre-check Pickup for them. Mirrors client_prices.py — pure functions over a
sqlite connection (testable).

Nothing writes this except an explicit operator toggle on the order builder.
Creating or saving an order NEVER writes it: ticking Pickup on one order is an
override for that order alone. That is why a Biofield hand-off invoice, which
sends pickup=True as a deliberate shipping courtesy, cannot silently teach the
system that every biofield client picks up.
"""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS client_prefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            pickup_default INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            UNIQUE(email)
        )
    """)
    cx.commit()


def set_pickup_default(cx, email, value):
    """Upsert this client's pickup default. Explicit operator action only."""
    email = _norm(email)
    if not email:
        raise ValueError("email required")
    cx.execute(
        "INSERT INTO client_prefs (email, pickup_default, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(email) DO UPDATE SET pickup_default=excluded.pickup_default, "
        "updated_at=excluded.updated_at",
        (email, 1 if value else 0, _now()))
    cx.commit()


def get_pickup_default(cx, email):
    """True when this client collects in person by default. Unknown -> False."""
    email = _norm(email)
    if not email:
        return False
    row = cx.execute("SELECT pickup_default FROM client_prefs WHERE email=?",
                     (email,)).fetchone()
    return bool(row[0]) if row else False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_client_prefs.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/client_prefs.py tests/test_client_prefs.py
git commit -m "feat(client-prefs): per-client pickup default store"
```

---

### Task 6: The one endpoint that writes it

**Files:**
- Modify: `app.py` — add a route directly above `api_console_client_prices` (currently at line 33104)
- Test: `tests/test_client_prefs.py` (append)

**Interfaces:**
- Consumes: `dashboard.client_prefs` from Task 5.
- Produces:
  - `GET /api/console/client-prefs?email=<e>` → `{ok, email, pickup_default}`
  - `POST /api/console/client-prefs` body `{email, pickup_default}` → `{ok, email, pickup_default}`

  Both owner-gated. Task 7's JavaScript calls both with the `X-Console-Key` header.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_client_prefs.py`. This is a source-level invariant test — it needs no secrets, and it is the guard on the whole design:

```python
def test_only_the_console_endpoint_writes_the_pickup_default():
    """The design's load-bearing promise: creating or saving an order never
    writes a client's pickup default. Exactly one call site in app.py may write
    it — the explicit console endpoint. If this count changes, an order path has
    almost certainly started persisting a per-order override as a preference."""
    src = (repo_root / "app.py").read_text()
    assert src.count("set_pickup_default") == 1


def test_the_order_builder_never_posts_a_pickup_default_with_an_order():
    """The order payloads carry `pickup` (per-order) and never `pickup_default`."""
    src = (repo_root / "static" / "order-new.html").read_text()
    for fn in ("async function createInvoice()", "async function editInvoice()"):
        start = src.index(fn)
        body = src[start:src.index("\n}", start)]
        assert "pickup_default" not in body, f"{fn} must not send pickup_default"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_client_prefs.py -k only_the_console -v`
Expected: FAIL — `assert 0 == 1` (no call site exists yet).

The second test PASSES already; it is a guard that must keep passing after Task 7.

- [ ] **Step 3: Write minimal implementation**

In `app.py`, immediately **above** `@app.route("/api/console/client-prices", ...)` at line 33104, add:

```python
@app.route("/api/console/client-prefs", methods=["GET", "POST"])
def api_console_client_prefs():
    """Owner: read/set a client's fulfillment defaults (today: pickup_default).
    GET ?email= -> {pickup_default}; POST {email, pickup_default} sets it.

    This is the ONLY writer. An order's Pickup tick is a per-order override and
    never lands here — a client's default changes when the owner says so, not as
    a side effect of one order being collected or mailed."""
    actor = _bos_actor()
    if actor is None or actor.role != _bos_rbac.OWNER:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import client_prefs as _cpf
    cx = _sqlite3.connect(LOG_DB)
    try:
        _cpf.init_table(cx)
        if request.method == "GET":
            email = (request.args.get("email") or "").strip().lower()
            if not email:
                return jsonify({"ok": False, "error": "email required"}), 400
            return jsonify({"ok": True, "email": email,
                            "pickup_default": _cpf.get_pickup_default(cx, email)})
        body = request.get_json(silent=True) or {}
        email = (body.get("email") or "").strip().lower()
        if not email:
            return jsonify({"ok": False, "error": "email required"}), 400
        _cpf.set_pickup_default(cx, email, bool(body.get("pickup_default")))
        return jsonify({"ok": True, "email": email,
                        "pickup_default": _cpf.get_pickup_default(cx, email)})
    finally:
        cx.close()
```

**Before committing, read `api_console_client_prices` (line 33104 onward) and match its connection-teardown idiom exactly** — if it closes `cx` differently (or not at all), follow the file, not this snippet.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_client_prefs.py -v`
Expected: PASS (8 tests)

Confirm the app still imports:
Run: `/tmp/dt.sh tests/test_invoice_edit.py -v`
Expected: all PASS (a syntax error or duplicate route name in `app.py` shows up here as a collection error).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_client_prefs.py
git commit -m "feat(client-prefs): owner-gated console endpoint (the only writer)"
```

---

### Task 7: The order builder

Two adjacent controls, and the distinction between them must be legible: the first says *this order*, the second says *from now on*.

**Files:**
- Modify: `static/order-new.html` — line 114 (Pickup markup), `loadCatalog` (161), `pickPerson` (188), `renderLines` (213), `loadOrderForEdit` (299), `init` (bottom)
- Test: `tests/test_order_new_pickup_ui.py` (create)

**Interfaces:**
- Consumes: `shippable` on `/api/products?all=1` rows (Task 4); `GET`/`POST /api/console/client-prefs` (Task 6).
- Produces: no server-side symbols.

- [ ] **Step 1: Write the failing test**

Create `tests/test_order_new_pickup_ui.py`:

```python
"""Static assertions on the order builder's pickup controls. No browser needed:
these pin the wiring that the design depends on."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SRC = (repo_root / "static" / "order-new.html").read_text()


def test_has_a_per_client_default_checkbox():
    assert 'id="pickup-default"' in SRC
    assert "Pickup by default for this client" in SRC


def test_the_default_checkbox_saves_on_its_own():
    """It POSTs immediately, not as part of the order payload."""
    assert "savePickupDefault()" in SRC
    assert "/api/console/client-prefs" in SRC


def test_pickup_is_disabled_when_nothing_is_shippable():
    assert "function syncShippingUI()" in SRC
    assert "No shipping — nothing physical in this order" in SRC


def test_edit_mode_lets_the_order_channel_win():
    """On edit, the stored order's channel decides Pickup — not the client default."""
    assert 'if (o.channel==="pickup") $("pickup").checked = true;' in SRC
    assert "!EDIT_OID" in SRC   # the guard inside loadPickupDefault
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_order_new_pickup_ui.py -v`
Expected: FAIL on the first three (`assert 'id="pickup-default"' in SRC`).

- [ ] **Step 3: Write minimal implementation**

**(a)** Replace line 114:

```html
    <div style="margin-top:10px"><label style="display:inline-flex;align-items:center;gap:6px;text-transform:none;font-size:14px;color:var(--text);cursor:pointer"><input type="checkbox" id="pickup"> Pickup (no shipping)</label></div>
```

with:

```html
    <div style="margin-top:10px"><label style="display:inline-flex;align-items:center;gap:6px;text-transform:none;font-size:14px;color:var(--text);cursor:pointer"><input type="checkbox" id="pickup"> <span id="pickup-label">Pickup (no shipping)</span></label></div>
    <div id="pickup-default-wrap" style="display:none;margin-top:6px;margin-left:22px">
      <label style="display:inline-flex;align-items:center;gap:6px;text-transform:none;font-size:13px;color:var(--muted);cursor:pointer"><input type="checkbox" id="pickup-default" onchange="savePickupDefault()"> Pickup by default for this client</label>
    </div>
```

**(b)** Add the shippability helpers. Put them just above `function renderLines()` (line 213):

```javascript
// ── shipping visibility ──
// "Nothing to ship" is derived from the cart, never stored: a services-only order
// has no shipping because it has no boxes, not because anyone ticked a box.
function hasShippableLine(){
  return LINES.some(l => { const p = CATALOG.find(x => x.slug===l.slug); return p ? p.shippable !== false : true; });
}
function syncShippingUI(){
  const phys = hasShippableLine(), pk = $("pickup");
  if (!phys){
    pk.checked = false; pk.disabled = true;
    $("pickup-label").textContent = "No shipping — nothing physical in this order";
    $("t-ship").textContent = dollars(0);
  } else {
    pk.disabled = false;
    $("pickup-label").textContent = "Pickup (no shipping)";
  }
  // An order with nothing to ship represents no fulfillment decision, so it
  // cannot teach us one — hide the client default entirely.
  $("pickup-default-wrap").style.display = (phys && $("c-email").value.trim()) ? "block" : "none";
}
async function loadPickupDefault(email){
  email = (email||"").trim();
  if (!email){ $("pickup-default").checked = false; syncShippingUI(); return; }
  try{
    const r = await fetch("/api/console/client-prefs?email="+encodeURIComponent(email), {headers:HEADERS});
    const j = await r.json(); if (!j.ok) return;
    $("pickup-default").checked = !!j.pickup_default;
    // On a NEW order the client's default pre-checks Pickup (still overridable).
    // On an EDIT the stored order's own channel already won — don't fight it.
    if (!EDIT_OID && j.pickup_default && hasShippableLine()) $("pickup").checked = true;
  }catch(e){ /* a missing preference is just "no default" */ }
  syncShippingUI();
}
async function savePickupDefault(){
  const email = $("c-email").value.trim();
  if (!email){ toast("Pick a customer first","error"); return; }
  try{
    const r = await fetch("/api/console/client-prefs", {method:"POST", headers:HEADERS,
      body: JSON.stringify({email, pickup_default: $("pickup-default").checked})});
    const j = await r.json();
    if (!j.ok){ toast(j.error||"failed","error"); return; }
    toast(j.pickup_default ? "Pickup is now this client's default" : "Pickup default cleared");
  }catch(e){ toast(e.message,"error"); }
}
```

**(c)** In `renderLines()` (line 213), add `syncShippingUI();` after `recalc();`:

```javascript
  refreshPreview();
  recalc();
  syncShippingUI();
```

**(d)** In `pickPerson(p)` (line 188), add as the last line of the function, after `recalc();`:

```javascript
  loadPickupDefault(p.email||"");
```

**(e)** In `loadOrderForEdit(oid)` (line 299), leave line 327 exactly as it is (`if (o.channel==="pickup") $("pickup").checked = true;`) and add after `renderLines();` at line 329:

```javascript
  loadPickupDefault(o.email||"");   // shows the client's default; EDIT_OID keeps it from overriding this order
```

**(f)** In `init()` at the bottom, add `syncShippingUI();` as the last line, so a fresh empty builder starts in a coherent state:

```javascript
async function init(){
  if (!KEY){ $("auth-warn").style.display="block"; return; }
  try{ await loadCatalog(); }catch(e){ if(e.message!=="unauthorized") toast(e.message,"error"); return; }
  if (EDIT_OID){ try{ await loadOrderForEdit(EDIT_OID); }catch(e){ toast(e.message,"error"); } }
  syncShippingUI();
}
```

**Do not change `createInvoice()` or `editInvoice()`.** Their payloads keep sending `pickup` and must never send `pickup_default` — `test_the_order_builder_never_posts_a_pickup_default_with_an_order` from Task 6 enforces this.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_order_new_pickup_ui.py tests/test_client_prefs.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Verify it in a browser, not just in pytest**

These are static-source assertions; they cannot prove the page works. Start the app and drive it:

```bash
/tmp/dt.sh --version >/dev/null 2>&1 || true
printf '#!/bin/bash\ndoppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 app.py\n' > /tmp/run.sh && chmod +x /tmp/run.sh
```

Run `/tmp/run.sh` in the background, open `/static/order-new.html?key=<CONSOLE_SECRET>`, and confirm all four:

1. Empty order → Pickup is disabled and reads "No shipping — nothing physical in this order".
2. Add **Biofield Analysis** only → still disabled; Shipping shows `$0.00`; the client-default checkbox is hidden.
3. Add **Neuro Magnesium** → Pickup enables; the label reverts; picking a customer reveals "Pickup by default for this client".
4. Tick the default, reload the page, pick the same customer → Pickup comes back pre-checked. Untick Pickup (not the default), create the order, then start a new order for that customer → Pickup is pre-checked again, because the per-order untick wrote nothing.

Step 4 is the whole design in one gesture. If it fails, the feature is wrong no matter what pytest says.

- [ ] **Step 6: Commit**

```bash
git add static/order-new.html tests/test_order_new_pickup_ui.py
git commit -m "feat(order-new): derive no-shipping in the UI + per-client pickup default"
```

---

### Task 8: Record the biofield courtesy

`pickup: True` on a hand-off invoice suppresses shipping on the remedy bottles as well as the fee. That is intentional. Without a comment, the next person to read the `is_shippable` fix will "clean it up" and start billing shipping on those invoices.

**Files:**
- Modify: `dashboard/biofield_invoice.py` line 119
- Test: `tests/test_biofield_invoice_courtesy.py` (create)

**Interfaces:**
- Consumes: nothing. Produces: nothing. Behavior is unchanged by design.

- [ ] **Step 1: Write the failing test**

Create `tests/test_biofield_invoice_courtesy.py`:

```python
"""The hand-off invoice ships free on purpose. Pin it so the derived-shipping
fix doesn't get "completed" by deleting the flag."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SRC = (repo_root / "dashboard" / "biofield_invoice.py").read_text()


def test_hand_off_invoice_still_sends_pickup_true():
    assert '"pickup": True' in SRC
```

We test the behavior (the flag is sent), not the comment. The comment is still
required by Step 3 — it is what stops a future reader from deleting the flag —
but prose is not a test subject.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_biofield_invoice_courtesy.py -v`
Expected: PASS already. This task's test is a **characterization test**: it pins
behavior that must not change while Tasks 1-3 remove the bug that made the flag
look accidental. It has no red phase, and that is correct — write it, watch it
pass, and it will fail the day someone deletes the flag.

- [ ] **Step 3: Write minimal implementation**

In `dashboard/biofield_invoice.py`, inside `default_create_order`, above the `body = {...}` at line 118:

```python
    # pickup=True is a deliberate shipping COURTESY, absorbed into the analysis
    # fee: the hand-off invoice carries real remedy bottles that Rae mails, and
    # we choose not to bill USPS on top of the $300. It is NOT a workaround for
    # the old bug where a service line inflated the box count — that is fixed in
    # shipping.is_shippable. Removing this flag starts charging these clients
    # shipping. Don't, without asking Glen.
    body = {"customer": {"name": customer.get("name") or "", "email": customer.get("email") or ""},
            "lines": lines, "pickup": True, "replace_open": bool(replace_open),
            "invoice_note": "Biofield Analysis and remedies. Payable by check."}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_biofield_invoice_courtesy.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Full suite, then commit**

```bash
python3 -m pytest tests/ -q
/tmp/dt.sh tests/test_invoice_edit.py tests/test_evox_products.py tests/test_price_cart_shippable.py tests/test_combined_shipments.py tests/test_orders_effective_shipping.py -v
```

Expected: no failures. Anything red here is a regression introduced by Tasks 1-7 — fix it before committing.

```bash
git add dashboard/biofield_invoice.py tests/test_biofield_invoice_courtesy.py
git commit -m "docs(biofield): record that hand-off invoices ship free on purpose"
```

---

## Done when

- A services-only order quotes `$0.00` shipping with nothing ticked.
- Four bottles cost the same to ship with or without a Biofield Analysis on the invoice.
- An overseas client can be invoiced for a Biofield Analysis.
- Ticking "Pickup by default for this client" pre-checks Pickup on that client's next order; unticking Pickup on one order changes nothing for the next one.
- Biofield hand-off invoices still ship free, and now say why.
