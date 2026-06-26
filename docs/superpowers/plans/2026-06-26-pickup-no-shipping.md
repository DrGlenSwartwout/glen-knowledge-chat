# Pickup (No Shipping) Option — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Pickup (no shipping)" option to in-house order entry so a local-pickup order is created with `shipping_cents = 0` and `channel='pickup'`.

**Architecture:** A pure helper `effective_shipping_cents(pickup, computed)` in `dashboard/orders.py` is the single shipping-zero rule; `/api/orders/manual` (create) and `/api/invoice/<token>/update` (edit) call it; `static/order-new.html` adds a checkbox that posts `pickup:true`. No schema change (reuse the `channel` column + stored `shipping_cents`).

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest.

## Global Constraints

- No DB schema change — pickup is persisted as `channel='pickup'` (column already exists, default `'retail'`); shipping stored in the existing `shipping_cents` column.
- Shipping is zeroed ONLY at the two in-house compute sites (create + invoice-update); funnel/subscription/cron checkouts are untouched.
- `effective_shipping_cents(pickup, computed) -> int`: `0` if `pickup` truthy else `int(computed or 0)`.
- `app.py` + `order-new.html` cannot be imported offline (Pinecone at import) → Tasks 2 & 3 are verified LIVE post-deploy with the exact checks in their steps. Task 1 (helper) is offline-TDD.
- Offline test cmd: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v`.

---

### Task 1: `effective_shipping_cents` helper

**Files:**
- Modify: `dashboard/orders.py` (append at end)
- Test: `tests/test_orders_effective_shipping.py`

**Interfaces:**
- Produces: `effective_shipping_cents(pickup, computed_cents) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orders_effective_shipping.py
from dashboard.orders import effective_shipping_cents

def test_pickup_zeroes_shipping():
    assert effective_shipping_cents(True, 1299) == 0
    assert effective_shipping_cents(True, 0) == 0
    assert effective_shipping_cents("pickup", 999) == 0   # any truthy

def test_non_pickup_passes_through():
    assert effective_shipping_cents(False, 1299) == 1299
    assert effective_shipping_cents(False, None) == 0
    assert effective_shipping_cents(False, "0") == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_orders_effective_shipping.py -v`
Expected: FAIL — `ImportError: cannot import name 'effective_shipping_cents'`.

- [ ] **Step 3: Implement** — append to `dashboard/orders.py`:

```python
def effective_shipping_cents(pickup, computed_cents):
    """Shipping for an order: 0 when it's a pickup (no shipping), else the
    computed amount. Single source of the pickup-shipping rule."""
    if pickup:
        return 0
    try:
        return int(computed_cents or 0)
    except (TypeError, ValueError):
        return 0
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_orders_effective_shipping.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/orders.py tests/test_orders_effective_shipping.py
git commit -m "feat(pickup): effective_shipping_cents helper (0 on pickup)"
```

---

### Task 2: Honor pickup in the two in-house pricing routes

**Files:**
- Modify: `app.py` — `/api/orders/manual` (the `pc = _price_cart(...)` block at ~24237 + the `upsert_order(...)` call in the same route) and `/api/invoice/<token>/update` (the `shipping_cents = int(pc.get("shipping_cents") or 0)` at ~24439).

**Interfaces:**
- Consumes: `effective_shipping_cents` (Task 1). The orders module is imported in app.py as `_bos_orders`/`_ord` — use the existing alias (`_bos_orders` is used for `set_order_payment`, `upsert_order`).

**Why no offline test:** `app.py` can't import offline; verified live (Step 4).

- [ ] **Step 1: `/api/orders/manual` — read pickup, zero shipping, set channel**

Near the top of the GET-less POST body of `api_orders_manual` (after `lines_in` is read), add:
```python
    pickup = bool(body.get("pickup"))
```
Change the shipping assignment (currently `shipping_cents = int(pc.get("shipping_cents") or 0)` inside the `try` at ~24238) to:
```python
        shipping_cents = _bos_orders.effective_shipping_cents(pickup, pc.get("shipping_cents"))
```
(The `except` fallback `shipping_cents, get_cents = 0, 0` stays — already 0.)
Then, in the `upsert_order(...)` call in this route, pass the channel: add the argument
```python
        channel=("pickup" if pickup else "retail"),
```
(If `upsert_order` is already called with a `channel=`, replace its value with this expression. Confirm `_bos_orders` is the alias app.py uses for `dashboard.orders`; if it's `_ord`, use that.)

- [ ] **Step 2: `/api/invoice/<token>/update` — keep pickup orders at 0 shipping**

This route loads `order` by token. Change its shipping line (currently `shipping_cents = int(pc.get("shipping_cents") or 0)` at ~24439) to:
```python
        shipping_cents = _bos_orders.effective_shipping_cents(
            (order.get("channel") or "") == "pickup", pc.get("shipping_cents"))
```

- [ ] **Step 3: Parse-check + commit**

```bash
~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
git add app.py
git commit -m "feat(pickup): zero shipping + channel=pickup in order-create and invoice-update"
```

- [ ] **Step 4: Live verification (post-deploy — record commands in report)**

```bash
# Create a pickup order (console key) — expect shipping_cents 0 + channel pickup:
doppler run -p remedy-match -c prd -- sh -c 'curl -s -X POST "https://glen-knowledge-chat.onrender.com/api/orders/manual" -H "X-Console-Key: $CONSOLE_SECRET" -H "Content-Type: application/json" -d "{\"email\":\"pickuptest@example.com\",\"name\":\"Pickup Test\",\"lines\":[{\"slug\":\"vitality\",\"qty\":1,\"unit_cents\":5000}],\"pickup\":true}"'
# Then GET /api/orders and confirm that order shows shipping_cents 0 + channel "pickup".
# Control: same POST WITHOUT pickup -> shipping_cents > 0.
```

---

### Task 3: "Pickup (no shipping)" checkbox in order entry

**Files:**
- Modify: `static/order-new.html`

**Interfaces:**
- Consumes: the `/api/orders/manual` `pickup` field (Task 2).

**Read first:** `static/order-new.html` — find where it builds the `/api/orders/manual` POST body (the fetch call) and where form fields are laid out, to match the file's markup/JS style.

- [ ] **Step 1: Add the checkbox + wire it**

In `static/order-new.html`, add a checkbox near the order options (e.g. by the discount field):
```html
<label><input type="checkbox" id="pickup"> Pickup (no shipping)</label>
```
In the JS that assembles the `/api/orders/manual` request body, include:
```javascript
pickup: document.getElementById('pickup').checked,
```
(Match the file's existing body-building style — if it builds an object literal, add the key there; if it uses `body.pickup = ...`, do that.)

- [ ] **Step 2: Commit**

```bash
git add static/order-new.html
git commit -m "feat(pickup): pickup checkbox in in-house order entry"
```

- [ ] **Step 3: Live verification (post-deploy — record in report)**

Load `/order-new` (console-keyed), check "Pickup (no shipping)", create an order, and confirm on `/console/orders` that the created order has $0 shipping. Uncheck → normal shipping.

---

## Self-Review

**1. Spec coverage:**
- `effective_shipping_cents` helper → Task 1. ✅
- Create route zeroes shipping + sets channel=pickup → Task 2 Step 1. ✅
- Invoice-update keeps pickup at 0 → Task 2 Step 2. ✅
- UI checkbox posts pickup → Task 3. ✅
- No schema change (reuse channel) → Task 2 uses existing `upsert_order(channel=...)`. ✅
- Live verification for the two app.py routes + UI → Task 2 Step 4, Task 3 Step 3. ✅

**2. Placeholder scan:** No TBD/TODO. Task 2/3 "match the existing alias/markup" notes point at concrete files to read, with exact line anchors given. ✅

**3. Type consistency:** `effective_shipping_cents(pickup, computed) -> int` used identically in Task 1 (def) and Task 2 (both call sites). The `pickup` body field (Task 2 reads `body.get("pickup")`) matches what Task 3 posts (`pickup: ...`). Channel value `"pickup"` consistent between create (set) and invoice-update (check). ✅
