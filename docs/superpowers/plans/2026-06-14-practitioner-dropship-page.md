# Practitioner-Paid Drop-Ship Page — Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A practitioner builds a drop-ship order **for a specific patient** — picks Functional Formulations + quantities, enters the **patient's shipping address**, pays the **drop-ship wholesale** (blended base + 33% service fee), and we ship to the patient. The practitioner bills the patient privately (their margin is off-platform). New portal page + endpoints; reuses the Plan 1 pricing core and the existing wholesale-checkout machinery.

**Architecture:** A new `dashboard/dropship_checkout.py` (`build_dropship_order`) mirroring `wholesale_checkout.build_order` but pricing each line at `base + 33%×(retail − base)` and shipping to the **patient**. New authed routes `/api/practitioner/dropship/quote` + `/api/practitioner/dropship/checkout`. A new portal page `static/practitioner-dropship.html`. Reuses Plan 1's `practitioner_pricing` (`quote_line`), the existing practitioner session/cart/wallet, QBO, Stripe, and the customer engine's shipping.

**DESIGN DECISION (flag for Glen):** practitioner-paid mode bases the fee on **retail**, not a declared patient price — `fee = 33% × (retail − base)` per bottle (RM's standard cut; RM can't see the private patient transaction, so this is ungameable). The practitioner enters **no selling price** here; their patient margin is private. (If Glen prefers a declared selling price with a retail floor, it's a small change to Task 1.)

**Tech Stack:** Python 3.11, Flask, QBO, Stripe, pytest. Route tests need full env (`doppler … pytest`); the pricing function can be unit-tested with stubs.

**Reuse (confirmed):**
- `practitioner_pricing.drop_ship_base_cents(qty, modules)`, `.service_fee_cents(selling, base, settings)`, `.load_settings`, `.quote_line` (Plan 1, in main).
- `wholesale_pricing.order_quote(items, practitioner)` → `{lines:[{slug,qty,...}], blended_unit_price_cents, total_bottles, subtotal_cents, ...}` and `wholesale_checkout.build_order` (the pattern to mirror: QBO customer/invoice, wallet redeem ≤50%, fee-free 3% earn, returns ok/invoice_id/total/customer_id).
- `app.py`: `_practitioner_session_pid()`, `_pp.portal_data(pid)` (has modules_completed, email, name, wholesale_unlocked, cart), `_pp.cart_set/cart_clear`, `_normalize_ship_address`, `_ingest_order`, `_stripe_checkout_url_for_order`, `_ALT_PAY`, `_STRIPE_ACTIVE`, `_shipping_for_cart` + `_shipping_line` (Plan-2-customer shipping), product retail via `_get_product(slug)["price_cents"]`.
- Products have `bottle_type` (for shipping) and `price_cents` (retail). FF only (Pure Powders excluded from practitioner pricing).

---

### Task 1: `build_dropship_order` — price at drop-ship wholesale, ship to patient

**Files:** Create `dashboard/dropship_checkout.py`; Test `tests/test_dropship_checkout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dropship_checkout.py
from dashboard import dropship_checkout as dc

def test_dropship_unit_price_is_base_plus_retail_fee():
    # base from blended curve; fee = 33% of (retail - base); drop-ship unit = base + fee
    # 1 bottle, uncertified: base $50.00, retail $70.00 -> fee 33%*(7000-5000)=660 -> unit 5660
    line = dc.dropship_line_cents(retail_cents=7000, qty=1, modules=0,
                                  settings=dc._settings())
    assert line["base_cents"] == 5000
    assert line["fee_cents"] == 660
    assert line["unit_cents"] == 5660          # what the practitioner pays per bottle
    assert line["line_cents"] == 5660          # x qty 1

def test_dropship_unit_uses_blended_volume_and_cert():
    # 12 bottles, fully certified: base $42.76, retail $70 -> fee 33%*(7000-4276)=899 -> 5175
    line = dc.dropship_line_cents(retail_cents=7000, qty=12, modules=12,
                                  settings=dc._settings())
    assert line["base_cents"] == 4276
    assert line["fee_cents"] == 899
    assert line["unit_cents"] == 5175
    assert line["line_cents"] == 5175 * 12

def test_dropship_fee_zero_when_retail_equals_base():
    line = dc.dropship_line_cents(retail_cents=5000, qty=1, modules=0, settings=dc._settings())
    assert line["fee_cents"] == 0 and line["unit_cents"] == 5000
```

- [ ] **Step 2: Run → fail** (`tests/test_dropship_checkout.py` / module missing).

- [ ] **Step 3: Implement the pricing helper**

```python
# dashboard/dropship_checkout.py
"""Practitioner-paid drop-ship: price each line at the drop-ship wholesale
(blended base + 33% of the RETAIL markup), invoice the practitioner, ship to the
patient. The practitioner bills the patient privately (margin off-platform)."""
from dashboard import practitioner_pricing as _pp


def _settings():
    # Drop-ship pricing settings; pairs with the pending console editor.
    return _pp.load_settings({})


def dropship_line_cents(*, retail_cents, qty, modules, settings):
    """Per-line drop-ship economics. Fee is 33% of (retail - base) — RM's standard cut,
    since the patient price is private in practitioner-paid mode. Reuses Plan 1's
    quote_line with selling=retail."""
    q = _pp.quote_line(selling_cents=int(retail_cents), qty=int(qty),
                       modules=int(modules), settings=settings)
    unit = q["dropship_wholesale_cents"]          # base + fee
    return {
        "base_cents": q["base_cents"],
        "fee_cents": q["fee_cents"],
        "unit_cents": unit,
        "line_cents": unit * int(qty),
    }
```

- [ ] **Step 4: Run → pass (3).**

- [ ] **Step 5: Write the failing test for `build_dropship_order`** (stub QBO + wallet + product lookup)

```python
def test_build_dropship_order_invoices_practitioner_ships_patient(monkeypatch):
    cart = [{"slug": "brain-boost", "qty": 6}]
    prac = {"id": "p1", "modules_completed": 0, "email": "doc@x.com", "name": "Doc"}
    patient_ship = {"name": "Pat", "state": "CA", "country": "US", "address1": "1 St"}
    # stubs
    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    cap = {}
    monkeypatch.setattr(dc.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(dc.qb, "create_invoice",
        lambda cust, lines, **k: cap.update(lines=lines, kw=k) or {"Id": "INV", "TotalAmt": 339.60})
    out = dc.build_dropship_order(cart, prac, patient_ship=patient_ship, method="zelle")
    assert out["ok"] is True
    assert out["ship_to"]["name"] == "Pat"            # ships to the PATIENT
    assert out["source"] == "dropship"
    # 6 bottles uncertified: base 4970? -> recompute via dropship_line_cents; invoice carries unit
    assert cap["lines"][0]["qty"] == 6
```

- [ ] **Step 6: Run → fail.**

- [ ] **Step 7: Implement `build_dropship_order`** — READ `dashboard/wholesale_checkout.build_order` and mirror its QBO customer + invoice + return shape. Differences: (a) price each line via `dropship_line_cents` (per the order's total bottles for the blended base) at unit = base+fee; (b) `find_or_create_customer` is the PRACTITIONER (they pay), but the invoice + order ship-to is the **patient**; (c) `source="dropship"`; (d) allow wallet redeem ≤50% + fee-free 3% earn exactly like `build_order` (same practitioner-purchase mechanics); (e) GET recorded-not-charged on the patient ship-to state; (f) return includes `ship_to` (patient), `customer_id` (practitioner), `invoice_id`, `total`, `method`. Add `_retail_for(slug)` reading `_get_product(slug)["price_cents"]` (injected/monkeypatchable). NO MAP (private). NO selling-price input.

- [ ] **Step 8: Run → pass.**
- [ ] **Step 9: Commit** — `feat(dropship): build_dropship_order (drop-ship wholesale, ship-to-patient)`

---

### Task 2: routes — quote + checkout

**Files:** Modify `app.py`; Test `tests/test_dropship_routes.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_dropship_routes.py
import app as appmod

def _auth(monkeypatch):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "p1")
    monkeypatch.setattr(appmod._pp, "portal_data",
        lambda pid: {"modules_completed": 0, "email": "doc@x.com", "name": "Doc",
                     "wholesale_unlocked": True, "cart": [{"slug": "brain-boost", "qty": 6}]})

def test_dropship_quote_requires_auth(monkeypatch):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: None)
    assert appmod.app.test_client().post("/api/practitioner/dropship/quote", json={}).status_code == 401

def test_dropship_checkout_ships_to_patient(monkeypatch):
    _auth(monkeypatch)
    monkeypatch.setattr(appmod._dropship, "build_dropship_order",
        lambda *a, **k: {"ok": True, "invoice_id": "INV", "total": 339.60, "customer_id": "C1",
                         "source": "dropship", "ship_to": k.get("patient_ship"), "method": "zelle",
                         "get_cents": 0})
    cap = {}
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: cap.update(kw))
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)
    r = appmod.app.test_client().post("/api/practitioner/dropship/checkout",
        json={"method": "zelle", "patient_address": {"name": "Pat", "state": "CA", "country": "US"}})
    assert r.status_code == 200
    assert cap["source"] == "dropship"
    assert cap["address"]["name"] == "Pat"
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — mirror `/api/practitioner/checkout`. `GET/POST /api/practitioner/dropship/quote` returns per-line drop-ship pricing for the cart (using `dropship_line_cents`) + total. `POST /api/practitioner/dropship/checkout`: authed; require `wholesale_unlocked`; read `patient_address` (required — `_normalize_ship_address`); call `_dropship.build_dropship_order(cart, prac, patient_ship=ship, method=...)`; on ok → `_pp.cart_clear`, `_ingest_order(source="dropship", address=patient_ship, channel="wholesale", get_cents=..., ...)`, alt-pay or `_stripe_checkout_url_for_order`. Import `dropship_checkout as _dropship` locally per convention.

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(dropship): /api/practitioner/dropship quote + checkout (ship to patient)`

---

### Task 3: portal page

**Files:** Create `static/practitioner-dropship.html`; Modify `app.py` (a `/practitioner/dropship` route serving it)

- [ ] **Step 1:** Add `GET /practitioner/dropship` serving the static page (mirror how the other practitioner portal pages are served). Build `static/practitioner-dropship.html` mirroring the existing practitioner portal styling: a cart of FF products + qty steppers, a **patient shipping address** form, a live **drop-ship wholesale total** (from `/api/practitioner/dropship/quote`), and a Pay action (zelle/wise/card) posting to `/api/practitioner/dropship/checkout`. Make clear the practitioner pays this and bills the patient separately. No selling-price field. Degrade gracefully if not signed in (link to portal sign-in).
- [ ] **Step 2:** Manual/static verification: valid HTML/JS; the POST body carries `patient_address` + `method`; quote refreshes on cart/qty change. Confirm `tests/test_dropship_routes.py` still green.
- [ ] **Step 3:** Commit — `feat(dropship): practitioner drop-ship portal page`

---

### Task 4: suite + doc

- [ ] **Step 1:** Run `tests/test_dropship_checkout.py tests/test_dropship_routes.py` + the existing `tests/test_wallet*.py tests/test_practitioner_pricing.py` — all green.
- [ ] **Step 2:** Create `docs/dropship.md`: the practitioner-paid drop-ship flow (pay base+fee, fee = 33% of retail markup, ship to patient, private patient billing, no MAP, wallet redeem/earn like wholesale), routes, and that the patient-paid client page is Plan 3.
- [ ] **Step 3:** Commit.

---

## Self-review
- **Spec coverage:** §B.2 practitioner-paid drop-ship page (all tasks); §A.3 practitioner-paid economics (T1, with the retail-fee design decision flagged); ship-to-patient + source=dropship (T1-2); reuses Plan 1 quote_line + existing wholesale-checkout/ wallet/QBO/Stripe.
- **Deferred:** the patient-paid **client page** (Plan 3) — that's where MAP + `earn_dropship_margin` live; white-label branding (Plan 4). This page is logo/name-branded minimally in Plan 4.
- **Risk:** practitioner-purchase money path → mirror `build_order` exactly (wallet redeem, GET-recorded-not-charged, Stripe). Patient address required + US-only (reuse the customer-engine ship-to validation). No live customer-checkout impact.
- **Type consistency:** `dropship_line_cents(*, retail_cents, qty, modules, settings) -> dict`, `build_dropship_order(cart, prac, *, patient_ship, method) -> dict`, routes `/api/practitioner/dropship/{quote,checkout}`.

## Next
Plan 3 — branded client page on `/dropship`/`/dispensary/<code>` (patient-paid: patient pays the practitioner's MAP-enforced price, margin → `earn_dropship_margin`). Plan 4 — white-label settings + branding across all three pages.
