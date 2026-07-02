# Pricing Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make product quantity/volume-discount pricing one single engine everywhere, open to everyone (no members-only gate), console-tunable, and repriced to a base ~$70 → floor ~$50 smooth-anchor curve; retire the obsolete member-price upsell and $1-trial-credit machinery.

**Architecture:**
- **One pricing curve** already exists: `dashboard/pricing.py::volume_pct` (linear interpolation through `volume_anchors`) consumed two ways that share that curve: (1) `app._price_cart` → `pricing.compute` (storefront checkout, reorder-engine) keeps **list** prices on QBO lines + a separate `discount_cents`; (2) `app._inhouse_ff_unit_cents` → `pricing.volume_pct` (in-house order entry, price-preview, portal reorder) bakes the **effective** per-unit price per line.
- The odd one out is the **legacy step-tier system** `_QTY_TIERS`/`_qty_unit_cents` (hardcoded, paid-members-only). We retire it and route its live charged callers onto the engine.
- The curve is console-tunable today via `POST /api/console/pricing-settings` + `static/console-pricing-settings.html` anchor editor; no new infra needed.

**Tech Stack:** Flask monolith (`app.py`), pure helper modules under `dashboard/`, SQLite (`LOG_DB`), pytest. Tests run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/<file> -q`.

## Global Constraints
- New default anchors: `[[1,0],[2,14],[4,21],[7,26],[12,29]]`.
- `months_per_unit` is uniformly `1` across all products in `data/products.json` (verified); no large-format 90/180/360-cap SKUs exist yet → keep the engine keyed on `total_months` (`dashboard/pricing.py:91`); do NOT add bottle-count keying.
- Do NOT move base price into settings — it stays per-product in `data/products.json` (`6997`); only curve/floors live in settings.
- Keep `_is_paid_member` (`app.py:4616`) and `_active_membership_for_email` as functions — remove only their **pricing** uses (they stay for non-pricing access gating).
- Every task ends with all touched test files green under the doppler command above.

### Verified curve cents against the $69.97 list (engine math, confirmed by running `volume_pct`)
| total_months | vol_pct | effective on 6997 |
|---|---|---|
| 1 | 0.00% | 6997 |
| 2 | 14.00% | 6017 |
| 3 | 17.50% | 5773 |
| 4 | 21.00% | 5528 |
| 6 | 24.33% | 5294 |
| 7 | 26.00% | 5178 |
| 12+ (flat beyond) | 29.00% | 4968 |

Discount floor `round(6997*0.57)=3988` never binds (min effective 4968 > 3988).

## File Structure
```
dashboard/pricing.py                 # MODIFY: DEFAULTS.volume_anchors
dashboard/trial_credit.py            # DELETE
app.py                               # MODIFY: retire legacy qty system, open gate, retire credit machinery, route callers
tests/test_pricing_engine.py         # MODIFY: new-curve asserts + cents-verification test
tests/test_price_cart.py             # MODIFY: new-curve numbers
tests/test_begin_checkout_engine.py  # MODIFY: new-curve numbers
tests/test_reorder_checkout_engine.py# MODIFY: new-curve numbers
tests/test_inhouse_volume_pricing.py # MODIFY: new-curve numbers + drop member kwarg
tests/test_paid_member_gate.py       # MODIFY: drop the two preview tests (open-to-all)
tests/test_invoice_member_credit.py  # DELETE (feature retired)
tests/test_portal_trial_credit_display.py # MODIFY: trial_credit_cents now always 0
tests/test_trial_credit.py           # DELETE
tests/test_trial_credit_grant.py     # DELETE
tests/test_reorder_cart.py           # MODIFY (engine path unconditional)
tests/test_product_data_volume_tiers.py # CREATE (Task 5)
```

---

## Task 1 — Reprice the default volume curve to base $70 → floor $50

**Files**
- Modify `dashboard/pricing.py:14` (`DEFAULTS["volume_anchors"]`)
- Modify `tests/test_pricing_engine.py` (lines 14, 75–90, 133 + add cents test)
- Modify `tests/test_price_cart.py:11-18`, `tests/test_begin_checkout_engine.py:31-33`, `tests/test_reorder_checkout_engine.py` (the `8209` asserts), `tests/test_inhouse_volume_pricing.py` (member=True numbers only; keep signature for now)

**Interfaces**
- Consumes/Produces: `pricing.volume_pct(months:int, settings:dict) -> float`, `pricing.load_settings(overrides) -> dict` (unchanged signatures). Only `DEFAULTS["volume_anchors"]` value changes from `[[1, 0], [12, 43]]` to `[[1, 0], [2, 14], [4, 21], [7, 26], [12, 29]]`.

**Steps**
- [ ] Update the pure-engine asserts. In `tests/test_pricing_engine.py` replace line 14 with `assert s["volume_anchors"] == [[1, 0], [2, 14], [4, 21], [7, 26], [12, 29]]`. Replace `test_volume_pct_at_anchors` / `test_volume_pct_interpolates_and_caps`:
```python
def test_volume_pct_at_anchors():
    s = pricing.load_settings({})
    assert pricing.volume_pct(1, s) == 0
    assert pricing.volume_pct(2, s) == 14
    assert pricing.volume_pct(4, s) == 21
    assert pricing.volume_pct(7, s) == 26
    assert pricing.volume_pct(12, s) == 29
    assert pricing.volume_pct(3, s) == pytest.approx(17.5)
    assert pricing.volume_pct(6, s) == pytest.approx(21 + 5 * 2 / 3)

def test_volume_pct_interpolates_and_caps():
    s = pricing.load_settings({})
    assert pricing.volume_pct(5, s) == pytest.approx(21 + 5 * 1 / 3)
    assert pricing.volume_pct(24, s) == 29
    assert pricing.volume_pct(0, s) == 0
```
  Replace the two `5632` asserts in `test_compute_volume_mix_and_match_beats_subscriber` (lines 133-134) with `5297` (`round(7000*(1-0.243333))`). Add:
```python
def test_curve_lands_target_cents_on_6997_list():
    s = pricing.load_settings({})
    L = 6997
    expected = {1: 6997, 2: 6017, 4: 5528, 7: 5178, 12: 4968, 99: 4968}
    for m, cents in expected.items():
        pct = pricing.volume_pct(m, s)
        assert int(round(L * (1 - pct / 100.0))) == cents
    assert pricing.unit_floor_cents({"slug": "ff", "price_cents": L}, L, s, "discount") == 3988
```
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_pricing_engine.py -q` → **FAIL** (DEFAULTS still `[[1,0],[12,43]]`).
- [ ] Implement: in `dashboard/pricing.py` change line 14 to `"volume_anchors": [[1, 0], [2, 14], [4, 21], [7, 26], [12, 29]],` and update the comment block above (lines 11-13) to describe the smooth base-$70→floor-$50 curve.
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_pricing_engine.py tests/test_pricing_settings.py -q` → **PASS** (`test_pricing_settings` reads DEFAULTS via `defaults_view`; its `[[1,0],[3,14]...]` payloads are validation fixtures, not defaults).
- [ ] Realign app-level numeric tests. `tests/test_price_cart.py` lines 11-18: `line_total_cents == 31780`, `out["discount_cents"] == 10220`, comment `6 units total → volume ~24.333% off the 42000 line`. `tests/test_begin_checkout_engine.py`: both `8209` (32-33) → `10220`. `tests/test_reorder_checkout_engine.py`: the two `8209` asserts → `10220`, line total → `31780`.
- [ ] In `tests/test_inhouse_volume_pricing.py` update member=True expected cents (leave `member=` kwargs; Task 2 removes them): `f(FF,3,member=True)==5773`, `f(FF,6,member=True)==5294`, `f(FF,12,member=True)==4968`; comment `vp(12)=29%`; `5294` at lines 58-59; `_inhouse_line_unit_cents(FF, None, 12, s, member=True) == 4968`; lines 88-91 `effective_unit_cents == 5294` (both) and `subtotal_cents == 5294*4 + 5294*2 + 7000*1`; lines 123-125 `unit_cents == 5294`, `subtotal_cents == 5294 * 6`.
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_price_cart.py tests/test_begin_checkout_engine.py tests/test_reorder_checkout_engine.py tests/test_inhouse_volume_pricing.py -q` → **PASS**.
- [ ] Commit: `git add -A && git commit -m "feat(pricing): reprice default volume curve to base \$70 → floor \$50"`

---

## Task 2 — Open volume pricing to everyone (in-house order entry + price-preview)

**Files**
- Modify `app.py:4580-4594` (`_inhouse_ff_unit_cents` — drop `member`), `app.py:4607-4613` (`_inhouse_line_unit_cents` — drop `member`), `app.py:26754,26762` (`_price_inhouse_invoice`), `app.py:27024,27026,27037,27044` (`api_orders_price_preview`)
- Modify `tests/test_inhouse_volume_pricing.py` (drop `member=` kwargs), `tests/test_paid_member_gate.py` (remove the two preview tests)

**Interfaces**
- Produces: `_inhouse_ff_unit_cents(p, total_ff_qty, settings) -> int`; `_inhouse_line_unit_cents(p, override, total_ff_qty, settings) -> int`.
- Consumes: `pricing.volume_pct`, `_inhouse_total_ff_qty(lines_in) -> int`.

**Steps**
- [ ] Rewrite inhouse tests to open-to-all. Replace `test_ff_unit_cents_by_total_qty`:
```python
def test_ff_unit_cents_by_total_qty():
    appmod = _app()
    s = _pricing.load_settings(None)
    f = appmod._inhouse_ff_unit_cents
    assert f(FF, 1, s) == 6997
    assert f(FF, 3, s) == 5773
    assert f(FF, 6, s) == 5294
    assert f(FF, 12, s) == 4968
    assert f(FF, 99, s) == 4968
    assert f(NONFF, 12, s) == 7000
```
  In `test_multi_ff_lines_share_total_rate` → `f(FF, 6, s) == 5294`, `f(FF2, 6, s) == 5294` (drop `member=`). Replace `test_line_unit_override_wins`:
```python
def test_line_unit_override_wins():
    appmod = _app()
    s = _pricing.load_settings(None)
    assert appmod._inhouse_line_unit_cents(FF, 5000, 12, s) == 5000
    assert appmod._inhouse_line_unit_cents(FF, None, 12, s) == 4968
    assert appmod._inhouse_line_unit_cents(NONFF, None, 12, s) == 7000
```
  In `test_price_preview_route` delete the two membership monkeypatches (lines 74-75). In `test_manual_charges_ff_effective_no_double_discount` delete the `_is_paid_member` monkeypatch (line 107).
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_inhouse_volume_pricing.py -q` → **FAIL**.
- [ ] Implement. Rewrite `_inhouse_ff_unit_cents` (app.py:4580):
```python
def _inhouse_ff_unit_cents(p, total_ff_qty, settings):
    """Effective in-house unit price (cents) for a $69.97 functional-formulation
    capsule (_qty_eligible): the order-wide volume rate driven by the TOTAL FF
    capsule quantity (1 bottle = 1 month). OPEN TO ALL. Clamped at the wholesale
    discount floor. Non-FF products return list price."""
    if not _qty_eligible(p):
        return int(p.get("price_cents") or 0)
    from dashboard import pricing as _pricing
    pct = _pricing.volume_pct(int(total_ff_qty or 0), settings)
    eff = int(round(6997 * (1 - (pct or 0) / 100.0)))
    floor = int(round(6997 * float(settings.get("discount_floor_pct", 0.57))))
    return max(eff, floor)
```
  Rewrite `_inhouse_line_unit_cents` (app.py:4607):
```python
def _inhouse_line_unit_cents(p, override, total_ff_qty, settings):
    """Explicit owner override wins; else FF capsules get the order-wide volume rate
    (open to all), everything else its list price."""
    if override not in (None, ""):
        return int(override)
    return _inhouse_ff_unit_cents(p, total_ff_qty, settings)
```
  In `_price_inhouse_invoice` delete line 26754 (`member = _is_paid_member(...)`); change 26762 to `unit_cents = _inhouse_line_unit_cents(p, ln.get("unit_cents"), total_ff_qty, settings)`. In `api_orders_price_preview` delete 27024 (`member = ...`); 27026 → `vol_pct = round(float(_pricing.volume_pct(total_ff_qty, settings) or 0), 2)`; 27037 → `unit = _inhouse_line_unit_cents(p, ov, total_ff_qty, settings)`; 27044 unchanged.
- [ ] In `tests/test_paid_member_gate.py` delete `test_trial_email_prices_regular_in_preview` and `test_full_email_prices_volume_in_preview` (lines 72-113). Keep the pure category→bool mapping tests.
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_inhouse_volume_pricing.py tests/test_paid_member_gate.py -q` → **PASS**.
- [ ] Commit: `git add -A && git commit -m "feat(pricing): open in-house/preview volume pricing to everyone"`

---

## Task 3 — Retire the member-credit + $1-trial-credit machinery

> Sequencing: must precede Task 4 (`_member_unit_prices`, deleted here, is the only non-checkout caller of `_qty_unit_cents`).

**Files**
- Modify `app.py`: delete `_member_unit_prices` (4634-4651), `_missed_member_discount_cents` (4654-4664), `_trial_credit_for_email` (4667-4674); `_invoice_summary` (27110-27118) `member_credit_cents` → `0`; member board (11414) → `0`; `/api/portal/<token>` (12512-12525) `trial_credit_cents` → `0`; subscription charge cron (22653-22673) remove the trial→full credit block
- Delete `dashboard/trial_credit.py`, `tests/test_trial_credit.py`, `tests/test_trial_credit_grant.py`, `tests/test_invoice_member_credit.py`
- Modify `tests/test_portal_trial_credit_display.py`

**Interfaces**
- Produces: `_invoice_summary(order)` with `"member_credit_cents": 0` (key retained; `static/invoice.html:274` renders only when `>0`). `/api/portal/<token>` with `"trial_credit_cents": 0` + unchanged `"membership_category"`.
- Removed symbols: `_member_unit_prices`, `_missed_member_discount_cents`, `_trial_credit_for_email`, module `dashboard.trial_credit`.

**Steps**
- [ ] Rewrite `tests/test_portal_trial_credit_display.py`: `EXPECTED = 0`; in `test_trial_buyer_sees_category_and_credit` assert `d["trial_credit_cents"] == 0` while keeping `d["membership_category"] == "trial"`. Delete `tests/test_invoice_member_credit.py`, `tests/test_trial_credit.py`, `tests/test_trial_credit_grant.py`.
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_portal_trial_credit_display.py -q` → **FAIL** (still computes `3000`).
- [ ] Implement deletions in `app.py`: remove the three helper defs (4634-4674). In `_invoice_summary` replace the `member_credit_cents` try/except (27110-27116) with `member_credit_cents = 0` and drop the dead comment (27106-27109). In the member board (11411-11417) replace with `credit = 0`. In the portal endpoint (12510-12521) replace the trial-credit block with `membership_cat = membership_category(email_for_reports) if email_for_reports else "none"` and `trial_credit_cents = 0` (keep both keys, 12524-12525). In the subscription cron delete the `was_trial`/trial-credit grant block (22653-22673) but keep `_subs.advance_after_charge` + `_subs.reset_failed_count`.
- [ ] `git rm dashboard/trial_credit.py`. Confirm: `grep -rn "trial_credit" app.py dashboard/` returns only doc comments (`dashboard/subscriptions.py:345,367`) — leave those.
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_portal_trial_credit_display.py tests/test_member_board.py tests/test_console_members_api.py tests/test_membership_charge_cron.py -q` → **PASS** (`member_board_row` at `dashboard/subscriptions.py:387` still accepts `credit_cents`; caller passes `0`).
- [ ] Commit: `git add -A && git commit -m "chore(pricing): retire member-price upsell + \$1-trial-credit machinery"`

---

## Task 4 — Retire legacy `_qty_unit_cents`/`_QTY_TIERS`; route charged callers through the engine

**Files**
- Modify `app.py:6810-6962` (`begin_checkout` — engine branch unconditional; delete legacy `else` 6905-6962), `app.py:15101-15169` (reorder — engine branch unconditional; delete legacy 15119-15169), `app.py:12380-12405` (`_portal_priced_lines`) + `app.py:12929` (drop `member=`), `app.py:4550-4577` (delete `_QTY_TIERS`, `_qty_unit_cents`; keep `_FF_BASE_CENTS`/`_FF_SRP_CENTS`/`_FORMATS`/`_qty_eligible`)
- Modify `tests/test_reorder_cart.py`

**Interfaces**
- Produces: `_portal_priced_lines(items) -> (lines, items_rec, subtotal_cents)` (drop `member`). Consumes `_inhouse_total_ff_qty`, `_inhouse_line_unit_cents(p, override, total_ff_qty, settings)`.
- Removed symbols: `_QTY_TIERS`, `_qty_unit_cents`.

**Steps**
- [ ] Update `tests/test_reorder_cart.py::test_checkout_builds_invoice_and_records` to the engine contract: qty-2 `terrain-restore` (6997) + qty-1 `brain-cleanse` (5997), `total_months=3` → `volume_pct(3)=17.5%`; QBO lines carry **list** amounts and discount rides `discount_cents`; keep `lines["Terrain Restore"]["amount"] == 69.97`, `lines["Brain Cleanse"]["amount"] == 59.97`, `o["total_cents"] == int((69.97*2 + 59.97)*100)`; assert `d["ok"]` and `d["stripe_url"]`.
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_reorder_cart.py -q` → baseline (pins the contract before the code edit).
- [ ] Implement `begin_checkout`: at `app.py:6810` delete the `if os.environ.get("PRICING_ENGINE_CHECKOUT"...):` guard so the engine body (6811-6903) always runs; delete the legacy `else:` block (6904-6962). Engine body already ends `return jsonify(out)` at 6903.
- [ ] Implement reorder `/reorder/checkout`: at `app.py:15102` delete the `if PRICING_ENGINE_CHECKOUT` guard; delete the legacy path (15119-15169).
- [ ] Implement `_portal_priced_lines`:
```python
def _portal_priced_lines(items):
    """Build QBO invoice lines from a portal's reorder items, honoring an optional
    per-item ``price_cents`` override (the practitioner-special price); else the
    order-wide volume rate (open to all). Returns (lines, items_rec, subtotal_cents)."""
    from dashboard import pricing as _pricing
    settings = _pricing.load_settings(_pricing_settings())
    total_ff_qty = _inhouse_total_ff_qty(items or [])
    lines, items_rec, subtotal_cents = [], [], 0
    for it in (items or []):
        slug = (it.get("slug") or "").strip()
        p = _get_product(slug) if slug else None
        if not p:
            continue
        try:
            qty = max(1, min(int(it.get("qty", 1) or 1), 99))
        except Exception:
            qty = 1
        unit_cents = _inhouse_line_unit_cents(p, it.get("price_cents"), total_ff_qty, settings)
        subtotal_cents += unit_cents * qty
        lines.append({"name": p["name"], "amount": round(unit_cents / 100.0, 2),
                      "qty": qty, "item_id": p.get("qbo_item_id"), "description": p["name"]})
        items_rec.append({"name": p["name"], "qty": qty, "desc": p["name"],
                          "slug": slug, "unit_cents": unit_cents, "line_cents": unit_cents * qty})
    return lines, items_rec, subtotal_cents
```
  At `app.py:12929` → `lines, items_rec, _subtotal = _portal_priced_lines(items)`.
- [ ] Delete `_QTY_TIERS` (4553) and `_qty_unit_cents` (4567-4577); keep `_FF_BASE_CENTS`, `_FF_SRP_CENTS`, `_FORMATS`, `_qty_eligible`. Confirm: `grep -n "_qty_unit_cents\|_QTY_TIERS" app.py` → empty.
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_reorder_cart.py tests/test_begin_checkout_engine.py tests/test_reorder_checkout_engine.py tests/test_price_cart.py -q` → **PASS**.
- [ ] Commit: `git add -A && git commit -m "refactor(pricing): retire legacy _qty_unit_cents; single engine everywhere"`

---

## Task 5 — Storefront product-data advertises open-to-all volume tiers

**Files**
- Modify `app.py:5194-5210` (`begin_product_data` — build `qty_tiers` from the engine curve)
- Create `tests/test_product_data_volume_tiers.py`

**Interfaces**
- Consumes: `pricing.volume_pct`, `_pricing_settings`, `_qty_eligible`, product `price_cents`.
- Produces: `/begin/product-data/<slug>` JSON `qty_pricing` list of `{"min","unit_cents","unit","save"}` from the live curve.

**Steps**
- [ ] Create `tests/test_product_data_volume_tiers.py`:
```python
import importlib, sys
from pathlib import Path
import pytest

def _app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")

def test_product_data_shows_open_volume_tiers(monkeypatch):
    appmod = _app()
    FF = {"slug": "brain", "name": "Brain Boost", "qty_pricing": True, "price_cents": 6997}
    monkeypatch.setattr(appmod, "_get_product", {"brain": FF}.get)
    c = appmod.app.test_client()
    d = c.get("/begin/product-data/brain").get_json()
    tiers = {t["min"]: t["unit_cents"] for t in d["qty_pricing"]}
    assert tiers == {1: 6997, 3: 5773, 6: 5294, 12: 4968}
```
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_product_data_volume_tiers.py -q` → **FAIL** (hardcoded `{1:6997,3:5997,6:4997,12:3997}`).
- [ ] Implement. In `begin_product_data` replace the `qty_tiers` block (5194-5198):
```python
    qty_tiers, formats = None, None
    if _qty_eligible(p):
        from dashboard import pricing as _pricing
        _s = _pricing.load_settings(_pricing_settings())
        _base = int(p["price_cents"])
        qty_tiers = []
        for m in (1, 3, 6, 12):
            u = int(round(_base * (1 - _pricing.volume_pct(m, _s) / 100.0)))
            qty_tiers.append({"min": m, "unit_cents": u, "unit": f"${u/100:.2f}",
                              "save": ((_base - u) // 100) if u < _base else 0})
        formats = _FORMATS
```
- [ ] Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_product_data_volume_tiers.py tests/test_founding_product_data.py -q` → **PASS**.
- [ ] Commit: `git add -A && git commit -m "feat(pricing): storefront product-data advertises open-to-all volume tiers"`

---

## Final verification (after all tasks)
- [ ] `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_pricing_engine.py tests/test_pricing_settings.py tests/test_console_pricing_settings_routes.py tests/test_price_cart.py tests/test_engine_item.py tests/test_inhouse_volume_pricing.py tests/test_paid_member_gate.py tests/test_begin_checkout_engine.py tests/test_reorder_checkout_engine.py tests/test_reorder_cart.py tests/test_invoice_edit.py tests/test_portal_trial_credit_display.py tests/test_member_board.py tests/test_console_members_api.py tests/test_membership_charge_cron.py tests/test_product_data_volume_tiers.py -q` → **all PASS**.
- [ ] `grep -rn "_qty_unit_cents\|_QTY_TIERS\|_member_unit_prices\|_missed_member_discount_cents\|_trial_credit_for_email\|from dashboard import trial_credit" app.py dashboard/ tests/` → empty (only doc-comment mentions may remain).
- [ ] Full-suite regression diff vs pristine tree (money-path change): capture failing node IDs before/after; confirm zero NEW failures attributable to this work.
