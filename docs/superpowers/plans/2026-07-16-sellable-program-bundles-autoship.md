# Sellable Program/Bundle SKUs with Bundle-Scoped Autoship — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the 10 cleanly-resolvable remedy/device bundles from remedymatch.com into `data/products.json` as sellable bundle SKUs priced by rule (10% off component retail sum), and give bundle autoships a bundle-scoped escalating discount ladder (12→29%) applied per line, with device bundles barred from autoship.

**Architecture:** Bundles are ordinary `products.json` entries with `bundle: true`. One-time price is materialized by a build script from `bundle_component_slugs` (rule-enforced by a drift-guard test, not hand-set). Autoship reuses the existing Stripe/subscriptions machinery; the only change is that the pricing engine's `subscriber_tier_pct` may now be a per-item callable, so a bundle line earns the 12→29 ladder while a single SKU in the same subscription keeps 3→25.

**Tech Stack:** Python 3, Flask (`app.py`), sqlite3, pytest. Pure pricing modules under `dashboard/` (`pricing.py`, `subscriptions.py`).

## Global Constraints

- **One-time bundle price = `round(0.9 × Σ(component price_cents × qty))`**, computed by rule, never hand-set. Enforced by a drift-guard test.
- **Component retail = the component's `price_cents`** (retail), resolved by **slug** (not name) on the money path.
- **Bundle autoship ladder** = `BUNDLE_SUBSCRIBE_TIERS = [12, 14, 16, 18, 20, 22, 24, 26, 28, 29]`; single-SKU ladder `SUBSCRIBE_TIERS = [3,…,25]` is **unchanged**.
- **Per-line**, not per-subscription: in a mixed subscription the bundle line gets 12→29, the single SKU gets 3→25.
- **Device bundles** (`autoship_eligible: false`) are one-time only; `/reorder/subscribe` must reject them.
- **No schema migration** — `order_count` already exists on `subscriptions`; existing subscriptions must charge exactly as before (all their lines are non-bundle → standard ladder).
- **deploy-chat is merge=deploy (no CI)** — verify live after deploy.
- Copy: the shopper headline (follow-up plan) is **"Save more each time"** — no em dashes, no ALL CAPS.

## The 10 bundles (this plan)

| Slug | Action | `autoship_eligible` | Component slugs (qty) |
|---|---|---|---|
| `crystalline-lens-program` | new | true | crystalline-clarity, clear-lens-eye-drops-aces-cat-eye-drops, clarity, golden-book, crucifer-complex |
| `gut-terrain-program` | upgrade | true | terrain-restore, microbiome, fiber-cleanse |
| `dry-eye-relief-program` | re-price | true | aces-eye-drops, moisturize, wholomega |
| `macular-wellness-program` | re-price | true | macular-wellness-lycopene, macular-wellness-astaxanthin, lipid-cleanse, lipid-zyme, wholomega |
| `glucose-tolerance-program` | re-price | true | glucose-tolerance, reverse-age |
| `brain-program` | upgrade | true | brain-boost, brain-cleanse, neuro-magnesium, neuroprotect, nous-energy |
| `scar-reduction-program` | new | true | scar-silk, scar-solve, vitamin-e-spectrum, msm-syntropy-powder |
| `iop-program` | upgrade | true | iop-syntropy ×3, ocuflow-daytime ×2 |
| `dental-bundle` | new | **false** | dental-powder, wicking-toothbrush ×2 |
| `sleep-bundle` | new | **false** | brain-cleanse, sleep-syntropy, therapeutic-nightlight, biocompatible-nightlight |

**Deferred (not this plan — missing component SKUs):** OcuFlow (Neuro Magnesium Drink Mix), Reverse Aging (Vitamin C Syntropy), Skin (MSM Lotion), Travel (4 devices).

**Follow-up plan (not here):** the shopper-facing "Subscribe & save — save more each time" CTA on `/begin/product/<slug>` that posts to `/reorder/subscribe`. No such UI exists today; it needs its own design pass (address capture, cadence picker, Stripe redirect). This plan makes bundles sellable one-time immediately (auto-generated product pages) and makes autoship pricing correct + gated at the engine level.

---

## Task 1: Bundle price rule + build script + drift-guard

**Files:**
- Create: `dashboard/bundle_pricing.py`
- Create: `scripts/compute_bundle_prices.py`
- Test: `tests/test_bundle_price_rule.py`

**Interfaces:**
- Produces: `dashboard.bundle_pricing.compute_bundle_price_cents(product: dict, products: dict) -> int` — the rule; and `resolve_component(slug, products) -> dict | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bundle_price_rule.py
import json, os
from dashboard.bundle_pricing import compute_bundle_price_cents

CATALOG = {
    "aces-eye-drops": {"name": "ACES Eye Drops", "price_cents": 6997},
    "moisturize": {"name": "Moisturize", "price_cents": 6997},
    "wholomega": {"name": "WholOmega", "price_cents": 6997},
    "iop-syntropy": {"name": "IOP Syntropy", "price_cents": 6997},
    "ocuflow-daytime": {"name": "OcuFlow Daytime", "price_cents": 6997},
}

def test_sum_less_10pct_simple():
    bundle = {"bundle": True, "price_rule": "components_less_10pct",
              "bundle_component_slugs": [
                  {"slug": "aces-eye-drops", "qty": 1},
                  {"slug": "moisturize", "qty": 1},
                  {"slug": "wholomega", "qty": 1}]}
    # 3 * 6997 = 20991 ; * 0.9 = 18891.9 -> 18892
    assert compute_bundle_price_cents(bundle, CATALOG) == 18892

def test_sum_less_10pct_with_qty():
    bundle = {"bundle": True, "price_rule": "components_less_10pct",
              "bundle_component_slugs": [
                  {"slug": "iop-syntropy", "qty": 3},
                  {"slug": "ocuflow-daytime", "qty": 2}]}
    # 5 * 6997 = 34985 ; * 0.9 = 31486.5 -> 31486  (banker's rounding of .5 -> even)
    assert compute_bundle_price_cents(bundle, CATALOG) == 31486

def test_unknown_component_raises():
    bundle = {"bundle": True, "price_rule": "components_less_10pct",
              "bundle_component_slugs": [{"slug": "does-not-exist", "qty": 1}]}
    try:
        compute_bundle_price_cents(bundle, CATALOG)
        assert False, "expected KeyError"
    except KeyError:
        pass

def test_live_catalog_prices_match_rule():
    """Drift guard: every components_less_10pct bundle in the real catalog
    stores exactly the rule price."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")
    with open(path) as f:
        data = json.load(f)
    products = data["products"]
    checked = 0
    for slug, p in products.items():
        if p.get("price_rule") == "components_less_10pct":
            expected = compute_bundle_price_cents(p, products)
            assert p.get("price_cents") == expected, (
                f"{slug}: stored {p.get('price_cents')} != rule {expected}")
            checked += 1
    assert checked >= 10, f"expected >=10 rule-priced bundles, saw {checked}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-f98be149 && python -m pytest tests/test_bundle_price_rule.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.bundle_pricing'`.

- [ ] **Step 3: Write the module**

```python
# dashboard/bundle_pricing.py
"""Rule pricing for bundle SKUs: one-time price = 10% off the summed retail
(price_cents) of the bundle's components, resolved by slug.

Pure module — no Flask, no I/O. The caller passes the products dict."""

DEFAULT_PRICE_CENTS = 6997


def resolve_component(slug: str, products: dict):
    """The component product for a slug, following superseded_by, dropping
    inactive. Returns the product dict (without mutation) or None."""
    seen = set()
    cur = slug
    while cur and cur not in seen:
        seen.add(cur)
        p = products.get(cur)
        if p is None:
            return None
        nxt = p.get("superseded_by")
        if nxt and nxt != cur:
            cur = nxt
            continue
        if p.get("inactive"):
            return None
        return p
    return None


def compute_bundle_price_cents(product: dict, products: dict) -> int:
    """round(0.9 * sum(component price_cents * qty)) in integer cents.
    Raises KeyError if any component slug does not resolve to a sellable product."""
    total = 0
    for comp in product.get("bundle_component_slugs") or []:
        slug = comp["slug"]
        qty = int(comp.get("qty", 1))
        p = resolve_component(slug, products)
        if p is None:
            raise KeyError(f"unresolvable bundle component slug: {slug!r}")
        total += int(p.get("price_cents", DEFAULT_PRICE_CENTS)) * qty
    return int(round(total * 0.9))
```

- [ ] **Step 4: Run the unit tests (the live drift test will still fail — no bundles carry the rule yet)**

Run: `python -m pytest tests/test_bundle_price_rule.py -q -k "not live_catalog"`
Expected: 3 passed.

(The `test_live_catalog_prices_match_rule` test stays red until Task 2 adds the bundles — that is intended; it is the drift guard.)

- [ ] **Step 5: Write the build script**

```python
# scripts/compute_bundle_prices.py
"""Materialize rule-computed one-time prices into data/products.json.

For every product with price_rule == "components_less_10pct", set
price_cents = round(0.9 * sum(component price_cents * qty)). Idempotent.

Usage:
  python scripts/compute_bundle_prices.py          # write
  python scripts/compute_bundle_prices.py --check   # exit 1 if any drift
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.bundle_pricing import compute_bundle_price_cents

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "products.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift, do not write")
    args = ap.parse_args()

    with open(PATH) as f:
        data = json.load(f)
    products = data["products"]

    drift, changes = [], []
    for slug, p in products.items():
        if p.get("price_rule") != "components_less_10pct":
            continue
        want = compute_bundle_price_cents(p, products)
        have = p.get("price_cents")
        if have != want:
            drift.append((slug, have, want))
            p["price_cents"] = want
            changes.append(slug)

    if args.check:
        for slug, have, want in drift:
            print(f"DRIFT {slug}: {have} -> {want}")
        sys.exit(1 if drift else 0)

    if changes:
        with open(PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        for slug, have, want in drift:
            print(f"set {slug}: {have} -> {want} (${want/100:.2f})")
    else:
        print("no changes; all bundle prices already match the rule")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-f98be149
git add dashboard/bundle_pricing.py scripts/compute_bundle_prices.py tests/test_bundle_price_rule.py
git commit -m "feat: bundle price rule (10% off component retail sum) + build script + drift guard"
```

---

## Task 2: Port the 10 bundle SKUs into products.json

**Files:**
- Modify: `data/products.json` (add 4 new bundle entries; add fields to 6 existing entries)
- Test: `tests/test_bundle_catalog.py`

**Interfaces:**
- Consumes: `dashboard.bundle_pricing.compute_bundle_price_cents` (Task 1), `scripts/compute_bundle_prices.py` (Task 1).
- Produces: 10 products with `bundle: true`, `bundle_component_slugs`, `price_rule: "components_less_10pct"`, `autoship_eligible`, and rule-computed `price_cents`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bundle_catalog.py
import json, os
from dashboard.bundle_pricing import resolve_component

def _catalog():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")
    with open(path) as f:
        return json.load(f)["products"]

PORTED = [
    "crystalline-lens-program", "gut-terrain-program", "dry-eye-relief-program",
    "macular-wellness-program", "glucose-tolerance-program", "brain-program",
    "scar-reduction-program", "iop-program", "dental-bundle", "sleep-bundle",
]
DEVICE_BUNDLES = {"dental-bundle", "sleep-bundle"}

def test_all_ten_present_and_flagged():
    c = _catalog()
    for slug in PORTED:
        assert slug in c, f"missing {slug}"
        p = c[slug]
        assert p.get("bundle") is True, f"{slug} not bundle:true"
        assert p.get("price_rule") == "components_less_10pct", f"{slug} missing price_rule"
        assert isinstance(p.get("bundle_component_slugs"), list) and p["bundle_component_slugs"], slug

def test_autoship_eligibility():
    c = _catalog()
    for slug in PORTED:
        p = c[slug]
        expected = slug not in DEVICE_BUNDLES
        assert p.get("autoship_eligible") is expected, f"{slug} autoship_eligible wrong"

def test_every_component_slug_resolves():
    c = _catalog()
    for slug in PORTED:
        for comp in c[slug]["bundle_component_slugs"]:
            assert resolve_component(comp["slug"], c) is not None, \
                f"{slug} component {comp['slug']} does not resolve"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bundle_catalog.py -q`
Expected: FAIL — `missing crystalline-lens-program` (new entries not added yet).

- [ ] **Step 3: Add the 4 new bundle entries to `data/products.json`**

Insert these into the `"products"` object (omit `price_cents` — the build script fills it). Keep 2-space indentation consistent with the file.

```jsonc
"crystalline-lens-program": {
  "name": "Crystalline Lens Program",
  "bundle": true,
  "price_rule": "components_less_10pct",
  "autoship_eligible": true,
  "bundle_components": ["Crystalline Clarity", "Clear Lens Eye Drops", "Clarity", "Golden Book", "Crucifer Complex"],
  "bundle_component_slugs": [
    {"slug": "crystalline-clarity", "qty": 1},
    {"slug": "clear-lens-eye-drops-aces-cat-eye-drops", "qty": 1},
    {"slug": "clarity", "qty": 1},
    {"slug": "golden-book", "qty": 1},
    {"slug": "crucifer-complex", "qty": 1}
  ],
  "bundle_description": "A monthly supply supporting crystalline lens health and clarity: Crystalline Clarity, Clear Lens Eye Drops, Clarity, Golden Book, and Crucifer Complex."
},
"scar-reduction-program": {
  "name": "Scar Reduction Program",
  "bundle": true,
  "price_rule": "components_less_10pct",
  "autoship_eligible": true,
  "bundle_components": ["Scar Silk", "Scar Solve", "Vitamin E Spectrum", "MSM Syntropy Powder"],
  "bundle_component_slugs": [
    {"slug": "scar-silk", "qty": 1},
    {"slug": "scar-solve", "qty": 1},
    {"slug": "vitamin-e-spectrum", "qty": 1},
    {"slug": "msm-syntropy-powder", "qty": 1}
  ],
  "bundle_description": "Internal and external scar-tissue support: Scar Silk, Scar Solve, Vitamin E Spectrum, and MSM Syntropy Powder."
},
"dental-bundle": {
  "name": "Dental Bundle",
  "bundle": true,
  "price_rule": "components_less_10pct",
  "autoship_eligible": false,
  "bundle_components": ["Dental Powder", "Wicking Toothbrush"],
  "bundle_component_slugs": [
    {"slug": "dental-powder", "qty": 1},
    {"slug": "wicking-toothbrush", "qty": 2}
  ],
  "bundle_description": "Oral-care essentials for remineralization and dental health: Dental Powder plus two Wicking Toothbrushes."
},
"sleep-bundle": {
  "name": "Sleep Bundle",
  "bundle": true,
  "price_rule": "components_less_10pct",
  "autoship_eligible": false,
  "bundle_components": ["Brain Cleanse", "Sleep Synergy", "Therapeutic Nightlight", "Biocompatible Nightlight"],
  "bundle_component_slugs": [
    {"slug": "brain-cleanse", "qty": 1},
    {"slug": "sleep-syntropy", "qty": 1},
    {"slug": "therapeutic-nightlight", "qty": 1},
    {"slug": "biocompatible-nightlight", "qty": 1}
  ],
  "bundle_description": "Support for restful, regenerative, cleansing sleep: Brain Cleanse, Sleep Synergy, and biocompatible lighting."
}
```

- [ ] **Step 4: Add the rule fields to the 6 existing entries**

For each existing product below, add `"bundle": true` (already present on the 3), `"price_rule": "components_less_10pct"`, `"autoship_eligible": true`, and `"bundle_component_slugs"`. Do NOT remove existing fields (e.g. `bundle_components`, `bundle_description`). Leave `price_cents` in place — Step 5 overwrites it via the rule.

```jsonc
// gut-terrain-program  (add):
"price_rule": "components_less_10pct",
"autoship_eligible": true,
"bundle_component_slugs": [
  {"slug": "terrain-restore", "qty": 1},
  {"slug": "microbiome", "qty": 1},
  {"slug": "fiber-cleanse", "qty": 1}
]

// brain-program  (add):
"price_rule": "components_less_10pct",
"autoship_eligible": true,
"bundle": true,
"bundle_component_slugs": [
  {"slug": "brain-boost", "qty": 1},
  {"slug": "brain-cleanse", "qty": 1},
  {"slug": "neuro-magnesium", "qty": 1},
  {"slug": "neuroprotect", "qty": 1},
  {"slug": "nous-energy", "qty": 1}
]

// iop-program  (add):
"price_rule": "components_less_10pct",
"autoship_eligible": true,
"bundle": true,
"bundle_component_slugs": [
  {"slug": "iop-syntropy", "qty": 3},
  {"slug": "ocuflow-daytime", "qty": 2}
]

// dry-eye-relief-program  (add; already bundle:true):
"price_rule": "components_less_10pct",
"autoship_eligible": true,
"bundle_component_slugs": [
  {"slug": "aces-eye-drops", "qty": 1},
  {"slug": "moisturize", "qty": 1},
  {"slug": "wholomega", "qty": 1}
]

// macular-wellness-program  (add; already bundle:true):
"price_rule": "components_less_10pct",
"autoship_eligible": true,
"bundle_component_slugs": [
  {"slug": "macular-wellness-lycopene", "qty": 1},
  {"slug": "macular-wellness-astaxanthin", "qty": 1},
  {"slug": "lipid-cleanse", "qty": 1},
  {"slug": "lipid-zyme", "qty": 1},
  {"slug": "wholomega", "qty": 1}
]

// glucose-tolerance-program  (add; already bundle:true):
"price_rule": "components_less_10pct",
"autoship_eligible": true,
"bundle_component_slugs": [
  {"slug": "glucose-tolerance", "qty": 1},
  {"slug": "reverse-age", "qty": 1}
]
```

Verify `gut-terrain-program`, `brain-program`, `iop-program` also have `"bundle": true` (added above where the plain SKU lacked it).

- [ ] **Step 5: Materialize the rule prices**

Run: `python scripts/compute_bundle_prices.py`
Expected output (order may vary):
```
set crystalline-lens-program: None -> 31486 ($314.86)
set gut-terrain-program: 15997 -> 18892 ($188.92)
set dry-eye-relief-program: 24997 -> 18892 ($188.92)
set macular-wellness-program: 38997 -> 31486 ($314.86)
set glucose-tolerance-program: 11997 -> 12595 ($125.95)
set brain-program: 29997 -> 31486 ($314.86)
set scar-reduction-program: None -> 22489 ($224.89)
set iop-program: 29985 -> 31486 ($314.86)
set dental-bundle: None -> 9892 ($98.92)
set sleep-bundle: None -> 35095 ($350.95)
```

- [ ] **Step 6: Run all bundle tests (drift guard now green)**

Run: `python -m pytest tests/test_bundle_price_rule.py tests/test_bundle_catalog.py -q`
Expected: all pass (including `test_live_catalog_prices_match_rule` and `test_sum_less_10pct_with_qty`).

- [ ] **Step 7: Sanity-check JSON validity + that nothing else moved**

Run: `python -c "import json; json.load(open('data/products.json')); print('ok')"`
Expected: `ok`
Run: `git diff --stat data/products.json`
Expected: only `data/products.json` changed.

- [ ] **Step 8: Commit**

```bash
git add data/products.json tests/test_bundle_catalog.py
git commit -m "feat: port 10 bundle SKUs (rule-priced); reprice existing bundles to 10%-off-sum"
```

---

## Task 3: Bundle-scoped autoship ladder

**Files:**
- Modify: `dashboard/subscriptions.py:21-28`
- Test: `tests/test_bundle_autoship.py`

**Interfaces:**
- Produces: `dashboard.subscriptions.BUNDLE_SUBSCRIBE_TIERS` and `tier_for_bundle(n: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bundle_autoship.py
from dashboard.subscriptions import tier_for, tier_for_bundle, BUNDLE_SUBSCRIBE_TIERS

def test_bundle_ladder_values():
    assert BUNDLE_SUBSCRIBE_TIERS == [12, 14, 16, 18, 20, 22, 24, 26, 28, 29]

def test_tier_for_bundle_first_and_cap():
    assert tier_for_bundle(0) == 12
    assert tier_for_bundle(1) == 14
    assert tier_for_bundle(9) == 29
    assert tier_for_bundle(50) == 29   # clamped at cap

def test_single_ladder_unchanged():
    assert tier_for(0) == 3
    assert tier_for(11) == 25
    assert tier_for(99) == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bundle_autoship.py -q`
Expected: FAIL — `ImportError: cannot import name 'tier_for_bundle'`.

- [ ] **Step 3: Add the ladder to `dashboard/subscriptions.py`** (immediately after `tier_for`, line 28)

```python
# Bundle-scoped loyalty ladder: bundles start richer (12%) and climb +2%/month to a
# 29% cap. Applied PER LINE (a bundle line only) — single SKUs use SUBSCRIBE_TIERS.
BUNDLE_SUBSCRIBE_TIERS = [12, 14, 16, 18, 20, 22, 24, 26, 28, 29]


def tier_for_bundle(n: int) -> int:
    """Bundle-line subscriber discount % for *n* completed active months
    (order_count). Clamped at the top step (29%)."""
    idx = min(int(n), len(BUNDLE_SUBSCRIBE_TIERS) - 1)
    return BUNDLE_SUBSCRIBE_TIERS[idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bundle_autoship.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/subscriptions.py tests/test_bundle_autoship.py
git commit -m "feat: bundle-scoped autoship ladder (12->29) tier_for_bundle"
```

---

## Task 4: Per-line subscriber tier in the pricing engine

**Files:**
- Modify: `dashboard/pricing.py:139-177`
- Test: `tests/test_pricing_per_line_tier.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `dashboard.pricing.compute(..., subscriber_tier_pct=...)` now accepts `subscriber_tier_pct` as **int|None (uniform, back-compat) OR callable(item_dict) -> int|None (per-line)**.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pricing_per_line_tier.py
from dashboard import pricing

SETTINGS = pricing.load_settings({})

def _item(slug, product):
    return {"slug": slug, "name": slug, "qty": 1, "product": product,
            "unit_cents": 10000, "months": 1, "volume_eligible": True}

def test_scalar_tier_uniform_backcompat():
    items = [_item("a", {}), _item("b", {})]
    res = pricing.compute(items, settings=SETTINGS, subscriber_tier_pct=20)
    assert [ln["pct_applied"] for ln in res["lines"]] == [20, 20]

def test_callable_tier_per_line():
    items = [_item("bundle", {"bundle": True}), _item("single", {})]
    def resolver(it):
        return 29 if it["product"].get("bundle") else 3
    res = pricing.compute(items, settings=SETTINGS, subscriber_tier_pct=resolver)
    by_slug = {ln["slug"]: ln["pct_applied"] for ln in res["lines"]}
    assert by_slug["bundle"] == 29
    assert by_slug["single"] == 3

def test_callable_none_falls_back_to_coupon():
    items = [_item("a", {})]
    res = pricing.compute(items, settings=SETTINGS,
                          subscriber_tier_pct=lambda it: None, coupon_pct=10)
    assert res["lines"][0]["pct_applied"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pricing_per_line_tier.py -q`
Expected: FAIL — `test_callable_tier_per_line` errors (callable treated as a number) / assertion fails.

- [ ] **Step 3: Make `compute` resolve the tier per item**

In `dashboard/pricing.py`, remove the pre-loop scalar assignment (lines 155-157):

```python
    base_pct = coupon_pct or 0
    if subscriber_tier_pct is not None:
        base_pct = subscriber_tier_pct      # subscriber tier wins whenever present, even 0
```

Replace it with a per-item resolver helper (keep the `total_months` / `open_pct` / `prog_pct` lines that follow):

```python
    def _sub_pct(it):
        v = subscriber_tier_pct(it) if callable(subscriber_tier_pct) else subscriber_tier_pct
        return v
```

Then inside the `for it in items:` loop, compute `base_pct` per line. Change the `line_pct` block (around lines 170-177) from using the module-level `base_pct` to:

```python
        sub_pct = _sub_pct(it)
        base_pct = coupon_pct or 0
        if sub_pct is not None:
            base_pct = sub_pct              # subscriber tier wins whenever present, even 0
        t1 = same_sku_pct(qty, settings) if eligible else 0.0       # type1: this line's SKU qty
        order_pct = max(prog_pct, open_pct) if eligible else 0.0     # type2 (gated) / type3
        rep_pct = 0.0
        if repertoire_slugs and eligible and (it.get("slug") or "").strip().lower() in repertoire_slugs:
            rep_pct = float(settings.get("repertoire_reorder_pct") or 0.0) * 100.0
        line_pct = max(t1, order_pct, base_pct, rep_pct)             # non-additive: best single offer
```

(Update the docstring note near line 156 to say `subscriber_tier_pct` may be an int or a callable(item)->int|None applied per line.)

- [ ] **Step 4: Run the new test + the full pricing suite (back-compat)**

Run: `python -m pytest tests/test_pricing_per_line_tier.py -q`
Expected: 3 passed.
Run: `python -m pytest tests/ -q -k "pricing or price or subscribe or subscription"`
Expected: all pass (no regression in existing scalar behavior).

- [ ] **Step 5: Commit**

```bash
git add dashboard/pricing.py tests/test_pricing_per_line_tier.py
git commit -m "feat: pricing.compute subscriber_tier_pct accepts a per-line callable resolver"
```

---

## Task 5: Wire per-line bundle ladder through _price_cart, subscribe route, and cron

**Files:**
- Modify: `app.py:5764-5766` (add `_price_cart` params) and `app.py:5847-5853` (resolver)
- Modify: `app.py:25386-25387` (subscribe route)
- Modify: `app.py:33027-33033` (cron)
- Test: `tests/test_subscription_tier_resolver.py`

**Interfaces:**
- Consumes: `dashboard.subscriptions.tier_for`, `tier_for_bundle` (Task 3); the per-line callable in `pricing.compute` (Task 4).
- Produces: `app._subscription_tier_resolver(order_count: int, active: bool) -> callable`; `_price_cart(..., subscriber_order_count=None, subscriber_active=True)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subscription_tier_resolver.py
import app

def _item(product):
    return {"slug": "x", "name": "x", "qty": 1, "product": product,
            "unit_cents": 10000, "months": 1, "volume_eligible": True}

def test_bundle_item_uses_bundle_ladder():
    r = app._subscription_tier_resolver(0, True)
    assert r(_item({"bundle": True, "autoship_eligible": True})) == 12

def test_single_item_uses_standard_ladder():
    r = app._subscription_tier_resolver(0, True)
    assert r(_item({})) == 3

def test_device_bundle_not_eligible_uses_standard_ladder():
    # a bundle flagged autoship_eligible False should NOT get the bundle ladder
    r = app._subscription_tier_resolver(0, True)
    assert r(_item({"bundle": True, "autoship_eligible": False})) == 3

def test_inactive_membership_zeroes_all():
    r = app._subscription_tier_resolver(5, False)
    assert r(_item({"bundle": True, "autoship_eligible": True})) == 0
    assert r(_item({})) == 0

def test_climbs_with_order_count():
    r = app._subscription_tier_resolver(9, True)
    assert r(_item({"bundle": True, "autoship_eligible": True})) == 29  # bundle cap
    assert r(_item({})) == 21                                           # tier_for(9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -- python -m pytest tests/test_subscription_tier_resolver.py -q`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_subscription_tier_resolver'`.
(This test imports `app`; app-importing tests SILENTLY SKIP under bare `pytest` without Doppler — use `doppler run --`, or a real FAIL is indistinguishable from a skip.)

- [ ] **Step 3: Add the resolver + `_price_cart` params in `app.py`**

Add the resolver helper just above `_price_cart` (before line 5764):

```python
def _subscription_tier_resolver(order_count, active=True):
    """Per-line subscriber-discount resolver for a subscription charge.
    Bundle lines flagged autoship_eligible climb the 12->29 bundle ladder;
    every other line climbs the standard 3->25 ladder. When the member is not
    paid-through (active False) all lines get 0 — matching the prior cron gate."""
    from dashboard import subscriptions as _subs
    oc = int(order_count or 0)

    def _resolve(it):
        if not active:
            return 0
        p = it.get("product") or {}
        if p.get("bundle") and p.get("autoship_eligible"):
            return _subs.tier_for_bundle(oc)
        return _subs.tier_for(oc)

    return _resolve
```

Change the `_price_cart` signature (line 5764-5766) to add two params:

```python
def _price_cart(cart, *, ship, coupon_pct=None, subscriber_tier_pct=None,
                subscriber_order_count=None, subscriber_active=True,
                points_to_redeem_cents=0, channel="retail", program_member=False,
                email=None):
```

Just before the `_pricing.compute(...)` call (line 5847), select the per-line resolver when a subscription order count is supplied:

```python
    tier_arg = subscriber_tier_pct
    if subscriber_order_count is not None:
        tier_arg = _subscription_tier_resolver(subscriber_order_count, subscriber_active)
    priced = _pricing.compute(items, settings=settings, coupon_pct=coupon_pct,
                              subscriber_tier_pct=tier_arg, channel=channel,
                              points_to_redeem_cents=int(points_to_redeem_cents or 0),
                              ship_to_state=ship.get("state", ""),
                              tax_fn=_tax.compute_get_cents,
                              program_member=bool(program_member),
                              repertoire_slugs=rep_slugs)
```

- [ ] **Step 4: Point the subscribe route at the resolver** (`app.py:25386-25387`)

Replace:

```python
            pc = _price_cart(cart, ship=ship, subscriber_tier_pct=_subs.tier_for(0),
                             program_member=_is_paid_member(email), email=email)
```

with (first order → order_count 0, unconditionally active, matching today's always-on first-order tier):

```python
            pc = _price_cart(cart, ship=ship, subscriber_order_count=0, subscriber_active=True,
                             program_member=_is_paid_member(email), email=email)
```

- [ ] **Step 5: Point the cron at the resolver** (`app.py:33027-33033`)

Replace the `tier_pct` line + pricing call:

```python
                order_count = sub.get("order_count", 0)
                # Member loyalty discount applies only while paid-through; the
                # tier VALUE is the earned tier_for(order_count) (held, not reset).
                tier_pct = _subs.tier_for(order_count) if _active_membership_for_email(sub["email"]) else 0

                # Price the order
                try:
                    pc = _price_cart(items, ship=ship, subscriber_tier_pct=tier_pct,
                                     program_member=_is_paid_member(sub["email"]),
                                     email=sub["email"])
```

with:

```python
                order_count = sub.get("order_count", 0)
                # Per-line loyalty: bundle lines climb 12->29, single SKUs 3->25.
                # Gated to 0 for all lines when the member isn't paid-through.
                active = bool(_active_membership_for_email(sub["email"]))

                # Price the order
                try:
                    pc = _price_cart(items, ship=ship, subscriber_order_count=order_count,
                                     subscriber_active=active,
                                     program_member=_is_paid_member(sub["email"]),
                                     email=sub["email"])
```

- [ ] **Step 6: Run resolver tests + existing subscribe/cron suites**

Run: `python -m pytest tests/test_subscription_tier_resolver.py -q`
Expected: 5 passed.
Run: `python -m pytest tests/ -q -k "subscribe or subscription or cron or price"`
Expected: all pass (existing single-SKU subscription behavior unchanged: a non-bundle line at order_count N still resolves `tier_for(N)`; inactive membership still 0).

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_subscription_tier_resolver.py
git commit -m "feat: per-line bundle autoship ladder through _price_cart, subscribe route, and charge cron"
```

---

## Task 6: Device gate — reject autoship for device bundles

**Files:**
- Modify: `app.py` (in `reorder_subscribe`, after cart is read at line 25376)
- Test: `tests/test_subscribe_device_gate.py`

**Interfaces:**
- Consumes: `_get_product` (existing), the `autoship_eligible` field (Task 2).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subscribe_device_gate.py
import app

def test_helper_flags_device_bundle():
    # dental-bundle is a real device bundle (autoship_eligible False) after Task 2
    assert app._cart_has_noautoship_bundle([{"slug": "dental-bundle", "qty": 1}]) is True

def test_helper_allows_remedy_bundle():
    assert app._cart_has_noautoship_bundle([{"slug": "crystalline-lens-program", "qty": 1}]) is False

def test_helper_allows_single_sku():
    assert app._cart_has_noautoship_bundle([{"slug": "wholomega", "qty": 1}]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -- python -m pytest tests/test_subscribe_device_gate.py -q`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_cart_has_noautoship_bundle'`.
(Imports `app` — run under `doppler run --`; app-importing tests silently skip under bare `pytest`.)

- [ ] **Step 3: Add the helper + the gate**

Add the helper near `_price_cart` (e.g. just after `_subscription_tier_resolver`):

```python
def _cart_has_noautoship_bundle(cart):
    """True if any cart line is a bundle explicitly barred from autoship
    (autoship_eligible False) — device bundles are one-time purchase only."""
    for c in (cart or []):
        p = _get_product((c.get("slug") or "").strip())
        if p and p.get("bundle") and not p.get("autoship_eligible", False):
            return True
    return False
```

In `reorder_subscribe`, right after `cart = body.get("items") or []` (line 25376) and the `ship` line, add the gate:

```python
        if _cart_has_noautoship_bundle(cart):
            return jsonify({"ok": False,
                            "error": "This item is available for one-time purchase only, not autoship."}), 400
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/test_subscribe_device_gate.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_subscribe_device_gate.py
git commit -m "feat: reject autoship for device bundles (autoship_eligible false)"
```

---

## Final verification (before deploy)

- [ ] **Full targeted suite:** `python -m pytest tests/test_bundle_price_rule.py tests/test_bundle_catalog.py tests/test_bundle_autoship.py tests/test_pricing_per_line_tier.py tests/test_subscription_tier_resolver.py tests/test_subscribe_device_gate.py -q` → all pass.
- [ ] **Regression sweep:** `python -m pytest tests/ -q -k "price or subscribe or subscription or cron or checkout"` → all pass (Doppler-gated app-import tests may skip under bare pytest — run with the project's usual `doppler run -- pytest` if available; note skips vs failures).
- [ ] **Drift guard in CI-less deploy:** `python scripts/compute_bundle_prices.py --check` → exit 0 (no drift).
- [ ] **Render one bundle page (real app):** confirm `/begin/product/crystalline-lens-program` returns 200 and `/begin/product-data/crystalline-lens-program` shows the $314.86 one-time price. Confirm `/begin/product/dental-bundle` renders (one-time; no autoship path exists yet).
- [ ] **Manual autoship-pricing check:** in a Python shell with the app catalog loaded, price a mixed cart `[crystalline-lens-program, wholomega]` via `_price_cart(..., subscriber_order_count=0, subscriber_active=True)` and confirm the bundle line = 12% off, the single = 3% off.

## Rollout notes

- Re-pricing the 3 existing bundles changes live one-time prices (Dry Eye $249.97→$188.92, Macular $389.97→$314.86, Glucose $119.97→$125.95, plus the upgraded plain SKUs). Intended — confirmed with Glen.
- No `subscriptions` schema change; existing subscriptions charge exactly as before.
- Device bundles ship sellable one-time with no autoship path.
- **Follow-up plan required for the shopper subscribe CTA** — `/reorder/subscribe` still has no shopper UI. Until then, autoship is correct + gated at the engine level but not yet clickable by shoppers.
