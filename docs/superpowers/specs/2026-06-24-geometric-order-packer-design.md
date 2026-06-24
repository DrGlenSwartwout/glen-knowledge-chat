# Geometric Order Packer → Flat-Rate Shipping

**Date:** 2026-06-24
**Author:** Glen Swartwout (design w/ Claude)
**Status:** Approved for planning
**Repo:** deploy-chat

## Problem

Shipping cost at checkout should be derived from which USPS flat-rate box(es)
an order's bottles actually pack into. The scaffolding exists but is dormant:

- `dashboard/shipping.py` has `bottle_types`, `box_capacity`, `usps_rates`
  tables, a `pick_box()` / `quote()` pipeline, and a portal rate watcher.
- `_shipping_for_cart()` (app.py:3373) already calls `shipping.quote()`.
- **But:** all 338 products in `data/products.json` have no `bottle_type`
  assigned, no bottle **dimensions** are stored anywhere, and the
  `box_capacity` matrix is unpopulated. So every order silently falls through
  to the crude qty rule (`_fallback_shipping_cents`: ≤4→$13, 5–12→$23, 13+→$32).
- The existing `pick_box()` uses a fractional-fill heuristic
  (`sum(qty/capacity) ≤ 1.0`) that is geometrically wrong for mixed loads:
  tall droppers and short jars occupy different vertical zones, so their fill
  fractions do not add linearly.

## Goal

Make order-time shipping cost reflect real geometric packing of the bottles,
auto-derived from measured bottle dimensions, with a tunable padding allowance
for protecting glass bottles, and automatic multi-box splitting for oversized
orders. Pricing itself stays where it is (flat rates auto-maintained in the
portal).

## Non-goals

- No change to the flat-rate pricing model, `usps_rates` table, the portal
  rate watcher, or `/admin/shipping`. Rates are read-only inputs to this work.
- No live carrier rate API at checkout (EasyPost stays tracking-only).
- No weight-based logic (weight is explicitly not a constraint).

## Box interiors (cm)

| Box | Interior (cm)   |
|-----|-----------------|
| S   | 5 × 15 × 23     |
| M   | 13 × 22 × 27    |
| L   | 14 × 29 × 30    |

## Bottle types (measured) — Ø × H (cm)

| Key (proposed) | Type                          | Ø  | H  |
|----------------|-------------------------------|----|----|
| `250wm`        | 250 ml wide-mouth             | 8  | 10 |
| `100drop`      | 100 ml dropper                | 5  | 16 |
| `30roll`       | 30 ml roll-on                 | 4  | 10 |
| `50drop`       | 50 ml dropper                 | 4  | 14 |
| `15drop`       | 15 ml dropper                 | 3  | 10 |
| `5drop`        | 5 ml dropper                  | 3  | 8  |
| `100cos`       | 100 ml cosmetic (30 g powder) | 7  | 7  |
| `100wm`        | 100 ml wide-mouth (30 caps)   | 5  | 9  |

These names should reconcile with the human-readable names already used in the
`bottle_types` table during implementation (the existing rows, if any, win on
naming; dimensions are added to them).

## Design

### Component 1 — `dashboard/packing.py` (new, pure-Python)

Geometric packer. No DB, no I/O. Fully unit-testable.

- Models each bottle as a square prism: footprint Ø × Ø, height H.
- Packs upright in horizontal shelves; tries all 3 box orientations (each box
  dimension as the vertical/stacking axis); takes the best.
- Conservative: does not stack a short bottle on top of another inside a tall
  layer, and ignores hex-nesting. "If it says it fits, it fits."
- Box interiors are module constants (the table above).
- Reference implementation already written and self-tested against all 8
  single-type counts (vault: `06 Business Ops/usps-mixed-load-packer.py`):

  | Type    | S  | M  | L   |
  |---------|----|----|-----|
  | 250wm   | 0  | 6  | 9   |
  | 100drop | 3  | 10 | 12  |
  | 30roll  | 6  | 36 | 63  |
  | 50drop  | 5  | 18 | 49  |
  | 15drop  | 10 | 72 | 108 |
  | 5drop   | 10 | 84 | 120 |
  | 100cos  | 0  | 9  | 32  |
  | 100wm   | 6  | 24 | 36  |

  (These are bare-geometry; padding will lower them — see Component 3.)

Public API (shape, finalize during planning):

```python
def pack_into_box(items, box_interior_mm, *, wrap_mm=0) -> int
    # items: list of (diameter_mm, height_mm); returns max that fit (best orientation)

def fits_in_box(items, box_interior_mm, *, wrap_mm=0) -> (bool, int)

def pick_boxes(bottles_by_type, dims_by_type, *, wrap_mm=0, box_margin_mm=0,
               caps=None) -> list[str] | None
    # Returns the chosen box size(s): a single-element list if one box fits,
    # else the fewest boxes (minimize count, then total flat-rate cost) that
    # hold the whole load. None only if a single bottle cannot fit even L.
```

### Component 2 — Data model

- **Extend `bottle_types`:** add `diameter_mm INTEGER` and `height_mm INTEGER`
  (both nullable). Seed/patch the 8 measured types with dimensions
  (cm × 10 → mm).
- **Per-product mapping:** auto-infer each product's `bottle_type` from its
  name/size/category, write into `data/products.json`. Products with no
  confident match get `default` and appear in a review list for Glen to
  correct. A standalone, re-runnable inference script produces the mapping +
  a human-review report; it does not silently overwrite manual corrections on
  re-run (only fills `default`/unset).
- **Padding settings** — new `packing_settings` table (single row), both
  tunable later from breakage feedback:
  - `wrap_mm` — added to each bottle's diameter & height (models
    bubble-wrapping each glass bottle). **Default 6.**
  - `box_margin_mm` — subtracted from each interior dimension (void fill / wall
    cushioning). **Default 10.**

### Component 3 — Packing flow at order time

`shipping.quote(bottles_by_type)` (extended):

1. Resolve each product's bottle type → `(diameter_mm, height_mm)` from
   `bottle_types`. Apply `wrap_mm` to bottle dims; apply `box_margin_mm` to box
   interiors.
2. Geometric `pick_boxes`. If one box fits → return that size + its flat rate
   (unchanged payload shape).
3. If it exceeds L → **auto-split** into the fewest boxes (minimize count, then
   total cost). Sum the flat rates. Return `box_sizes: [...]` plus a per-box
   breakdown and the summed `shipping_cents`. The order proceeds automatically.
4. **Fallback:** any bottle type with no dimensions, or an unknown type, →
   fall back to the existing qty rule so checkout never hard-fails.

`pick_box()` keeps its single-size signature for back-compat (returns the first
element of `pick_boxes` when a single box suffices); callers needing the
multi-box list use `pick_boxes` / the extended `quote` payload.

### Component 4 — Manual override (retained lever)

The existing `box_capacity` matrix is retained as an **optional hard cap**: if a
`(bottle_type, box_size)` cell is set, the packer never places more than that
many of that type in that box, regardless of geometry. Unset = pure geometry.
This is the human escape hatch for "we learned only N fit safely" independent of
the padding knobs. `/admin/shipping` continues to edit this matrix.

### Component 5 — Testing

- `tests/test_packing.py` (new): the 8 single-type counts (must match the table
  above with `wrap_mm=0, box_margin_mm=0`); mixed-load cases; padding lowers
  capacity; multi-box split minimizes count then cost; override cap clamps a
  geometric result; orientation selection.
- Extend `tests/test_shipping.py`: `quote()` single-box (unchanged shape),
  multi-box payload, unknown/dimensionless-type fallback to qty rule, override
  interaction.
- pytest; reuse the `seeded_db` fixture.

## Rollout / tuning

`wrap_mm` and `box_margin_mm` start at 6 and 10 and are tuned from real-world
breakage feedback once the calculator is in use. The override matrix is the
faster per-cell correction when a specific type/box is observed to be a problem.

## Open implementation details (resolve in plan)

- Exact `packing_settings` storage (table vs existing settings mechanism) and
  where the admin edits these two knobs.
- Reconciling proposed keys with any existing `bottle_types.name` rows.
- The product bottle-type inference heuristics + review-report format.
