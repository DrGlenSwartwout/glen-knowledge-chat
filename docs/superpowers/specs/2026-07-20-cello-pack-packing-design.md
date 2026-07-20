# Cello-Pack Packing — Correct Shipping Rate + Separate Bottle/Cello Counts

**Date:** 2026-07-20
**Repo:** deploy-chat
**Status:** Design — pending review

## Summary

Some units ship as **cello packs** (the existing "Cellophane refill packs" format —
capsules in cellophane, **no rigid bottle**), which pack tighter than bottles and cost
less to ship. Today the packer counts every unit as a rigid bottle, so shipping
over-estimates (e.g. order #49: 27 units quoted $55 when the cello-aware cost is ~$32).

This pass makes the shipping **rate** correct for cello packs and carries a **separate
count of bottles vs cello packs** through pricing and the display surfaces. EasyPost /
label automation (weights, parcel dimensions) is explicitly **out of scope** this pass.

## Decisions locked (Glen, 2026-07-20)

1. **A cello pack is the existing format** `refill` — "Cellophane refill packs", capsules
   only, no bottle (`app.py:6125` `_FORMATS`). Not a multipack of intact bottles.
2. **Per-line format picker, splittable.** Each order line carries one format; a product
   can be split between bottle and cello by appearing on **two lines** (same slug, one
   `format:"bottle"`, one `format:"refill"`). No intra-line quantity split.
3. **Rate + separate counts this pass.** EasyPost parcel weight/dimension wiring is a
   later phase, not this one.

## Background — how shipping is computed today

(From the engine map; all deterministic, no external service on the cost path.)

- A **geometric bin-packer** over three USPS flat-rate boxes S/M/L
  (`dashboard/packing.py`), keyed by a per-product **`bottle_type`**; box choice priced
  from the confirmed `usps_rates` table (`dashboard/shipping.py`; S $12.65 / M $22.95 /
  L $31.50). `pick_boxes` (`shipping.py:325`) has three tiers: Rae's **capacity matrix**
  (authoritative per-box counts), else **geometric** dims, else a legacy fractional rule.
- **Dimensions live on the `bottle_type`** (SQLite `bottle_types`, hand-built via
  `/admin/shipping`), NOT on the product. `resolve_bottle_type` (`shipping.py:750`).
- `_price_cart` (`app.py:6490`) builds **`box_counts = {bottle_type: qty}`** plus a
  scalar **`total_bottles`**, then `_shipping_for_cart` → `_shipping.quote(box_counts)`
  (`app.py:6445`). The per-line branch is at `app.py:6519-6575`.
- **`format` already exists but is cosmetic.** A cart line may carry
  `format ∈ {bottle, larger, refill}`; inside `_price_cart` it only decorates the
  display/QBO description (`app.py:6527-6532`) and does **not** affect `bottle_type`,
  `box_counts`, or shipping. The storefront sets it (`begin_checkout`, `app.py:9505`);
  the **console order form has no format control at all**.
- **Counts** come from `physical_units` (`dashboard/orders.py:593`) /
  `_order_physical_units` (`app.py:6286`), surfaced as a single scalar in ~6 places
  (see Blast radius).

## Requirements

### R1 — A cello packing unit
- Introduce a cello `bottle_type` (e.g. `"cello-refill"`) representing a capsules-only
  cellophane pack, added to the prod bottle vocabulary (`PROD_BOTTLE_NAMES`,
  `shipping.py:80`) and to the runtime `bottle_types` / capacity data.
- **Sizing approach:** because a cello pack is soft/flat, prefer a **capacity-matrix row**
  (how many cello packs fit per S/M/L box — Rae's authoritative tier in `pick_boxes`)
  over geometric Ø/H dims. Geometric dims are the fallback if a matrix isn't supplied.
- **Data input needed (not a code decision):** the per-box cello-pack capacity (or the
  pack's effective dims). Glen/Rae provides — measured, entered via `/admin/shipping`.
  Until entered, cello lines fall back to today's behavior (no regression).

### R2 — Wire format → packing unit + separate counter
- In `_price_cart` (`app.py:6527-6575`), when a line's `format` is the cello/refill
  format, resolve its packing key to the cello `bottle_type` instead of the product's
  bottle type, so `box_counts` reflects the tighter unit and the quote is correct.
- Carry a **`total_cello_packs`** counter parallel to `total_bottles` (the two never
  merge). Thread it through the priced result so callers can display each separately.
- Bundles containing cello components honor the same mapping.

### R3 — Per-line format picker on the console order form
- Add a **format selector** to each line in `static/order-new.html` (Product /
  Qty / Unit / … / **Format**), values Standard bottle / Cellophane refill pack (reuse
  `_FORMATS`). Default `bottle`.
- The line payload (`linesPayload()`) includes `format`; the create + edit endpoints
  pass it into the pricer (they already accept arbitrary line fields; `_price_cart`
  reads `format`).
- **Splitting:** to ship part of a product as cello, the operator adds the product a
  second time with the cello format. Two lines, same slug, different formats — supported
  by the existing per-line model; no schema change.

### R4 — Separate bottle vs cello counts through counting + display
- **Do NOT change `physical_units`' meaning.** It stays *total shippable product units*
  (a cello unit and a bottle unit both count as one product unit) because it feeds
  **reorder/backorder demand**, which is packaging-agnostic — you restock the same
  capsules whether they ship in a bottle or a cello pack. Changing it would silently
  drop reorder demand.
- **Add a per-format breakdown** for the **fulfillment / shipping** surfaces only —
  `{bottles, cello_packs}` derived from the line formats — so pick/pack and the order
  board show what to physically grab. This is the "counted separately for automation"
  requirement, scoped to where packaging matters.
- Fulfillment surfaces render the split (e.g. "24 bottles + 3 cello packs"); the
  reorder rollup keeps the combined product-unit count unchanged.

### R5 — Label/description (already present)
- The invoice/QBO line already labels the cello format via `_FORMAT_LABELS`
  (`app.py:6527`). Keep it; no new work beyond confirming it reads for console-created
  lines too.

## Non-goals (this pass)

- **EasyPost / label automation** — `build_shipment` (`dashboard/easypost.py:46`) uses a
  crude flat weight, sends no dimensions, and ignores the packer entirely. Making labels
  cello-aware needs **weights** (no weight field exists anywhere today) and the packer's
  box dims fed to the parcel. Deferred to a later phase.
- **Intra-line quantity split** (one line = N bottles + M cello). Splitting is done with
  two lines (Decision 2).
- No change to the rate table, box definitions, or the pickup rule.

## Data flow

```
order-entry line: {slug, qty, format:"refill"}   (console form: new Format picker)
      │
      ▼
_price_cart (app.py:6527-6575)
   format==cello ? key = cello bottle_type : product bottle_type
   box_counts[key] += qty
   format==cello ? total_cello_packs += qty : total_bottles += qty
      │
      ▼
_shipping.quote(box_counts)  → cello unit packs tighter → correct shipping_cents
      │
      ▼
counts: {bottles: total_bottles, cello_packs: total_cello_packs}  → display surfaces
```

## Blast radius — count display surfaces (each must show the split)

| Surface | File:line |
|---|---|
| Order board annotate | `app.py:42262-42268` |
| Console order render | `static/console-orders.html:244, 478` |
| Order-entry "units to ship" | `app.py:40463-40468`; `static/order-new.html:95, 370` |
| Order/invoice API payloads | `app.py:39153, 39176, 39301, 41249` |
| Reorder/backorder rollup | `dashboard/orders.py:589, 639`; `app.py:42380` |

Each shows a single scalar today; the split model requires each to render bottles and
cello packs distinctly (or a combined "N bottles + M cello packs" string).

## Testing
- Pricing: a cart of N cello-format lines quotes cheaper than the same N as bottles, and
  matches the cello capacity/dims; a mixed cart (bottles + cello) prices each correctly.
- Split: same slug on two lines (bottle + cello) prices and counts both, independently.
- Counts: `total_bottles` and `total_cello_packs` are separate and correct across
  bottle-only, cello-only, and mixed carts; membership/service still count 0 for both.
- Regression: a cart with no cello lines prices and counts exactly as today.
- UI: the format picker round-trips through create + edit; default is bottle.

## Open data input (not a code blocker for the spec)
- The cello pack's **per-box capacity** (preferred) or **effective dimensions** — from
  Rae, entered via `/admin/shipping`. The code ships with a safe fallback (cello lines
  behave as today until the data is entered), so the build isn't blocked on it.

## Phasing
- **This pass:** R1–R5 (rate correct + separate counts + console picker).
- **Later:** EasyPost automation — weights on packing units + packer box dims into
  `build_shipment`, once label automation is turned on.
