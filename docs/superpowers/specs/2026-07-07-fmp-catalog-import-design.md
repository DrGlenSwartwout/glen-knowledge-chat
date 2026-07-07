# FMP → Sellable Catalog Import — Design

**Date:** 2026-07-07
**Status:** Draft for Glen's review (decisions flagged below)
**Repo:** deploy-chat (`data/products.json` + a local reconciliation script)

## Problem

The sellable catalog (`data/products.json`, 328 active) is the single source for what can be **priced, invoiced, and added to an order** (`_get_product(slug)` and the "Add Product" list read it). FMP (`fmp_snap_products`, 1187 rows) is Glen's master product list. **~1088 FMP products are absent from the catalog** — so remedies Glen actually sells cannot be put on an invoice (surfaced when Green Jasper Gem Elixir was skipped on Donna Banks' order). This is a systemic data gap, not a one-off.

## Scope of the gap (missing, by FMP `type`)

| FMP type | Missing | In scope? |
|---|---|---|
| Essence (gem elixirs, flower & chakra essences) | 396 | **Yes** — remedies |
| Functional Formulation | 93 | **Yes** — core remedies |
| Infoceutical | 83 | Decision |
| Homeopathic | 48 | Decision |
| Tincture | 19 | **Yes** — remedies |
| Simple Solution | 16 | Decision |
| Gemmotherapy | 12 | Decision |
| Spirit Mineral | 10 | Decision |
| Pure Powders | 9 | **Yes** — remedies |
| Product | 295 | **No** (mixed: equipment, bottles, machines) |
| Book | 67 | **No** |
| Service | 15 | **No** (handled separately, e.g. Biofield Analysis) |

Filtering to **active + priced + remedy-type** already yields ~413 clear imports (394 Essence + 19 Tincture) before the FFs and the "decision" types.

## Approach

A local, idempotent reconciliation script (run under `doppler`, reads `~/AI-Training` FMP snapshot, writes `data/products.json`):

1. **Select** `fmp_snap_products` rows where `active='Yes'`, `sold_price` is set, `type` ∈ the approved whitelist, and the product's name/slug is **not already** in `products.json` (dedupe by lowercased name and slug).
2. **Map** each to a catalog entry:
   - `slug` = a clean slugify of `product_name`, de-collided (append `-2`, etc., on conflict; log every collision).
   - `name` = `product_name`; `pinecone_title` = `product_name`.
   - `price_cents` = `round(sold_price * 100)` (FMP `sold_price`; see decision on which price field).
   - `qty_pricing` = per the FF-eligibility rule (see decision).
   - `fmp_id` = `id_pk`; `ingredients_source` = `"fmp_snap"`.
   - `description` = composed from `healing_qualities` + `indications` + the dosage display (no ingredient list — essences don't carry one).
   - No `bottle_type` (matches existing no-dims remedies like `b17-syntropy`); shipping-dims decision below.
3. **Emit a review artifact** first: `data/fmp-import-preview.json` + a one-line-per-product summary (slug, name, price, type) and the collision log. Glen reviews before anything goes live.
4. **Apply**: on approval, merge the entries into `products.json`, commit, PR, deploy. The invoice button's exact-name match then resolves every imported remedy automatically.

## Decisions needed from Glen

1. **Type whitelist** — confirmed remedies (Essence, Functional Formulation, Tincture, Pure Powders) plus which of the "decision" types (Infoceutical, Homeopathic, Simple Solution, Gemmotherapy, Spirit Mineral)?
2. **Price field** — `sold_price` (retail, used for Green Jasper) vs `retail_sug_price`? Green Jasper: sold 70 / suggested 80.
3. **FF volume eligibility (`qty_pricing`)** — FFs: yes. Do essences / gem elixirs / tinctures also get the FF volume rate, or list price only?
4. **Shipping** — essences are 50ml dropper bottles. Omit `bottle_type` for now (safe; pickup and manual shipping unaffected), or define a dropper bottle_type + dims for the packer? (Can be a fast-follow.)
5. **GrooveKart pages** — imported items have no GK page. Mark them so they don't show as "stale GK" false-positives; they're in-house/invoice-sellable only until pages exist.

## Scope / rollout

- v1 = import the approved remedy types as sellable catalog entries (invoice-ready). **Not** in scope: GrooveKart page generation, ingredient enrichment, images. Those are separate follow-ons.
- Idempotent: re-running never duplicates (dedupe by name/slug), so it can be re-run as FMP grows.
- One reviewed batch → one PR → one deploy. Green Jasper (already added manually) is the pattern for a single entry.

## Testing

- The mapper is a pure function `fmp_row -> catalog_entry`; unit-test the field mapping, price conversion, slug de-collision, and the dedupe filter against a small fixture.
- Post-apply: assert `products.json` still validates as JSON and the active count rose by the expected N; spot-check three imported slugs resolve via `/api/orders/price-preview`.
