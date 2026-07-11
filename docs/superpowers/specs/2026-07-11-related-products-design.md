# Related Products — Design Spec

**Date:** 2026-07-11
**Status:** Approved (design), pending spec review
**Feature flag:** `RELATED_PRODUCTS_ENABLED` (Doppler, dark until verified)

## Goal

Add a "related products" section to the bottom of every `/begin/product/<slug>`
page to support solution / program / basket building, where **exploring is an
essential part of the process**. Associations are hybrid: manual "Dr. Glen
recommends" picks plus an automatic list, and Glen curates the manual picks from
a console editor. This is also the surface a later Wishlist project will hang its
"Add to my wishlist" action on.

## Non-goals

- **Wishlist** — a separate project (its own spec) that follows this one.
- Behavioral / "customers also bought" recommendations. Relatedness here is
  Glen's curation (harvested remedymatch merchandising) plus content similarity.
- Any change to the existing formula-page sections or the device-page section
  hiding shipped in #804.

## Data sources & model

Three inputs, merged at render time:

1. **Harvested remedymatch related list** (`data/related-harvested.json`,
   versioned in the repo). A one-time + refreshable script scrapes the
   "Related Products:" block from each catalog product that has a remedymatch
   `url` (304 of them today) and maps each related item to a catalog slug. Shape:
   `{ "<slug>": ["<related-slug>", ...] }`. This is Glen's original storefront
   merchandising and is the primary source of both the console pick-list and the
   auto list.
2. **Manual "Dr. Glen recommends"** (`related-manual.json` on the `/data` disk,
   console-editable). Shape: `{ "<slug>": ["<related-slug>", ...] }`. `products.json`
   is a read-only repo file, so manual curation lives on the data disk like the
   existing `products-page-fixed.json`, and edits need no code change / PR.
3. **Semantic neighbors** (Pinecone). ~976 products carry a `pinecone_title`, so
   nearest-neighbor lookup by the product's own vector fills the auto list after
   the harvested entries. Computed on demand and cached (see Caching).

### Relatedness resolution (per product slug)

- **manual** = `related-manual.json[slug]` (order preserved, as entered).
- **auto** = harvested[slug] first (curation), then semantic neighbors to fill,
  deduped against manual and against each other, guardrail-filtered, capped at 12.

## Guardrails

Applied to the **auto** list only. Manual picks are Glen's explicit choice and
always show, bypassing every guardrail below.

- Exclude the product itself.
- Resolve and exclude `superseded_by` targets (show the survivor, never the
  retired twin — reuse `superseded_slug`).
- Exclude `inactive` products and anything not sellable.
- Exclude the **do-not-recommend** set, centralized in one place:
  Electrolyte Mineral Manna, Living Water (Bottle / Ionizer), plus the
  conditional rules (Fungifuge only after Candida Cleanse; steer AllerFree →
  Immune Modulation). NB the harvested IOP Syntropy list *includes* Living Water
  Ionizer, so this filter is load-bearing.

## Slug mapping (harvest)

remedymatch URLs take two forms:
- `/remedies/<category>/<id>-<slug>` (formulas/remedies)
- `/resources/<id>-<slug>` (books, devices, tools)

Extract the trailing `<slug>`, then resolve to a catalog slug:
1. Exact catalog-slug match.
2. A small **alias map** for hand-added device/book SKUs whose catalog slug
   differs from the storefront slug (e.g. `healing-glaucoma-book` →
   `book-healing-glaucoma`, `denas-microcurrent-system-for-eye-healing` →
   `denas-scenar`, `kloud-mini-pemf-mat` → `kloud-pemf-mini`, living-water →
   the water-ionizer slug). Two-naming-universes applies; the alias map is the
   seam.
3. Unresolved targets are **dropped and logged** to a harvest report for manual
   review, never guessed.

Books and devices ARE shown as related (Glen: "show all related products").

## Console editor

- Route `/console/related-products` (OWNER/OPS), following the Support Programs
  editor pattern.
- Select a product (catalog search). The panel shows:
  - Current **"Dr. Glen recommends"** manual list — removable, reorderable.
  - The **harvested related list** for that product as checkboxes to promote into
    the manual list.
  - A **free catalog search** to add any other product.
- Saves through a console action on the Business-OS dispatch spine
  (`related_products.set`, LOW_WRITE, OWNER/OPS) that writes `related-manual.json`.

## Page rendering

- `begin_product_page_data` gains a `related` section appended at the **bottom**
  (after `cta`, the final content before the footer), gated on
  `RELATED_PRODUCTS_ENABLED`. Body:
  - `featured`: manual picks (labeled "Dr. Glen recommends") + the top 1 auto
    pick. This is the default-visible set.
  - `more`: the remaining auto list (cap 12 total), revealed by a
    **"See more like this"** button that expands **inline** on the same page.
  - The whole section is omitted when `featured` is empty after guardrails.
- Shows for ALL products (formulas and devices) — not gated by `has_ingredients`.
- Each card: product image if available (`page_images` / hero), name, price, and
  a link to `/begin/product/<related-slug>`. Falls back to a text card (name +
  price) when no image exists.
- Frontend: a `renderRelatedBody` in `static/begin-product.html` mirroring the
  existing section renderers, with the inline expand.

## Module boundaries (for testability)

`app.py` is not importable in the test env (needs `pinecone`), so the pure logic
lives in a dedicated module and is unit-tested directly:

- `dashboard/related_products.py`:
  - `resolve_related(slug, *, manual, harvested, semantic, products) -> {featured, more}`
    — the merge/dedup/guardrail/cap/split logic. No I/O.
  - `map_storefront_slug(url, catalog_slugs, aliases) -> slug | None` — the harvest mapper.
  - do-not-recommend set + `superseded`/`inactive` filters reused from `dashboard.products`.
- Harvest script `scripts/harvest_related_products.py` — scrape + map + write
  `data/related-harvested.json` + a harvest report. Parser unit-tested against a
  saved remedymatch HTML fixture (no live network in tests).

## Caching

- Semantic neighbors per slug cached in LOG_DB (like `sales_pages`), refreshed on
  a miss. Harvested + manual are file reads (cheap), read per request.

## Testing

- `resolve_related`: manual-before-auto ordering; dedup manual∩auto; guardrails
  drop self/superseded/inactive/do-not-recommend from auto but NOT from manual;
  featured = manual + 1 auto; more = remainder; cap at 12; empty → section omitted.
- `map_storefront_slug`: both URL forms; alias hits; unmapped → None.
- Harvest parser: extract names+URLs from a saved "Related Products:" fixture.
- Console action: save/load round-trip on a temp `related-manual.json`.

## Rollout

- Land dark behind `RELATED_PRODUCTS_ENABLED`. Harvest, verify the mapping report,
  spot-check the console editor, then render-verify a formula page and a device
  page (headless browser) before flipping the flag in Doppler.

## Open risks

- **Mapping coverage:** some harvested targets are `/resources/...` pages that may
  not be catalog SKUs; those drop out (logged). Acceptable — auto fills from
  semantic neighbors.
- **Scrape politeness:** ~304 pages; the harvest script rate-limits and is run
  ad hoc (not on the request path).
- **Card images:** not every product has an image; text-card fallback covers it.
