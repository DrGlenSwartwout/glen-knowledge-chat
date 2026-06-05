# BOS Phase 11: Products catalog enrichment (FMP ingredients + GK descriptions)

**Status:** Reconciliation tool. Source-of-truth DECIDED: `products.json` is canonical (enriched, not replaced).

**Goal:** Enrich `deploy-chat/data/products.json` per slug with authoritative **ingredients** (new FMP -> older Formulations-DB fallback) and **descriptions** (GrooveKart storefront copy from Pinecone), and flag **stale GrooveKart pages** (where the GK page's ingredients differ from FMP, FMP wins). Nothing touches the live `products.json` until Glen reviews the candidate + reports.

**Why staged + review-first:** products.json drives checkout price + (after this) the ingredients/descriptions customers see. Wrong ingredients on a sales page is a real harm. So the tool produces an enriched CANDIDATE + reports; Glen reviews matches + diffs; then we commit.

## Sources (located)
- New FMP ingredients: `~/AI-Training/00 System/fmp-extracts/2026-05-23/` -> `products.csv` (id_pk, product_name), `products_items.csv` (id_fk_product, id_fk_raw, qty, unit_measurement, zc_raw_display), `ingredients.csv` (id_pk, name_common). Authoritative for ingredients.
- Older Formulations DB (fallback): `~/AI-Training/00 System/fmp-extracts/2026-05-24/T33_FORMULAS.csv` -> `Name` (multi-line aliases), `Key Ingredients for Formula` (multi-version free text; the TOP block is current).
- GrooveKart descriptions: Pinecone index `remedy-match-llc`, namespace `specific-formulations`, metadata `title` == products.json `pinecone_title`, chunks ordered by `chunk_index`, text in `metadata["text"]`. (May be STALE for ingredients.)
- products.json: slug -> {name, pinecone_title, price_cents, qbo_item_id}. Join key into all sources is `pinecone_title` (curated for exactly this) or `name`.

## Staged plan

### Stage A (this build): offline matching + ingredient extraction (NO Pinecone, NO products.json write)
`scripts/enrich_products.py`:
- Parse products.json + the FMP CSVs + T33.
- Per slug: match to an FMP product (normalize + fuzzy on `pinecone_title`/`name`, confidence high/medium/low). If matched, build the ingredient list from `products_items` (resolve `id_fk_raw` -> `ingredients.name_common`; carry `qty`+`unit_measurement` or `zc_raw_display`). If NO FMP match, fall back to T33: match `name` against any `\n`-split line of T33 `Name`, take the TOP block of `Key Ingredients for Formula` as the current formula (parse the ingredient lines out of it).
- Output (to `data/` but NOT products.json):
  - `data/products-enrich-candidate.json`: `{slug: {source: "fmp_new|fmp_old_t33|none", confidence, ingredients: [{name, qty, unit, raw}], formula_text?}}`
  - `data/products-enrich-report.md`: counts by source+confidence; the FULL low-confidence + unmatched lists (slug, name, pinecone_title, best guess) for Glen's review.
- Validate: run it, report match-quality stats + 10 sample enriched entries. Reuse the `normalize()` approach from `~/AI-Training/02 Skills/backfill-formulations-from-pinecone.py`.

### Stage B (next build): Pinecone descriptions + GK-vs-FMP ingredient diff
- Add the `specific-formulations` pull per slug (reuse `dashboard/product_content.py` `_page_text` pattern) -> the GK description + the ingredients mentioned on the page.
- Diff GK page ingredients vs FMP ingredients -> a `data/products-stale-gk-report.md` listing products whose GK page is out of date (needs a sales-page update). Run under doppler (Pinecone key).

### Stage C (after Glen's review): commit + module
- Apply the reviewed candidate into products.json (`ingredients` + `description` fields), commit.
- Build the Products module/board: surface ingredients/descriptions, the stale-GK list, and audited `products.adjust_price`/`products.publish` actions; light the Products cell from real data.

## Safety
- Stage A/B are READ-ONLY w.r.t. products.json (write only the candidate + reports).
- Name matching surfaces every low-confidence/unmatched product for human review; no silent mismatches.
- FMP is authoritative for ingredients; GK descriptions are advisory + flagged when stale.

## Self-Review
Scoped to Stage A here (the matching core + reports). Stage B (Pinecone diff) and Stage C (apply + module) are separate builds gated on Glen's review of the Stage A match quality.
