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

### Stage A.1 (fallback fix, before B): FMP-empty -> Formulations DB
The new FMP is not fully populated: some products match an FMP product by name but have ZERO `products_items` rows (e.g. AllerFree, ACES Eye Drops, Brain Boost). Per Glen's rule ("for those not in the newest FMP, the older Formulations database is the most recent"), treat an FMP match with NO ingredients the SAME as no FMP match: fall through to the T33 (older Formulations DB) ingredients. Keep the FMP identity match, but source the ingredient list from T33. New source value: `fmp_new_empty->t33` (or just `t33` with a note). Re-run; report the improved coverage.

### Stage B: Pinecone descriptions + the full ingredient priority + stale-GK diff
**Ingredient source priority (Glen): new FMP -> older Formulations DB (T33) -> GrooveKart.** Some products have ingredients ONLY on the GK sales page (no FMP, no T33 record) -- for those, GK is the ingredient source.
- Pull per slug from Pinecone `specific-formulations` (reuse `dashboard/product_content.py` `_page_text`): the GK **description** (always) + the **ingredients mentioned on the page** (parse the Contents/ingredient panel out of the concatenated page text: lines with mg/mcg/IU/% or a "Contents:"/"Ingredients:" block).
- Resolve final ingredients per product by priority: FMP-new (non-empty) -> T33 (non-empty) -> GK-parsed. Record `ingredients_source`.
- **Diff** the GK page ingredients against the authoritative set (FMP or T33) ONLY where an authoritative set exists -> `data/products-stale-gk-report.md`: products whose GK page differs from FMP/T33 (the GK sales page needs updating; FMP/T33 wins). Show added/removed ingredient names.
- **Flag GK-only** products (ingredients_source=GK, no FMP/T33 record) in the report as "GK-only, unverified" so Glen knows these aren't cross-checked.
- Output the updated `data/products-enrich-candidate.json` (now with `description`, final `ingredients`, `ingredients_source`) + the stale-GK report. Run under doppler (Pinecone + OpenAI embed keys). STILL no products.json write.

### Stage C (after Glen's review): commit + module
- Apply the reviewed candidate into products.json (`ingredients` + `description` fields), commit.
- Build the Products module/board: surface ingredients/descriptions, the stale-GK list, and audited `products.adjust_price`/`products.publish` actions; light the Products cell from real data.

## Safety
- Stage A/B are READ-ONLY w.r.t. products.json (write only the candidate + reports).
- Name matching surfaces every low-confidence/unmatched product for human review; no silent mismatches.
- FMP is authoritative for ingredients; GK descriptions are advisory + flagged when stale.

## Self-Review
Scoped to Stage A here (the matching core + reports). Stage B (Pinecone diff) and Stage C (apply + module) are separate builds gated on Glen's review of the Stage A match quality.
