# Products catalog enrichment — follow-up queue

Queued work from the 2026-06-05 enrichment session (Glen). The ingredient enrichment
(148 products, from FMP/Formulations, Glen-verified corrections) is DONE and applied to
`data/products.json`. These items remain.

## 1. GK scrape stale list — FIXED 2026-06-05 (PR #40)
**Done.** Re-parsed the intact raw cache (`scripts/reparse_gk.py`: green-plus `<p>` bullets,
mapped by filename slug) + re-judged the 44 flagged products (synonym-aware parallel pass) +
applied (`scripts/apply_reparse.py`). Trustworthy result: **40 stale** (36 GK pages genuinely
missing current advanced actives + 4 un-reverified), 8 cleared, 38 flagged `gk_has_extra`.
neuromagnesium cleared (Glen updated its page). See #1b below for the un-reverified 4.

## 1b. 4 un-reverified stale (LOW)
`iron-out`, `nerve-repair`, `man-manna`, `snake--dragon` had no green-plus panel parsed (their
GK pages use a different ingredient format, or a slug mismatch). Still flagged from the OLD
(possibly inflated) diff. Re-check their GK page format + re-judge.

## 1c. 38 products: GK lists MORE than our FMP/Formulations record (DECISION)
`gk_has_extra` on 38 products — our authoritative ingredient list is PARTIAL vs the GK page.
Glen's rule was "FMP/Formulations wins", but here GK is the richer source. Decision: merge GK's
extra ingredients into the authoritative `ingredients` (and re-verify against Glen's true current
formula), or leave FMP-as-authoritative and accept the gaps. `gk_has_extra` lists the candidates per product.

## (was 1, superseded) original scrape-field note — kept for context
The scrape's `_extract_body` dropped the full ingredient panel (`~/AI-Training/02 Skills/scrape-remedymatch.py`).
Raw cache intact at `~/Downloads/remedymatch-scrape/raw/` (321 `.html`). Re-parse recovers it; a future
re-ingestion of the FULL panels into Pinecone `specific-formulations` would also fix the chatbot/card copy. To re-run the Stage-B GK diff (`scripts/enrich_products.py --with-gk`) + the
parallel cleanup synonym-diff to regenerate a TRUSTWORTHY stale list. Update `products.json`
`gk_stale` flags from the corrected diff.

## 2. Single-ingredient Pure Powders pass (LOW — Glen deferred)
Not high priority. Many single-ingredient powders are not yet listed on GK (the active list is
partial), so hold off. Later: go through the full Pure Powder list and set each one's ingredient
to the product itself, EXCLUDING the highly hygroscopic ones (Glen to flag those). Pattern proven
on humic-acid + magnesium-taurate (done as manual corrections).

## 3. Held products needing review (8) — not yet enriched
From `scripts/apply_enrichment.py` HOLD (low-confidence/wrong/bundle matches):
- `heart-health` — WRONG match (Rhythm Section/Restore is a DIFFERENT product, Glen). Needs its real formula.
- `dry-eye-relief-program`, `glucose-tolerance-program`, `macular-wellness-program` — BUNDLES/programs (multiple products), not single formulas. Model as bundles (list component products), not an ingredient list.
- `c15-syntropy-pentadecanoic-acid` — low-confidence T33 match (0.79); verify it's the right formula.
- `serenity` ("Serene Blue Green") — low-confidence (0.73) vs "Serenity Blue Green Balance"; verify.
- (`humic-acid`, `magnesium-taurate` — RESOLVED as single-ingredient Pure Powders.)

## 4. Empty-after-cleanup (6) — matched a source but produced no clean ingredients
6 of the 159 matched products came back with an empty clean ingredient list (vague/blend-only
GK data). Identify + fill manually or from a better source.

## 5. 162 recipe-less products
Mostly intentionally ingredient-less (Infoceuticals, Flower Essences, EI/ET/ES/ED/MB codes).
A subset are single raw ingredients (see #2) or bundles (see #3). No action for the true essences/infoceuticals.

## 6. Description cleanup (LOW)
The applied `description` fields are lightly-cleaned raw GK copy (breadcrumb + price stripped,
trimmed to 600 chars). A per-product description-cleanup pass (like the ingredient cleanup) would
improve them. Also blocked on #1 (the missing field may hold better description copy).

## 7. Products module — price editing (deferred by design)
`/console/products` is read + the stale-page work queue (`products.mark_page_fixed`). Price/field
EDITING needs a `DATA_DIR` override overlay (products.json is a read-only repo file at runtime).
Build the overlay + `products.adjust_price` when needed.

## Confirmed stale (real, once #1 is verified)
- `neuromagnesium` — GK page genuinely missing P5P + Vitamin D3 (Glen confirmed it needs updating).
