# Products catalog enrichment — follow-up queue

Queued work from the 2026-06-05 enrichment session (Glen). The ingredient enrichment
(148 products, from FMP/Formulations, Glen-verified corrections) is DONE and applied to
`data/products.json`. These items remain.

## 1. GK scrape missed a description field — stale list is PROVISIONAL (HIGH)
**Confirmed 2026-06-05:** the GrooveKart scrape (`~/AI-Training/02 Skills/scrape-remedymatch.py`,
namespace `specific-formulations`) captured only part of each product page. The raw HTML for
`macular-wellness-crocin` contains CoQ10, NAC, C3G, lutein — but the scrape's `_extract_body`
dropped them (it flattens the page and trims at footer markers like "Related Products:", and
likely keeps only the short description, missing the full description/ingredient panel).

**Impact:** the 48-item stale-GK list (`data/products-stale-gk-clean.md`) and the `gk_stale`
flags in `products.json` are INFLATED with false positives (the whole macular-wellness family,
and any "GK lists only N of M ingredients" row). Do NOT action the stale list for GK page
updates until this is fixed.

**Fix (cheap — no re-scrape):** the raw cache is intact at `~/Downloads/remedymatch-scrape/raw/`
(321 `.html` files). Re-parse capturing BOTH PrestaShop description fields
(`description_short` + the full `description`/data-sheet body) instead of trimming at the first
footer marker. Then re-run the Stage-B GK diff (`scripts/enrich_products.py --with-gk`) + the
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
