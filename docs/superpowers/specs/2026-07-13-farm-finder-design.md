# Regenerative Farm Finder — Design

**Date:** 2026-07-13
**Status:** Phase 0 (pilot) complete; Phases 1–2 pending approval
**Related:** `2026-05-26-practitioner-finder-design.md`,
`2026-07-01-portal-practitioner-finder-design.md`

## Goal

Add regenerative-agriculture farms & ranches to illtowell.com's finder — a
"find a regenerative farmer near me" capability that fits Glen's soil-health /
regenerative-food ecosystem (RegenerateVision, the soil-magnesium narrative,
"ill to well").

## Decisions (locked with Glen, 2026-07-13)

1. **Integrated as a new top-level CATEGORY in the existing practitioner
   finder** — NOT a separate table/page. Farms are stored as `practitioners`
   rows so they reuse the whole finder pipeline (search, Mapbox map, radius,
   country selector) unchanged. A new parent chip "🌱 Regenerative Farms" sits
   alongside Eye Care / Dental / Holistic Health / Certification / Healing Oasis.
2. **Lives on BOTH surfaces** — public `/practitioner-finder` and the client-
   portal finder card. Free, because both already render the same page; adding
   a category needs no per-surface work.
3. **Source: open directories, not Localize.** The original ask named
   "Localize" (localizefood.com), but its directory is paid-membership + mobile-
   app gated and its ToS forbids scraping. We source from open, public,
   scrape-friendly regenerative directories instead.
4. **Opt-out reuses `removal_requested`** (the existing practitioner flag;
   `v_practitioners_public` already filters it).
5. **Add more sources when available** (Regeneration International once its
   endpoint is back, American Grassfed, etc.) behind the same adapter contract.
6. **Scope this session: plan + pilot scrape** (Phase 0, below).

## How the integration works (no backend query changes)

The finder's frontend maps every parent category chip to `specialties[]` (only
'certification' maps to `tier[]`) and searches with `specialties && %s`. So a
farm is a `practitioners` row where:

- `tier = 'farm'` — a new tier value; the card renderer branches on it to show a
  farm card (products, ordering options, "visit farm website") instead of a
  clinician card.
- `specialties = ['regenerative_farms', <slug per practice>]` — the parent tag
  `regenerative_farms` makes the "Regenerative Farms" chip match every farm;
  each practice slug (`pasture_raised`, `rotational_grazing`, `grass_fed`, …) is
  a filterable sub-chip. Both resolve through the existing specialty filter with
  **zero** query/endpoint changes.
- `products[]`, `order_options[]` — new farm-only array columns (NULL for
  clinicians).
- name / bio(description) / phone / email / website / photo_url / address /
  lat / lng map straight across. Coordinates arrive exact from the source →
  `geocode_quality = 'full'` (no Mapbox geocoding needed on ingest).

Schema change: `migrations/practitioners-farms.sql` (PROPOSED, not yet applied)
— rebuild the tier CHECK to add `'farm'`, add `products` + `order_options`.

## Source selection

Evaluated four open directories. **Food for Humans (findfoodforhumans.com)** wins:
1,822 farms (US + Canada), already fully geocoded, a 16-marker regenerative
practice vocabulary, and a robots-clean crawl path. (Regeneration International:
clean JSON but currently 403s. Regenerative Farmers of America: Google MyMaps,
only ~8 of 6,193 geocoded. EatWild: static HTML, no lat/lng, no regenerative
label.)

### Food for Humans — crawl approach (robots-compliant)

`robots.txt`: `Allow: /`, disallows only `/admin/` and `/api/`.
- Discovery: `GET /sitemap.xml` → 1,822 `/listing/<slug>/` URLs.
- Per listing: parse the schema.org `LocalBusiness` `ld+json` (name,
  description, email, telephone, `PostalAddress`, `GeoCoordinates`,
  `makesOffer[]` products) + the practice/ordering badge `<span>`s grouped by
  section heading. lat/lng are pre-geocoded.
- NOT used: `/api/explore/` (robots-disallowed, and its pagination is broken
  from outside — same first 40 rows every page). Referenced once only to lift
  the controlled-vocabulary filter lists.
- Politeness: single-threaded, `sleep=0.5s`, desktop UA, per-listing failures
  isolated. Each farm card links back to its source listing (attribution).

## Pilot result (Phase 0 — done)

`scrapers/farm_finder/`:
- `models.py` `NormalizedFarmRow` (adapter output)
- `foodforhumans.py` adapter (sitemap → listing parse; pure `parse_listing`)
- `mapping.py` `to_practitioner_row()` — the farm → practitioners-row mapping
  described above
- `export_pilot.py` — pilot runner
- `pilot_sample.json` (farm-native) + `pilot_sample_practitioner_rows.json`
  (the integrated shape actually written to `practitioners`)
- `tests/test_ff_foodforhumans.py` — 9 passing (parser + mapping), fixture-
  driven, no network.

Live pilot of 12 farms: **12/12 scraped, 12/12 pre-geocoded, 12/12 with
practices + products + contact**, each mapped to a `tier='farm'` row. Example:

> **Meadowdale Farm and Sawmill** — Lenoir City, TN (35.876, -84.320) —
> specialties: `regenerative_farms, pasture_raised, no_till, rotational_grazing,
> non_gmo, …` — products: Chicken, Turkey, Eggs — ordering: Farm Pickup, Bulk
> Orders — meadowdalefarm.com — (802) 380-1014

## Remaining work

**Phase 1 — ingest (CODE DONE; prod write pending)**
- `scrapers/farm_finder/ingest.py` crawls → maps → upserts via the column-generic
  `practitioner_finder/db.run_upsert` (writes `products`/`order_options`
  automatically). Idempotent on `source_url`. **Dry-run by default**; `--apply`
  writes. Dry run validated (8/8 mapped, all geocoded).
- **Two gated prod actions remain (need Glen / a non-bg run):**
  1. Apply `migrations/practitioners-farms.sql` to prod Supabase.
  2. `python3 -m scrapers.farm_finder.ingest --apply` — full crawl of 1,822
     listings into `practitioners`.

**Phase 2 — surface (DONE)**
- `static/practitioner-finder.html`: "🌱 Regenerative Farms" parent chip +
  practice sub-chips; card + side-panel branch on `tier==='farm'` (Offers,
  Ordering, prettified practice badges, direct contact, source attribution; no
  inquiry checkbox/note). Clickable website links added to all result cards
  (gold, `target=_blank`, http(s)-scheme-guarded) — previously side-panel only.
- Render-verified headless (list + farm side panel) against the real card path.
  Serves both the public page and the portal embed (same file → free).

**Phase 3 — automation + growth (automation DONE)**
- Weekly re-crawl: added `_run_farm_scrape` as the `foodforhumans_farms` adapter
  in the EXISTING `scrapers/practitioner_finder/run_all.py` (not a separate cron)
  — so farms re-crawl in the same Sunday 11 pm HST launchd run, inheriting its
  per-run logging, failure isolation, and Glen-notification. Idempotent upsert;
  no aggressive staleness removal (mirrors the practitioner adapters). The global
  geocode sweep skips farms (pre-geocoded) and the GHL sync skips them
  (`WHERE tier='org_member'`), so no config change was needed.
- De-dup: `scrapers/farm_finder/dedupe.py` (`dedupe_farms`) collapses the same
  farm across sources by website domain, else name-slug + coarse coords. A no-op
  for one source; plugs in when a 2nd adapter's rows are combined before upsert.
- **Deferred (needs an available source):** a 2nd directory. Regeneration
  International has the ideal clean JSON endpoint but currently 403s; revisit and
  add behind the `NormalizedFarmRow` contract + `dedupe_farms` when it returns.

## Open questions

- **Card CTA for farms:** link straight to the farm's own website/phone (no
  in-app inquiry). Assumed yes.
- **Country default:** farms are US + Canada; the finder defaults to US with a
  country selector — no change needed.
```
