# Regenerative Farm Finder — Design

**Date:** 2026-07-13
**Status:** Phase 0 (pilot) complete; Phases 1–3 pending approval
**Related:** `2026-05-26-practitioner-finder-design.md` (the template this forks)

## Goal

Add a public, map-based directory of **regenerative-agriculture farms &
ranches** to illtowell.com — a "find a regenerative farmer near me" surface
that fits Glen's soil-health / regenerative-food ecosystem (RegenerateVision,
the soil-magnesium narrative, "ill to well").

## Decisions (locked with Glen, 2026-07-13)

1. **Separate surface, not a category inside the practitioner finder.** Farms
   are a different entity (practices/products/ordering, not
   specialties/telehealth). We fork the practitioner-finder architecture into a
   parallel `farms` table + `scrapers/farm_finder/` + `/farm-finder` page rather
   than overloading `practitioners`.
2. **Source: open directories, not Localize.** The original ask named
   "Localize" (localizefood.com), but its farm map/directory is gated behind a
   **paid membership**, delivered mainly through a mobile app, and its ToS
   forbids scraping — so it cannot be sourced legitimately. We instead use open,
   public, scrape-friendly regenerative directories.
3. **Scope this session: plan + pilot scrape.** Write this spec and prove one
   real adapter end-to-end (below). Full ingest/API/UI/cron follow on approval.

## Source selection

Evaluated four open directories. **Food for Humans (findfoodforhumans.com)** wins:

| Source | Verdict |
|---|---|
| **Food for Humans** | ✅ **Chosen.** 1,822 farms (US + Canada), already fully geocoded, 16-marker regenerative practice vocabulary, robots-clean crawl path. |
| Regeneration International | Clean JSON endpoint exists but currently returns 403 to all requests — map is broken. Revisit later. |
| Regenerative Farmers of America | Google MyMaps KML: 6,193 placemarks but only ~8 geocoded; the rest a raw signup dump. Too messy. |
| EatWild | Static HTML, no lat/lng, no explicit regenerative label. Needs full geocoding. Lowest value. |

### Food for Humans — crawl approach (robots-compliant)

`robots.txt`: `Allow: /`, disallows only `/admin/` and `/api/`.

- **Discovery:** `GET /sitemap.xml` → 1,822 `/listing/<slug>/` URLs.
- **Per listing:** the rendered HTML carries a schema.org `LocalBusiness`
  `ld+json` block (name, description, email, telephone, full `PostalAddress`,
  `GeoCoordinates`, `makesOffer[]` products) plus the **regenerative practices**
  and **ordering options** as styled badge `<span>`s grouped under section
  headings. lat/lng arrive pre-geocoded → **no Mapbox step needed** on ingest.
- **NOT used:** `/api/explore/` — it is robots-disallowed *and* its pagination
  is broken from outside (returns the same first 40 rows for every page). We
  only referenced it once to lift the controlled-vocabulary filter lists.

Politeness: single-threaded, `sleep=0.5s` between requests, desktop UA, per-
listing failures isolated.

## Pilot result (Phase 0 — done)

`scrapers/farm_finder/foodforhumans.py` + a fixture test
(`tests/test_ff_foodforhumans.py`, 6 passing) + a live pilot of 15 farms
(`scrapers/farm_finder/pilot_sample.json`). Outcome: **15/15 scraped, 15/15
pre-geocoded, 15/15 carry practices + products + contact.** Example:

> **Meadowdale Farm and Sawmill** — Lenoir City, TN (35.876, -84.320) —
> practices: Pasture-Raised, No Till, Rotational Grazing, Non-GMO,
> Antibiotic-Free … — products: Chicken, Turkey, Eggs —
> meadowdalefarm.com — (802) 380-1014

## Data model (Phase 1)

`migrations/farms.sql` (proposed, **not yet applied**) mirrors `practitioners`:
same `cube`/`earthdistance` radius-search machinery, GIN index on `practices`.

```
farms(
  id uuid pk, source_org text, source_url text UNIQUE (partial, upsert key),
  name text not null, description text,
  practices text[] default '{}',      -- GIN indexed  (regenerative markers)
  products text[] default '{}',
  order_options text[] default '{}',
  phone, email, website, image_url text,
  address1, city, state, postal text, country text default 'US',
  lat numeric(10,6), lng numeric(10,6),
  geocode_quality text,               -- 'source' when the directory provided it
  removal_requested bool default false,
  last_scraped_at, created_at, updated_at timestamptz )
```

Idempotent upsert on `source_url` (same pattern as `practitioner_finder/db.py`).
A `v_farms_public` view filters `removal_requested = false AND lat IS NOT NULL`.

## Search + surface (Phase 2)

- `GET /api/farm-finder/search` — clone of the practitioner search: geocode the
  typed location via Mapbox, radius search, filter by `practices[]`,
  `products[]`, `state`/`country`. Returns `{count, farms[], search_center}`.
- `GET /farm-finder` — Mapbox page cloned from `static/practitioner-finder.html`
  with farm-appropriate filters (practice chips, product chips) and cards
  (products, ordering options, "visit farm website").
- Attribution: each card links back to the Food for Humans listing
  (`source_url`) — good etiquette and good for the source.

## Automation + growth (Phase 3)

- Weekly cron (fork `run_all.py` + a launchd plist) re-crawls the sitemap;
  upsert keeps it idempotent. Rows absent from a fresh crawl are aged out (do
  NOT hard-delete — mark stale, mirroring practitioner conventions).
- Add sources over time behind the same `NormalizedFarmRow` contract
  (Regeneration International once its endpoint is back; American Grassfed).

## Open questions

- **Removal / opt-out:** these are third-party businesses. Reuse the
  practitioner `removal_requested` flag + a contact path? (Recommended.)
- **De-dup across sources** once we add a 2nd directory: key on
  name+lat/lng proximity, or website domain.
- **Consumer vs practitioner framing:** does this live on the public site, in
  the client portal, or both (like the practitioner finder card)?
```
