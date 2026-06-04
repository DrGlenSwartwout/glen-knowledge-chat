# Practitioner Finder — International Expansion

**Date:** 2026-06-04
**Status:** Approved design, ready for implementation plan
**Author:** Glen + Claude
**Related:** `docs/superpowers/specs/2026-05-26-practitioner-finder-design.md` (original finder), Tier-1 adapter spec (2026-05-29)

---

## Problem

The practitioner finder is country-aware at the data layer (`practitioners.country` column, `run_search(countries=...)`, and several adapters that capture non-US countries), but it is locked to the United States in practice:

1. The search API (`app.py` `practitioner_finder_search`) hardcodes `countries=["US"]`.
2. The same endpoint requires a 5-digit US `zip` to geocode the map center.
3. The frontend (`static/practitioner-finder.html`) has a ZIP-only input (`pattern="\d{5}"`) with no country selector.

### Audit (run 2026-06-04 against prd Supabase)

- **26,928 practitioners total.** US = 25,804. **~1,124 international across 60+ countries** already in the table (CA 419, GB 83, KR 80, AU 66, AE 47, MX 43, PH 23, IN 21, IL 20, SG 19, CZ 19, … plus 40+ countries with single-digit counts).
- Every international row is **currently invisible** to the finder.
- **~50 rows have un-normalized `country` values**: full names ("Slovenia", "Guatemala", "Puerto Rico", "Iran (Islamic Republic of)", "Croatia (Hrvatska)"), typos ("United Sates", "U.S.A"), and one ambiguous "Georgia". Exact-match country filtering (`country = ANY(%s)`) mis-handles these.
- Geocode coverage is uneven for some countries (PH 6/23, MY 3/8, DE 6/9, ZA 8/13) — those rows will not surface in a radius search until geocoded.

---

## Goal

Make the ~1,124 existing international practitioners discoverable through the finder, with a search experience that works outside the US, **without** adding new scraper sources (deferred to separate specs). Normalize the country data as part of the work so filtering is correct and future rows stay clean.

Non-goals (explicitly out of scope):
- New international directory adapters (separate spec per source).
- GHL prospect sync changes for international rows.
- Localization / translation of the finder UI.

---

## Approach (selected: A)

**Country filter + a "Country-wide" radius option.** Selecting a country filters results to that country and biases geocoding; the radius dropdown gains a "Country-wide / no distance limit" choice that returns all practitioners in the selected country sorted by distance from the typed place. This preserves today's US radius behavior while handling sparse countries (a patient in Germany sees the nearest practitioner even if it is 200 km away) instead of returning empty results.

Rejected alternatives:
- **B — pure radius, no country filter:** mixes countries near borders; sparse countries return nothing at a 25-mi radius.
- **C — country filter, fixed radius only:** simplest UI, but sparse countries frequently return zero results.

---

## Design

### 1. Country normalization — `scrapers/practitioner_finder/normalize.py`

Add `normalize_country(raw: str | None) -> str | None`:

- Trims and upper-cases input.
- Returns a 2-letter ISO 3166-1 alpha-2 code when resolvable, else the original value unchanged (never silently drops data).
- Maps the known US aliases ("U.S.A", "United Sates", "UNITED STATES", "U.S.", "USA") → `US`, plus a curated full-name → ISO-2 table covering the messy values found in the audit (Slovenia→SI, Guatemala→GT, Venezuela→VE, Lebanon→LB, Iraq→IQ, Cyprus→CY, Luxembourg→LU, "Iran (Islamic Republic of)"→IR, Guyana→GY, Tanzania→TZ, "Croatia (Hrvatska)"→HR, Jamaica→JM, Zimbabwe→ZW, Latvia→LV, Barbados→BB, Paraguay→PY, Mauritius→MU, "Dominican Republic"→DO, Jordan→JO).
- Already-valid 2-letter codes pass through unchanged (upper-cased).
- **Ambiguous "Georgia" (1 row):** treat as the country `GE` (the column is a country column), and log it in the migration report so Glen can correct it if it was a mislabeled US state.
- **"Puerto Rico" (5 rows):** normalize to `PR` (distinct ISO code) rather than folding into `US`, so it can be selected on its own; the geocoder's `mapbox_country_filter` continues to treat PR addresses correctly.

This is the single source of truth for country normalization. The existing `geocode._US_COUNTRY_ALIASES` / `mapbox_country_filter` logic stays as-is for geocode bias, but `normalize_country` is what the upsert path and the migration use to write the stored column.

**Wire into upsert:** `scrapers/practitioner_finder/db.py` applies `normalize_country` to the `country` field inside `upsert_sql_and_params` (or `run_upsert` before building params) so every new scraped row stores a clean value. Adapters are unchanged — they keep emitting whatever the source gives; normalization is centralized at the write boundary.

### 2. One-time migration + audit — `scrapers/practitioner_finder/migrate_normalize_country.py`

- Selects all distinct `country` values, computes `normalize_country` for each, and `UPDATE`s rows whose stored value differs.
- Prints a before/after report (the audit deliverable): distinct value count, rows changed, and any value that could not be resolved to an ISO-2 code (flagged for manual review, including the "Georgia" case).
- After normalization, re-runs the global geocode sweep (reuse `run_all._global_geocode_sweep`) so newly-consistent international rows with missing coordinates get geocoded.
- Idempotent: a second run reports zero changes.
- Invoked manually via `doppler run … -- python3 -m scrapers.practitioner_finder.migrate_normalize_country`.

### 3. Free-text geocode helper — `scrapers/practitioner_finder/geocode.py`

Add `geocode_place(place: str, country_iso: str | None) -> tuple[float | None, float | None]`:

- Forward-geocodes an arbitrary free-text location string ("Berlin", "Tokyo", "90210", "London, UK") through Mapbox.
- Applies the country bias via `mapbox_country_filter(country_iso)` when a country is given.
- **Bypasses `detect_geocode_quality`** (which requires city+state or a postal and would reject a bare city name). Returns `(None, None)` when Mapbox yields no feature.
- Reuses the existing throttle, token resolution, and error handling (`MapboxError`).

`geocode_row` is unchanged — it stays the structured-row path used by adapters and the global sweep.

### 4. Search API — `app.py` `practitioner_finder_search`

- New query params:
  - `country` — ISO-2, default `"US"`. Special value `"ANY"` (or empty) means international/anywhere → pass `countries=None` to `run_search` (no country filter, pure radius).
  - `location` — free-text place. **Back-compat:** if `location` is absent but `zip` is present, use `zip` as the location string.
- Geocode the center with `geocode_place(location, country)` instead of constructing a structured `PfRow`.
- Radius:
  - A normal numeric `radius_miles` behaves as today.
  - `radius_miles` = `"country-wide"` (or a sentinel like `0`/`max`) → use a very large radius (e.g. 12,500 mi, larger than any single country) so the earthdistance circle effectively becomes "everything in the country", still distance-sorted. The country filter does the real scoping.
- `run_search` is called with `countries=[country]` unless `country` is `ANY`/empty (then `None`). No change to `db.build_search_sql` — it already supports `countries`.
- Error messages updated to not assume a US ZIP ("could not locate that place" instead of "could not locate zip X").

### 5. Frontend — `static/practitioner-finder.html`

- Replace the ZIP-only field with:
  - A **country `<select>`** defaulting to United States. Options populated from the countries actually present in the DB. Implementation: a new lightweight endpoint `GET /api/practitioner-finder/countries` returns `[{code, name, count}]` (distinct normalized countries with counts), rendered with friendly names via a small ISO→name map; the dropdown also offers a top "International (anywhere)" option mapping to `country=ANY`.
  - A **free-text location input** (drop `pattern="\d{5}"`, relabel "City, postal code, or address"). Required.
- The radius `<select>` gains a "Country-wide" option (value maps to the API sentinel).
- JS submit handler sends `country` + `location` (+ `radius_miles`) instead of `zip`.
- Empty-state and error copy updated to be country-neutral.

### 6. Tests

- `tests/test_pf_normalize.py` (extend): `normalize_country` — ISO passthrough, US aliases, full-name table, typos, unresolved passthrough, "Georgia"→GE, "Puerto Rico"→PR.
- New `geocode_place` test (mock Mapbox response): country bias param set, bare-city query succeeds, no-feature → `(None, None)`.
- `tests/test_pf_search_api.py` (extend): `country` param flows to `run_search`; `country=ANY` → `countries=None`; `location` used as center; `zip` back-compat; "country-wide" radius maps to large radius.
- Migration: a small unit over `normalize_country` mapping table is sufficient; the DB `UPDATE` itself is integration-tested manually against prd.

---

## Data flow (international search)

```
User picks country=DE, location="Berlin", radius="Country-wide"
  → GET /api/practitioner-finder/search?country=DE&location=Berlin&radius_miles=country-wide
  → geocode_place("Berlin", "DE")  [Mapbox, country=de bias]  → (52.52, 13.40)
  → run_search(lat,lng, radius_miles=12500, countries=["DE"], …)
  → rows in DE, distance-sorted from Berlin
  → JSON → map + list
```

US search is unchanged in behavior: `country=US` (default), `location` = a ZIP, normal radius.

---

## Risks / notes

- **Geocode coverage gaps** mean some international rows won't appear until the post-migration sweep geocodes them; the migration report surfaces remaining gaps.
- **Country dropdown counts** reflect only geocodable+present rows; a country with rows but zero geocoded coordinates will show in the list but return no map pins until geocoded. Acceptable for v1; the sweep closes the gap over time.
- **`ANY` / international mode** does an unfiltered radius search; with a "Country-wide" radius that would return the whole table, so the UI restricts "Country-wide" to when a specific country is chosen (ANY mode keeps a finite radius selector).
- No change to GHL sync, inquiry, or claim flows — they operate on practitioner ids regardless of country.
