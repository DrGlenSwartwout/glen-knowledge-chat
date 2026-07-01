# Practitioner Finder in the Client Portal — Design

**Date:** 2026-07-01
**Status:** Draft for review
**Owner:** Glen (illtowell.com / deploy-chat)

## Goal

Surface the **existing** public Practitioner Finder inside each client portal as a
card, auto-located to the client (their zip or city), with international fallback
and all categories shown.

## Key finding — reuse, don't build

A full practitioner finder already exists and is production:
- Page `/practitioner-finder` (`static/practitioner-finder.html`), Mapbox map,
  ~27k practitioners incl. international.
- API `GET /api/practitioner-finder/search` (zip/city → geocode → radius search),
  `GET /api/practitioner-finder/countries`, multi-select inquiry.
- Categories already present as chips: Eye Care, Dental, Holistic Health,
  Certification, Healing Oasis (+ sub-specialties). Default view shows all.
- International handled: country selector + "International (anywhere)" (`country=ANY`).

The client's three literal requirements — **zip-or-city search, international
fallback, all categories** — are already satisfied by that page. So this work is
an **iframe embed + auto-locate prefill**, not a new finder.

Same-origin framing confirmed OK: no `X-Frame-Options`/CSP `frame-ancestors`
anywhere in the app.

## Scope (approved)

**Embed + auto-locate.** Behind flag `PORTAL_FINDER_ENABLED` (dark until flipped).
Chrome-hide (`?embed=1`) is out of scope for v1.

## Architecture

Client address is already in the portal view at `account.address`
(`_ADDRESS_KEYS` = address1, address2, city, state, zip, country;
`dashboard/portal_view.py:172-179`). Prefill is built from it — no new data source.

The finder page runs searches via `runSearch()` reading `#location-input` and
`#country-select` (`static/practitioner-finder.html:454,450,671`); it does **not**
currently read a location from its own URL. The only finder-page change is: on
load, read `?location=`/`?country=`, set those inputs, and call `runSearch()`.

### Components / files

1. **`dashboard/portal_view.py`** — new pure helper
   `_practitioner_finder_block(address: dict, enabled: bool) -> dict` returning
   `{"enabled": bool, "location": <zip or city or "">, "country": <country or "US">}`.
   Zip preferred over city (more precise); empty when no address (finder falls back
   to its own empty/US default). Included in `get_portal_view(...)` return under key
   `"practitioner_finder"`.

2. **`app.py`** — define `PORTAL_FINDER_ENABLED = _env_bool("PORTAL_FINDER_ENABLED")`
   and pass `finder_enabled=PORTAL_FINDER_ENABLED` into the `get_portal_view(...)`
   call site (mirrors how `offers_enabled_keys` is threaded).

3. **`static/client-portal.html`** — in `render(d, v)`, when
   `v.practitioner_finder && v.practitioner_finder.enabled`, append a card:
   - Title "Find a Practitioner Near You"
   - `<iframe loading="lazy">` whose `src` = `/practitioner-finder` +
     (`?location=<enc>&country=<enc>` when a location is present)
   - Responsive height (e.g. `min-height:640px; height:80vh`), full width.
   - Placement: directly under the "Ask Dr. Glen" chat card (top, prominent).

4. **`static/practitioner-finder.html`** — after the country `<select>` is
   populated, read `new URLSearchParams(location.search)`; if `location` present,
   set `#location-input.value` (and `#country-select.value` when `country` present
   and the option exists), then call `runSearch()`. Backward-compatible: no params
   → unchanged behavior (public page untouched).

### Data flow

portal `/api/portal/<token>/view` → `get_portal_view` → `practitioner_finder`
block → client-portal.html builds iframe `src` → finder page reads params →
auto-`runSearch()` → results located to the client, all categories shown.

## Testing

- `_practitioner_finder_block`: zip present → `location==zip`; no zip but city →
  `location==city`; neither → `location==""`; country passthrough with `US`
  default; `enabled` reflects the flag. (pytest, pure — no `import app`.)
- String test on `client-portal.html`: contains the finder card + `/practitioner-finder` iframe, gated on `practitioner_finder.enabled`.
- String test on `practitioner-finder.html`: reads `URLSearchParams` for `location`
  and calls `runSearch()` on load.
- Render-verify (owner): flip `PORTAL_FINDER_ENABLED`, open a portal whose client
  has an address → finder card shows, prefilled, results load; a portal with no
  address → card shows, finder default (type-to-search); zero console errors;
  international location returns results.

## Non-goals / deferred

- Chrome-hide `?embed=1`; prefilling the inquiry form with portal identity;
  a true tabbed portal nav (portal stays card-based); any change to the finder's
  search/data.

## Risks

- Country option may not be loaded when prefill runs → set country after the
  `/countries` populate step; if the client's country isn't in the list, leave the
  select at its default and still prefill the location text.
- Nested iframe (portal → finder → its own `/embed` chat) adds weight; acceptable
  for v1 (chrome-hide would drop it later).
