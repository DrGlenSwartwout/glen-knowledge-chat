"""One-shot migration: scrape the NCCAOM Find-a-Practitioner directory
and load into the practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \\
    python3 -m scrapers.practitioner_finder.migrate_nccaom

Single-stage scrape:
  - Walk every (Country, [State]) tuple NCCAOM's CountryCode/StateCode
    dropdowns expose (50 US states + DC + 5 territories + 20 non-US
    countries) at /FAP/SearchResultWithoutMap.
  - Each results page (<=20 practitioners) carries every field we need
    (name, status, credentials via cert-codes, address, phone, website)
    in the card view — no per-record detail fetch needed.

NCCAOM is the largest adapter in the Practitioner Finder fleet (~15-20k
listed diplomates). The on-page disclaimer notes the directory is
"voluntary, not all certified Diplomates will be listed" — total
NCCAOM certified count is ~25,000+, but only the opted-in subset is
publicly searchable.

directory.nccaom.org is Cloudflare-protected. Every HTTP request goes
through the shared Playwright fetcher so the cf_clearance cookie is
granted once and reused across the full ~2,000-page walk. Be defensive:

- 0.5s sleep between fetches (the fetcher handles this)
- If a page returns a Cloudflare challenge HTML (no result cards AND
  also no "0 Practitioners found" banner), retry once after a longer pause
- Log progress every 50 pages so the operator can see status

Idempotent — re-running upserts by source_url (the bare
``/FAP/PractitionerDetail?AgencyClientId=<b64>`` URL is stable across
re-runs since AgencyClientId is the opaque NCCAOM-assigned identifier).
After load, runs the shared geocoder over any rows still lacking
lat/lng.
"""
import sys
from typing import Optional
from urllib.parse import urlencode

from scrapers.practitioner_finder.nccaom import (
    NON_US_COUNTRIES,
    SEARCH_URL,
    US_STATES,
    parse_search_html,
    parse_total_pages,
)
from scrapers.practitioner_finder.geocode import geocode_row, MapboxError
from scrapers.practitioner_finder.db import (
    run_upsert,
    list_ungeocoded,
    update_geocode,
)
from scrapers.practitioner_finder.models import NormalizedPractitionerRow
from scrapers.practitioner_finder.playwright_fetch import (
    PlaywrightFetcher,
    playwright_session,
)


_PROGRESS_EVERY = 50


def _search_url(country: str, state: Optional[str], page: int) -> str:
    params = {
        "Radius": "0",
        "CountryCode": country,
        "SearchType": "2",
        "Latitude": "0",
        "Longitude": "0",
        "SortBy": "DisplayName",
        "SortDirection": "ASC",
        "SearchFormType": "FAP",
        "PageNo": str(page),
    }
    if state:
        params["StateCode"] = state
    return f"{SEARCH_URL}?{urlencode(params)}"


def _looks_like_challenge(html: str) -> bool:
    """Heuristic: True when the response looks like a Cloudflare interstitial.

    The real result page has either practitioner cards (``result-card__item``)
    OR a "0 Practitioners found" banner. A Cloudflare challenge has neither
    AND typically mentions "Just a moment" or a turnstile script.
    """
    if not isinstance(html, str):
        return True
    if "result-card__item" in html or "Practitioners found" in html:
        return False
    if "Just a moment" in html or "challenge-platform" in html or "cf-browser-verification" in html:
        return True
    # Empty or near-empty body — also treat as challenge so we retry once.
    return len(html) < 2000


def fetch_search_page(
    *,
    country: str = "USA",
    state: Optional[str] = None,
    page: int = 1,
    fetcher: Optional[PlaywrightFetcher] = None,
) -> str:
    """Fetch one page of /FAP/SearchResultWithoutMap via Playwright.

    If the response looks like a Cloudflare challenge, sleep briefly and
    retry once with a fresh navigation. Returns whatever HTML the second
    attempt produced (caller is responsible for parsing — if the retry
    also fails, ``parse_search_html`` will simply return [])."""
    url = _search_url(country, state, page)
    if fetcher is None:
        with playwright_session() as f:
            return fetch_search_page(
                country=country, state=state, page=page, fetcher=f
            )
    html = fetcher.get(url, wait_for_selector=".result-card__item, .citySearchList__content, .result-info, body")
    if _looks_like_challenge(html):
        # Brief pause then retry once — Cloudflare's grace period is short.
        fetcher.get(url, wait_for_selector="body", sleep_s=2.0)
        html = fetcher.get(
            url,
            wait_for_selector=".result-card__item, .citySearchList__content, .result-info, body",
        )
    return html


def fetch_all_records() -> list[NormalizedPractitionerRow]:
    """Walk every (Country, [State]) tuple the NCCAOM dropdown exposes
    and return a flat list of NormalizedPractitionerRow. Dedups by
    source_url within the run.

    Strategy:
      1. For Country=USA, walk all 50 states + DC + 5 territories.
      2. For every non-US country in ``NON_US_COUNTRIES``, walk
         Country=<NAME> (state is unavailable for these and search
         returns the country's full list).

    A single Playwright session wraps the entire walk so the
    cf_clearance cookie persists across all ~2,000 page fetches.

    Logs progress every 50 pages so the operator can see status."""
    seen: set[str] = set()
    out: list[NormalizedPractitionerRow] = []
    pages_fetched = 0

    def _log_progress():
        if pages_fetched and pages_fetched % _PROGRESS_EVERY == 0:
            print(
                f"  [NCCAOM] {pages_fetched} pages fetched, "
                f"{len(out)} rows accumulated"
            )

    with playwright_session() as fetcher:
        # US states + DC + territories.
        for st in US_STATES:
            page = 1
            while True:
                html = fetch_search_page(
                    country="USA", state=st, page=page, fetcher=fetcher
                )
                pages_fetched += 1
                rows = parse_search_html(html)
                if not rows:
                    # If the banner says >0 results but we got no rows, we
                    # likely hit a transient challenge — but we already
                    # retried once inside fetch_search_page. Move on.
                    break
                added = 0
                for r in rows:
                    if r.source_url and r.source_url not in seen:
                        seen.add(r.source_url)
                        out.append(r)
                        added += 1
                _log_progress()
                total = parse_total_pages(html)
                if total > 0 and page >= total:
                    break
                if total == 0 or added == 0:
                    # Defensive: no pagination info OR no new rows -> break.
                    break
                page += 1
            print(f"  [NCCAOM] USA / {st}: {len(out)} cumulative rows")

        # Non-US countries.
        for c in NON_US_COUNTRIES:
            page = 1
            while True:
                html = fetch_search_page(
                    country=c, state=None, page=page, fetcher=fetcher
                )
                pages_fetched += 1
                rows = parse_search_html(html)
                if not rows:
                    break
                added = 0
                for r in rows:
                    if r.source_url and r.source_url not in seen:
                        seen.add(r.source_url)
                        out.append(r)
                        added += 1
                _log_progress()
                total = parse_total_pages(html)
                if total > 0 and page >= total:
                    break
                if total == 0 or added == 0:
                    break
                page += 1
            print(f"  [NCCAOM] {c}: {len(out)} cumulative rows")

    return out


def main() -> int:
    print("Fetching NCCAOM directory via directory.nccaom.org (Playwright-backed)...")
    rows = fetch_all_records()
    print(f"  parsed {len(rows)} rows")

    print(f"\nUpserting {len(rows)} rows...")
    for row in rows:
        run_upsert(row.to_dict())
    print("  upsert complete")

    print("\nGeocoding ungeocoded rows...")
    ungeocoded = list_ungeocoded()
    print(f"  {len(ungeocoded)} rows need geocoding")
    geocoded_count = 0
    for r in ungeocoded:
        row_for_geocode = NormalizedPractitionerRow(
            tier="org_member",
            name="X",
            specialties=[],
            address1=r.get("address1"),
            city=r.get("city"),
            state=r.get("state"),
            postal=r.get("postal"),
            country=r.get("country", "US"),
        )
        try:
            lat, lng, quality = geocode_row(row_for_geocode)
            update_geocode(r["id"], lat, lng, quality)
            if lat is not None:
                geocoded_count += 1
        except MapboxError as e:
            print(f"  WARN: geocode failed for {r['id']}: {e}")
    print(f"  successfully geocoded {geocoded_count}/{len(ungeocoded)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
