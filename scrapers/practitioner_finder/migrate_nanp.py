"""One-shot migration: scrape NANP (National Association of Nutrition
Professionals) directory and load into the practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \\
    python3 -m scrapers.practitioner_finder.migrate_nanp

Two-stage scrape (mirrors OEPF):
  Stage 1: walk every results page at
    GET https://mynanp.nanp.org/search/newsearch.asp?cdlMemberTypeID=1705148&...&page=N
    Each page is up to 20 rows of (profile_id, name, city, state, country).
  Stage 2: fetch each
    GET https://mynanp.nanp.org/profile/?ID=<numeric>
    and parse the full practitioner row (practice, address, phone, email,
    website, BCHN credential flag).

mynanp.nanp.org is Cloudflare-protected — every HTTP request goes through
the shared Playwright fetcher so the cf_clearance cookie is granted once
and reused across the whole run.

Idempotent — re-running upserts by source_url (the bare
``/profile/?ID=<numeric>`` URL is stable across re-runs). After load,
runs the shared geocoder over any rows still lacking lat/lng.
"""
import sys
from typing import Optional
from urllib.parse import urlencode

from scrapers.practitioner_finder.nanp import (
    PROFILE_URL,
    SEARCH_PUBLIC_DIRECTORY_PIN,
    SEARCH_MEMBER_TYPE_ID,
    SEARCH_RESULTS_URL,
    parse_member_profile_html,
    parse_search_results_html,
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


def _search_results_url(page: int) -> str:
    qs = urlencode(
        {
            "cdlMemberTypeID": SEARCH_MEMBER_TYPE_ID,
            "cdlCustomFieldValueIDPublicDirectoriesSelection": SEARCH_PUBLIC_DIRECTORY_PIN,
            "page": str(page),
        }
    )
    return f"{SEARCH_RESULTS_URL}?{qs}"


def _profile_url(profile_id: str) -> str:
    return f"{PROFILE_URL}?{urlencode({'ID': profile_id})}"


def fetch_search_results_page(
    page: int, fetcher: Optional[PlaywrightFetcher] = None
) -> str:
    """Hit one page of /search/newsearch.asp via Playwright and return HTML."""
    url = _search_results_url(page)
    if fetcher is not None:
        return fetcher.get(url, wait_for_selector="table")
    with playwright_session() as f:
        return f.get(url, wait_for_selector="table")


def fetch_member_profile_html(
    profile_id: str, fetcher: Optional[PlaywrightFetcher] = None
) -> str:
    """Hit a single /profile/?ID=<n> page via Playwright and return HTML."""
    url = _profile_url(profile_id)
    if fetcher is not None:
        return fetcher.get(url, wait_for_selector="h1")
    with playwright_session() as f:
        return f.get(url, wait_for_selector="h1")


def fetch_all_records() -> list[NormalizedPractitionerRow]:
    """Walk every results page through Playwright, then fetch + parse each
    member profile. Returns a flat list of NormalizedPractitionerRow,
    dedup'd by profile_id within the run.

    A single Playwright session wraps the entire walk so the cf_clearance
    cookie is granted once and reused across stage 1 + stage 2.

    Profiles that fail to render (network blip, missing page, etc.) are
    logged + skipped — sibling profiles are independent.
    """
    seen: set[str] = set()
    stubs: list[dict] = []
    out: list[NormalizedPractitionerRow] = []

    with playwright_session() as fetcher:
        # Stage 1: walk results pages.
        page = 1
        while True:
            html = fetch_search_results_page(page, fetcher=fetcher)
            page_stubs = parse_search_results_html(html)
            if not page_stubs:
                break
            for s in page_stubs:
                pid = s.get("profile_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    stubs.append(s)
            total = parse_total_pages(html)
            if page >= total:
                break
            page += 1

        # Stage 2: fetch + parse each profile.
        for stub in stubs:
            pid = stub["profile_id"]
            try:
                profile_html = fetch_member_profile_html(pid, fetcher=fetcher)
            except Exception as e:  # pragma: no cover - live IO
                print(f"  WARN: NANP profile {pid} fetch failed: {e}")
                continue
            row = parse_member_profile_html(
                profile_html, profile_id=pid, stub=stub
            )
            if row is not None:
                out.append(row)
    return out


def main() -> int:
    print("Fetching NANP directory via mynanp.nanp.org (Playwright-backed)...")
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
