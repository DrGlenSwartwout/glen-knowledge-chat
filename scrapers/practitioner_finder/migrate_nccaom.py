"""One-shot migration: scrape the NCBAHM (formerly NCCAOM) Find-a-
Practitioner directory and load into the practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \\
    python3 -m scrapers.practitioner_finder.migrate_nccaom

Single-stage scrape (no per-practitioner detail fetch needed — the
list-grid cards carry every field we need):

  - Walk all 50 US states + DC + 5 territories via the
    ``/FAP/SearchResultWithoutMap`` GET URL with ``StateCode=<XX>`` and
    ``PageNo=<N>`` parameters.
  - For each state: fetch page 1, read ``hdnlastpage`` for the last
    page number, then walk PageNo=2..N.
  - Parse each page's ~20 cards into NormalizedPractitionerRow and
    upsert into the practitioners table AS WE GO (don't accumulate
    20k rows in memory before the first DB write).

Cloudflare Turnstile
--------------------
directory.ncbahm.org gates the search POST with a Cloudflare Turnstile
widget. HEADLESS Playwright FAILS the Turnstile challenge — the JS
solver detects automation and refuses to grant the cf_clearance cookie.

This runner uses ``playwright_session(headless=False)``, which opens a
visible Chromium window. The first navigation to the directory will
prompt Turnstile; once the cookie is granted, the entire walk
(~50 states × ~20 pages average = ~1,000 pages) re-uses the same
session and Turnstile does NOT re-prompt for subsequent navigations
within the same browser context.

If the operator needs to run unattended, they can pre-warm the
``cf_clearance`` cookie in a separate visible session, persist the
browser state, and re-launch in headless mode — but the
out-of-the-box runner just opens a window and lets the user-driven
Turnstile auto-solve do its job.

Per-page sleep is 2.5 seconds — NCBAHM has no public-API
expectation and we want to be polite. At ~1,000 pages × 2.5s that's
~42 minutes of network sleep alone; total wall-clock for a full
nationwide walk is ~50-75 minutes depending on render latency.

Idempotent — re-running upserts by source_url (the bare
``/FAP/PractitionerDetail?AgencyClientId=<b64>`` URL is stable across
re-runs since AgencyClientId is the opaque NCBAHM-assigned identifier).
After load, runs the shared geocoder over any rows still lacking
lat/lng.
"""
import sys
import time
from typing import Optional
from urllib.parse import urlencode

from scrapers.practitioner_finder.nccaom import (
    SEARCH_URL,
    US_STATES,
    parse_search_html,
    parse_total_count,
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


_PROGRESS_EVERY = 25            # log every N pages
_PER_PAGE_SLEEP_S = 2.5         # politeness floor between page fetches
_PER_STATE_SLEEP_S = 3.0        # additional pause between state transitions


def _search_url(state: str, page: int) -> str:
    """Build the GET URL the live NCBAHM directory accepts.

    SortDirection=DESC mirrors the live search-form default (the form's
    sort dropdown defaults to descending DisplayName). PageSize=20 is
    fixed by the back-end."""
    params = {
        "Radius": "0",
        "CountryCode": "USA",
        "StateCode": state,
        "SearchType": "2",
        "Latitude": "0",
        "Longitude": "0",
        "SortBy": "DisplayName",
        "SortDirection": "DESC",
        "SearchFormType": "FAP",
        "PageNo": str(page),
        "PageSize": "20",
    }
    return f"{SEARCH_URL}?{urlencode(params)}"


def _looks_like_challenge(html: str) -> bool:
    """Heuristic: True when the response looks like a Turnstile/Cloudflare
    interstitial rather than a real result page.

    The real result page has either practitioner cards (``result-card__item``)
    OR a "0 Practitioners found" banner. A challenge has neither AND
    typically embeds the Turnstile widget or a CF browser-verification
    interstitial.
    """
    if not isinstance(html, str):
        return True
    if "result-card__item" in html or "Practitioners found" in html:
        return False
    if (
        "cf-turnstile" in html
        or "Just a moment" in html
        or "challenge-platform" in html
        or "cf-browser-verification" in html
    ):
        return True
    # Empty or near-empty body — also treat as challenge so we retry once.
    return len(html) < 2000


# Big-state result pages (CA=2007, FL=1018) render slowly — 20s is too
# tight and caused whole-state aborts on the first run. 60s gives the
# heavy pages room while still bounding a genuinely hung navigation.
_PAGE_TIMEOUT_MS = 60_000
_ROOT_URL = "https://directory.ncbahm.org/"


def _rewarm(fetcher: PlaywrightFetcher) -> None:
    """Re-navigate to the directory root to refresh the cf_clearance
    cookie. Cloudflare Turnstile clearance decays over a long run (the
    first scrape lost the session after ~15 states); re-warming restores
    it. Best-effort — swallows any error."""
    try:
        fetcher.get(_ROOT_URL, wait_for_selector="body", sleep_s=3.0,
                    timeout_ms=_PAGE_TIMEOUT_MS)
    except Exception as e:  # pragma: no cover - live IO
        print(f"    WARN: re-warm failed: {e}")


def fetch_search_page(
    *,
    state: str,
    page: int,
    fetcher: PlaywrightFetcher,
) -> str:
    """Fetch one page of /FAP/SearchResultWithoutMap via Playwright.

    Robust against two failure modes seen on the first full run:
      1. Big-state pages exceeding the default 20s timeout (now 60s).
      2. Cloudflare session decay over a long run — on a challenge or a
         navigation timeout we re-warm (re-solve Turnstile at the root)
         then retry once.

    Returns whatever HTML the final attempt produced (caller parses — if
    the retry also fails, ``parse_search_html`` returns [])."""
    url = _search_url(state, page)
    try:
        html = fetcher.get(
            url,
            wait_for_selector=".result-card__item, .result-info, body",
            sleep_s=_PER_PAGE_SLEEP_S,
            timeout_ms=_PAGE_TIMEOUT_MS,
        )
    except Exception:  # navigation timeout — re-warm + retry
        _rewarm(fetcher)
        html = fetcher.get(
            url,
            wait_for_selector=".result-card__item, .result-info, body",
            sleep_s=_PER_PAGE_SLEEP_S,
            timeout_ms=_PAGE_TIMEOUT_MS,
        )
    if _looks_like_challenge(html):
        # Cookie decayed — re-solve Turnstile at the root, then retry.
        _rewarm(fetcher)
        html = fetcher.get(
            url,
            wait_for_selector=".result-card__item, .result-info, body",
            sleep_s=_PER_PAGE_SLEEP_S,
            timeout_ms=_PAGE_TIMEOUT_MS,
        )
    return html


# ---------------------------------------------------------------------------
# Streaming walker: upsert as we go (don't accumulate 20k rows in memory)
# ---------------------------------------------------------------------------

def _upsert_row(row: NormalizedPractitionerRow) -> None:
    """Single-row upsert wrapper — separate function so tests can monkeypatch."""
    run_upsert(row.to_dict())


def walk_state(
    state: str, fetcher: PlaywrightFetcher
) -> tuple[int, int]:
    """Walk all pages for a single state, upserting as we go.

    Returns ``(rows_emitted, pages_fetched)``. Per-row dedup happens at
    the DB layer (source_url UNIQUE), so we don't carry a per-state seen
    set — if a re-emit occurs at a page boundary the second upsert is
    a no-op."""
    page = 1
    rows_emitted = 0
    pages_fetched = 0
    total_pages = None
    total_count = None

    while True:
        html = fetch_search_page(state=state, page=page, fetcher=fetcher)
        pages_fetched += 1

        if total_pages is None:
            total_pages = parse_total_pages(html)
            total_count = parse_total_count(html)
            if total_count is not None:
                print(
                    f"  [NCCAOM] {state}: total={total_count} pages={total_pages}"
                )

        rows = parse_search_html(html)
        if not rows:
            # Either past the end of results or a challenge slipped
            # through — break either way (next state).
            break

        for r in rows:
            _upsert_row(r)
            rows_emitted += 1

        if pages_fetched % _PROGRESS_EVERY == 0:
            print(
                f"    [NCCAOM] {state} page {page}: cumulative {rows_emitted} rows"
            )

        if total_pages and page >= total_pages:
            break
        if not total_pages:
            # Defensive: no pagination metadata available -> stop after
            # one page rather than spinning forever.
            break
        page += 1

    return rows_emitted, pages_fetched


def main() -> int:
    print(
        "Fetching NCBAHM directory via directory.ncbahm.org "
        "(non-headless Playwright for Cloudflare Turnstile)..."
    )
    total_rows = 0
    total_pages = 0
    # CRITICAL: headless=False so Cloudflare Turnstile auto-solves.
    # Headless mode is detected by the Turnstile JS and BLOCKED.
    with playwright_session(headless=False) as fetcher:
        # Warm-up navigation: visit the directory root to trigger the
        # initial Turnstile challenge and warm the cf_clearance cookie
        # before the first search request.
        print("  warm-up: directory root (auto-solve Turnstile)")
        try:
            fetcher.get(
                "https://directory.ncbahm.org/",
                wait_for_selector="body",
                sleep_s=3.0,
            )
        except Exception as e:  # pragma: no cover - live IO
            print(f"  WARN: warm-up failed: {e}")

        for i, state in enumerate(US_STATES, start=1):
            print(f"\n[{i}/{len(US_STATES)}] {state}")
            try:
                rows, pages = walk_state(state, fetcher=fetcher)
            except Exception as e:  # pragma: no cover - live IO
                print(f"  ERROR fetching {state}: {e}")
                rows, pages = 0, 0
            # A 0-row state is ambiguous: genuinely empty (small territory)
            # OR the Cloudflare session decayed mid-run. Re-warm and retry
            # once — if it's truly empty the retry is cheap; if the session
            # died this recovers it (the first run lost everything after
            # ~15 states this way).
            if rows == 0:
                print(f"  {state}: 0 rows — re-warming + retrying once")
                _rewarm(fetcher)
                try:
                    rows, pages2 = walk_state(state, fetcher=fetcher)
                    pages += pages2
                except Exception as e:  # pragma: no cover - live IO
                    print(f"  ERROR on {state} retry: {e}")
            total_rows += rows
            total_pages += pages
            print(
                f"  {state} done: {rows} rows over {pages} pages "
                f"(cumulative: {total_rows} rows, {total_pages} pages)"
            )
            # Per-state cool-down on top of the per-page sleep.
            if i < len(US_STATES):
                time.sleep(_PER_STATE_SLEEP_S)

    print(
        f"\nUpsert complete: {total_rows} rows emitted over "
        f"{total_pages} pages."
    )

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


def fetch_all_records() -> list[NormalizedPractitionerRow]:
    """Compatibility shim for run_all.py's ``_run_nccaom_scrape`` wrapper.

    Walks every (USA, state) tuple and returns the flat list of
    NormalizedPractitionerRow. Note this loads ALL rows into memory —
    the streaming ``main()`` path above is preferred for production runs
    since the nationwide row count is ~10-15k.
    """
    out: list[NormalizedPractitionerRow] = []
    seen: set[str] = set()

    with playwright_session(headless=False) as fetcher:
        try:
            fetcher.get(
                "https://directory.ncbahm.org/",
                wait_for_selector="body",
                sleep_s=3.0,
            )
        except Exception as e:  # pragma: no cover - live IO
            print(f"  WARN: warm-up failed: {e}")

        for state in US_STATES:
            page = 1
            total_pages: Optional[int] = None
            while True:
                html = fetch_search_page(state=state, page=page, fetcher=fetcher)
                if total_pages is None:
                    total_pages = parse_total_pages(html)
                rows = parse_search_html(html)
                if not rows:
                    break
                for r in rows:
                    if r.source_url and r.source_url not in seen:
                        seen.add(r.source_url)
                        out.append(r)
                if total_pages and page >= total_pages:
                    break
                if not total_pages:
                    break
                page += 1
            time.sleep(_PER_STATE_SLEEP_S)

    return out


if __name__ == "__main__":
    sys.exit(main())
