"""One-shot migration: scrape AANP directory and load into practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \\
    python3 -m scrapers.practitioner_finder.migrate_aanp

AANP (naturopathic.org) uses a YourMembership / AssociationVoice CMS:

  1. The form at /search/custom.asp?id=5613 POSTs to /search/search.asp,
     which returns a shell with an iframe pointing at
     /searchserver/people.aspx?id=<one-shot-session-uuid>.
  2. The iframe page paginates ~25 rows per page, with the total in
     <span id="DocCount">N</span> and pagination via JS __doPostBack.
  3. Each row has /members/?id=<numeric_id> as the canonical detail URL.

Because the directory is state-partitioned (national search returns 5,000+
results that the public-facing grid would never paginate through cleanly),
this migration walks the 50 US states + DC + the 5 Canadian provinces that
have ND licensure, issuing a separate search per state. For each state we:

  a) GET /search/search.asp?txt_state=<State> via Playwright
  b) Extract the iframe session UUID from the returned shell HTML
  c) GET /searchserver/people.aspx?id=<uuid>&... for page 1
  d) For page 2+, scrape __VIEWSTATE + __EVENTVALIDATION out of the
     prior response and POST them back to the same iframe URL with
     __EVENTTARGET set to the pagination control. This is the
     ``__doPostBack`` continuation pattern ASP.NET WebForms uses.
  e) Per-row optionally fetch the detail page (we don't here — the list-
     grid rows are sufficient for the NormalizedPractitionerRow contract).

The site is Cloudflare-protected. All HTTP goes through Playwright so
the cf_clearance cookie is granted once and reused across the whole run.

Idempotent — re-running upserts by source_url. After load, runs the
shared geocoder over any rows still lacking lat/lng.
"""
import re
import sys
from typing import Optional
from urllib.parse import urlencode

from scrapers.practitioner_finder.aanp import (
    BASE,
    SEARCH_URL,
    parse_search_results_html,
    parse_record_count,
    parse_page_info,
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


# US states + DC + the Canadian provinces with established ND licensure.
# We intentionally skip the AANP's own state-chapter sites — the national
# directory is the only source we scrape here, as noted in the playbook.
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]
CA_PROVINCES = [
    "Alberta", "British Columbia", "Manitoba", "Nova Scotia",
    "Ontario", "Saskatchewan",
]


_IFRAME_UUID_RE = re.compile(
    r'/searchserver/people\.aspx\?id=([0-9A-Fa-f-]+)', re.I
)

# ASP.NET WebForms hidden-field extractors.
_VIEWSTATE_RE = re.compile(
    r'<input[^>]*\bname="__VIEWSTATE"[^>]*\bvalue="([^"]*)"', re.I
)
_VIEWSTATEGENERATOR_RE = re.compile(
    r'<input[^>]*\bname="__VIEWSTATEGENERATOR"[^>]*\bvalue="([^"]*)"', re.I
)
_EVENTVALIDATION_RE = re.compile(
    r'<input[^>]*\bname="__EVENTVALIDATION"[^>]*\bvalue="([^"]*)"', re.I
)
# YourMembership renders the page-number pagination as
# <a href="javascript:__doPostBack('ctl00$ContentPlaceHolder1$gvPeople','Page$2')">2</a>
# (the control name varies across deployments — capture the full target).
_DOPOSTBACK_RE = re.compile(
    r"javascript:__doPostBack\(\s*'([^']+)'\s*,\s*'(Page\$\d+)'\s*\)",
    re.I,
)


def _extract_iframe_uuid(shell_html: str) -> Optional[str]:
    """Pull the session UUID out of the iframe ``src`` on a search shell page."""
    m = _IFRAME_UUID_RE.search(shell_html)
    if not m:
        return None
    return m.group(1)


def _extract_hidden_field(pattern: re.Pattern, html: str) -> Optional[str]:
    m = pattern.search(html)
    return m.group(1) if m else None


def _iframe_results_url(session_uuid: str) -> str:
    """First-page GET URL for a freshly-issued iframe session."""
    qs = urlencode(
        {
            "id": session_uuid,
            "cdbid": "",
            "canconnect": "0",
            "canmessage": "0",
            "map": "True",
            "toggle": "True",
            "hhSearchTerms": "",
        }
    )
    return f"{BASE}/searchserver/people.aspx?{qs}"


def _find_doPostBack_target_for_page(
    html: str, page_number: int
) -> Optional[tuple[str, str]]:
    """Locate the (control_name, event_argument) for the link that pages to
    ``page_number`` on the current results page. None if absent."""
    want = f"Page${page_number}"
    for m in _DOPOSTBACK_RE.finditer(html):
        if m.group(2) == want:
            return m.group(1), m.group(2)
    return None


def _build_postback_form(html: str, control: str, argument: str) -> Optional[dict]:
    """Build the form_data payload for a __doPostBack continuation.

    Returns None if any of the three required ASP.NET hidden fields are
    missing (in which case the caller should abort the pagination walk
    for this state — the page can't be replayed without them)."""
    vs = _extract_hidden_field(_VIEWSTATE_RE, html)
    ev = _extract_hidden_field(_EVENTVALIDATION_RE, html)
    if vs is None or ev is None:
        return None
    vsg = _extract_hidden_field(_VIEWSTATEGENERATOR_RE, html) or ""
    return {
        "__EVENTTARGET": control,
        "__EVENTARGUMENT": argument,
        "__VIEWSTATE": vs,
        "__VIEWSTATEGENERATOR": vsg,
        "__EVENTVALIDATION": ev,
    }


# ---------------------------------------------------------------------------
# Live fetch helpers (Playwright-backed)
# ---------------------------------------------------------------------------

def fetch_state_directory_html(
    state: str, fetcher: Optional[PlaywrightFetcher] = None
) -> str:
    """Fetch the rendered search-result SHELL HTML for a single US state.

    Hits ``/search/search.asp?txt_state=<State>`` through Playwright so
    the Cloudflare challenge is solved once and the cookie persists.
    Returns the raw rendered HTML body — the iframe-uuid extractor reads
    the iframe src out of this page.

    ``fetcher`` is required for live calls (run_all-style entrypoints pass
    None, in which case we open a one-shot session — slower but the
    fallback is needed for parity with the existing signature)."""
    qs = urlencode({"txt_state": state})
    url = f"{SEARCH_URL}?{qs}"
    if fetcher is not None:
        return fetcher.get(url)
    with playwright_session() as f:
        return f.get(url)


def fetch_iframe_results_html(
    session_uuid: str, fetcher: Optional[PlaywrightFetcher] = None
) -> str:
    """Fetch the page-1 iframe results page for a previously-issued search."""
    url = _iframe_results_url(session_uuid)
    if fetcher is not None:
        return fetcher.get(url, wait_for_selector="#SearchResultsGrid")
    with playwright_session() as f:
        return f.get(url, wait_for_selector="#SearchResultsGrid")


# ---------------------------------------------------------------------------
# Per-state walker with multi-page __doPostBack replay
# ---------------------------------------------------------------------------

def fetch_rows_for_state(
    state: str, fetcher: Optional[PlaywrightFetcher] = None
) -> list[NormalizedPractitionerRow]:
    """Walk all pages of the AANP directory for a single state.

    Returns the deduplicated list of NormalizedPractitionerRow records
    (deduplication by source_url, since paginating through certain
    YourMembership searches can re-emit the same row at the page boundary).

    Page 2..N use ``__doPostBack`` POST replay: scrape ``__VIEWSTATE`` +
    ``__EVENTVALIDATION`` from page N-1 and POST them back to the same
    iframe URL with ``__EVENTTARGET`` set to the page-N pagination control.
    The Playwright session keeps the cf_clearance cookie warm across all
    of these requests.

    If ``fetcher`` is None we open a one-shot Playwright session — useful
    for run_all's single-function-per-state interface; the migration
    runner passes a long-lived fetcher to amortize browser startup.
    """
    if fetcher is None:
        with playwright_session() as f:
            return fetch_rows_for_state(state, fetcher=f)

    shell_html = fetch_state_directory_html(state, fetcher=fetcher)
    uuid = _extract_iframe_uuid(shell_html)
    if uuid is None:
        # Cloudflare blocked or HTML changed — bail out for this state.
        print(f"  WARN: no iframe uuid found for {state!r}, skipping")
        return []

    seen_urls: set[str] = set()
    out: list[NormalizedPractitionerRow] = []
    page_html = fetch_iframe_results_html(uuid, fetcher=fetcher)
    for r in parse_search_results_html(page_html):
        if r.source_url and r.source_url not in seen_urls:
            seen_urls.add(r.source_url)
            out.append(r)

    page_info = parse_page_info(page_html)
    total_pages = page_info[1] if page_info else 1

    # __doPostBack replay for pages 2..total_pages.
    iframe_url = _iframe_results_url(uuid)
    current_html = page_html
    for page_num in range(2, total_pages + 1):
        target = _find_doPostBack_target_for_page(current_html, page_num)
        if target is None:
            print(
                f"  WARN: {state!r} page {page_num} pagination target not "
                f"found; stopping at page {page_num - 1}"
            )
            break
        form = _build_postback_form(current_html, target[0], target[1])
        if form is None:
            print(
                f"  WARN: {state!r} page {page_num} missing __VIEWSTATE / "
                f"__EVENTVALIDATION; stopping at page {page_num - 1}"
            )
            break
        try:
            current_html = fetcher.post(
                iframe_url,
                form_data=form,
                wait_for_selector="#SearchResultsGrid",
            )
        except Exception as e:  # pragma: no cover - live IO
            print(
                f"  WARN: {state!r} page {page_num} POST replay failed: {e}"
            )
            break
        rows = parse_search_results_html(current_html)
        added = 0
        for r in rows:
            if r.source_url and r.source_url not in seen_urls:
                seen_urls.add(r.source_url)
                out.append(r)
                added += 1
        if added == 0:
            # Either we paged past the real end or the server returned
            # a duplicate page — break to avoid an infinite spin.
            break
    return out


def main() -> int:
    print("Fetching AANP directory state-by-state (Playwright-backed)...")
    all_rows: list[NormalizedPractitionerRow] = []
    seen_urls: set[str] = set()
    with playwright_session() as fetcher:
        for state in US_STATES + CA_PROVINCES:
            print(f"  state={state!r}")
            try:
                rows = fetch_rows_for_state(state, fetcher=fetcher)
            except Exception as e:  # pragma: no cover - live IO
                print(f"  ERROR fetching {state!r}: {e}")
                continue
            for r in rows:
                if r.source_url and r.source_url not in seen_urls:
                    seen_urls.add(r.source_url)
                    all_rows.append(r)
            print(f"    +{len(rows)} rows  (total unique: {len(all_rows)})")

    print(f"\nUpserting {len(all_rows)} rows...")
    for row in all_rows:
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
