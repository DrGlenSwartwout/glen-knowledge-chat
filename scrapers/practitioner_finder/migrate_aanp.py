"""One-shot migration: scrape AANP directory and load into practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \\
    python3 -m scrapers.practitioner_finder.migrate_aanp

AANP (naturopathic.org) uses a YourMembership / AssociationVoice CMS:

  1. The form at /search/custom.asp?id=5613 POSTs to /search/search.asp,
     which returns a shell with an iframe pointing at
     /searchserver/people2.aspx?id=<one-shot-session-uuid>.
  2. The iframe page paginates ~24 cards per page (live layout), with
     the total record count in <span id="DocCount">N</span> (or
     "1000+" for unbounded queries) and pagination via JS __doPostBack.
  3. Each card has /members/?id=<numeric_id> as the canonical detail URL.
     The card itself carries only name + city/state/postal — street,
     phone, email, website, credentials, practice_name all live on the
     /members/?id=<id> profile page and are merged in below.

Because the directory is state-partitioned (national search returns 5,000+
results that the public-facing grid would never paginate through cleanly),
this migration walks the 50 US states + DC + the 5 Canadian provinces that
have ND licensure, issuing a separate search per state. For each state we:

  a) GET /search/search.asp?txt_state=<State> via Playwright
  b) Extract the iframe session UUID from the returned shell HTML
  c) GET /searchserver/people2.aspx?id=<uuid>&... for page 1
  d) For page 2+, scrape __VIEWSTATE + __EVENTVALIDATION out of the
     prior response and POST them back to the same iframe URL with
     __EVENTTARGET set to the pagination control whose rendered button
     text matches the desired page number. The live layout uses
     ``SearchResultsGrid$ctlNN$ctlMM`` event-target names (NOT the
     older ``Page$N`` event-argument style) — we match by button text
     to stay robust against the ctl-index numbering.
  e) For each unique member_id collected across all pages, fetch the
     /members/?id=<id> profile page and merge the profile fields
     (street, phone, email, website, credentials, practice_name) into
     the stub row from the list page.

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
    DIRECTORY_FORM_URL,
    SEARCH_URL,
    parse_profile_html,
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
    r'/searchserver/people2\.aspx\?id=([0-9A-Fa-f-]+)', re.I
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
# Live YourMembership renders the page-number pagination as
# <button onclick="javascript:__doPostBack('SearchResultsGrid$ctl28$ctl02','')">2</button>
# (the control name varies across deployments — capture the full target).
# The event argument is empty on the live layout; the rendered button
# TEXT is the page number we match against.
_DOPOSTBACK_BUTTON_RE = re.compile(
    r"""<button[^>]*onclick=["']javascript:__doPostBack\(\s*&#39;([^&]+)&#39;\s*,\s*&#39;([^&]*)&#39;\s*\)["'][^>]*>(.*?)</button>""",
    re.I | re.S,
)
# Same pattern but with raw single-quotes (the live HTML uses raw quotes
# inside double-quoted onclick attrs, not HTML-encoded entities).
_DOPOSTBACK_BUTTON_RAW_RE = re.compile(
    r"""<button[^>]*onclick=["']javascript:__doPostBack\(\s*'([^']+)'\s*,\s*'([^']*)'\s*\)["'][^>]*>(.*?)</button>""",
    re.I | re.S,
)
# Legacy anchor-based pagination (kept as a fallback for the pre-migration
# layout in case the site ever serves a mixed/cached response):
#   <a href="javascript:__doPostBack('ctl00...','Page$2')">2</a>
_DOPOSTBACK_ANCHOR_RE = re.compile(
    r"""<a[^>]*href=["']javascript:__doPostBack\(\s*'([^']+)'\s*,\s*'([^']*)'\s*\)["'][^>]*>(.*?)</a>""",
    re.I | re.S,
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
    return f"{BASE}/searchserver/people2.aspx?{qs}"


def _strip_button_text(raw: str) -> str:
    """Strip inner <i>...</i> / whitespace from a button label so we can
    match the visible page-number text."""
    s = re.sub(r"<[^>]+>", "", raw or "")
    return s.strip()


def _is_next_arrow_html(raw: str) -> bool:
    """Detect the right-arrow forward-paging button (rendered as
    ``<i class="fa fa-arrow-right">``)."""
    return "fa-arrow-right" in (raw or "").lower()


def _find_doPostBack_target_for_page(
    html: str, page_number: int
) -> Optional[tuple[str, str]]:
    """Locate the (control_name, event_argument) for the pagination
    control whose rendered button text matches ``page_number`` on the
    current results page. None if absent.

    The live layout uses ``<button>`` controls with raw single-quoted
    onclick attrs and an empty event-argument (the second __doPostBack
    arg). The legacy layout used ``<a>`` controls with a ``Page$N``
    event-argument. We try buttons first (live), then anchors (legacy),
    then HTML-entity-encoded buttons (defensive — in case the response
    is ever served through an entity-encoding proxy).

    Note: only ~10 page-number buttons render at once on the live
    layout. For pages outside the visible numeric window, use
    ``_find_next_arrow_target`` to step forward one page at a time."""
    want = str(page_number)

    for pattern in (_DOPOSTBACK_BUTTON_RAW_RE, _DOPOSTBACK_BUTTON_RE, _DOPOSTBACK_ANCHOR_RE):
        for m in pattern.finditer(html):
            control = m.group(1)
            argument = m.group(2)
            text = _strip_button_text(m.group(3))
            # Live form: text is the page number.
            if text == want:
                return control, argument
            # Legacy form: event argument is 'Page$N'.
            if argument == f"Page${page_number}":
                return control, argument
    return None


def _find_next_arrow_target(html: str) -> Optional[tuple[str, str]]:
    """Locate the (control_name, event_argument) for the forward-arrow
    pagination button. None if absent (e.g. on the last page).

    The forward arrow renders as ``<button ...><i class="fa fa-arrow-right">``
    on the live layout."""
    for pattern in (_DOPOSTBACK_BUTTON_RAW_RE, _DOPOSTBACK_BUTTON_RE):
        for m in pattern.finditer(html):
            if _is_next_arrow_html(m.group(3)):
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
    """Fetch the page-1 iframe results page for a previously-issued search.

    The live layout renders results into ``<ul id="search-results">``;
    the legacy layout used ``<table id="SearchResultsGrid">``. We wait
    for whichever exists — Playwright's selector engine accepts the
    comma-joined OR form for this."""
    url = _iframe_results_url(session_uuid)
    selector = "ul#search-results, table#SearchResultsGrid"
    if fetcher is not None:
        return fetcher.get(url, wait_for_selector=selector)
    with playwright_session() as f:
        return f.get(url, wait_for_selector=selector)


def fetch_profile_html(
    member_id: str, fetcher: Optional[PlaywrightFetcher] = None
) -> str:
    """Fetch a single member's /members/?id=<id> profile page via Playwright."""
    url = f"{BASE}/members/?id={member_id}"
    if fetcher is not None:
        return fetcher.get(url, wait_for_selector="#tdEmployerName")
    with playwright_session() as f:
        return f.get(url, wait_for_selector="#tdEmployerName")


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

    # __doPostBack replay for pages 2..total_pages. The live layout only
    # renders ~10 page-number buttons at a time — for page numbers outside
    # that visible window we fall back to the right-arrow control which
    # advances exactly one page per click.
    iframe_url = _iframe_results_url(uuid)
    current_html = page_html
    for page_num in range(2, total_pages + 1):
        target = _find_doPostBack_target_for_page(current_html, page_num)
        if target is None:
            target = _find_next_arrow_target(current_html)
        if target is None:
            print(
                f"  WARN: {state!r} page {page_num} pagination target not "
                f"found (no number button and no next-arrow); stopping at "
                f"page {page_num - 1}"
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
                wait_for_selector="ul#search-results, table#SearchResultsGrid",
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


_MEMBER_ID_FROM_URL_RE = re.compile(r"/members/\?id=(\d+)")


def _member_id_from_url(source_url: Optional[str]) -> Optional[str]:
    if not source_url:
        return None
    m = _MEMBER_ID_FROM_URL_RE.search(source_url)
    return m.group(1) if m else None


def _merge_profile_into_stub(
    stub: NormalizedPractitionerRow,
    profile: NormalizedPractitionerRow,
) -> NormalizedPractitionerRow:
    """Layer a profile-page row over a list-page stub, taking profile-
    derived fields as authoritative when present.

    Locked invariants (tier, source_org, specialties, source_url) are
    fixed by the stub. fellowship_level is taken from the profile (which
    has access to the Credentials custom field). Other fields fall back
    to the stub when the profile field is None — this lets the list-page
    city/state/postal cover cases where the profile employer block is
    blank but the card still has location info."""
    def _pick(profile_v, stub_v):
        return profile_v if profile_v is not None else stub_v

    return NormalizedPractitionerRow(
        tier=stub.tier,
        name=profile.name or stub.name,
        specialties=list(stub.specialties),
        source_org=stub.source_org,
        source_url=stub.source_url,
        fellowship_level=profile.fellowship_level or stub.fellowship_level,
        practice_name=_pick(profile.practice_name, stub.practice_name),
        credentials=_pick(profile.credentials, stub.credentials),
        phone=_pick(profile.phone, stub.phone),
        email=_pick(profile.email, stub.email),
        website=_pick(profile.website, stub.website),
        address1=_pick(profile.address1, stub.address1),
        city=_pick(profile.city, stub.city),
        state=_pick(profile.state, stub.state),
        postal=_pick(profile.postal, stub.postal),
        country=profile.country or stub.country or "US",
    )


def main() -> int:
    print("Fetching AANP directory (Playwright-backed, single unfiltered walk)...")
    stubs_by_url: dict[str, NormalizedPractitionerRow] = {}
    with playwright_session() as fetcher:
        # Warm-up: the /search/search.asp endpoint returns an empty body
        # on the first hit of a fresh session — it requires the search-form
        # session cookie set by visiting /search/custom.asp first.
        print("  warm-up: /search/custom.asp?id=5613")
        try:
            fetcher.get(DIRECTORY_FORM_URL)
        except Exception as e:  # pragma: no cover - live IO
            print(f"  WARN: warm-up failed: {e}")

        # Single unfiltered walk — the txt_state URL param doesn't actually
        # filter on the live site (the form fills it via JS at submission
        # time), so any state value walks the full ~46 pages of results.
        # 1 state iteration is enough; the per-state loop just re-walks
        # the same ~1,100 rows.
        try:
            rows = fetch_rows_for_state("OR", fetcher=fetcher)
        except Exception as e:  # pragma: no cover - live IO
            print(f"  ERROR fetching directory: {e}")
            rows = []
        for r in rows:
            if r.source_url and r.source_url not in stubs_by_url:
                stubs_by_url[r.source_url] = r
        print(f"    +{len(rows)} stubs collected ({len(stubs_by_url)} unique)")

        print(
            f"\nEnriching {len(stubs_by_url)} stub rows with per-member "
            f"profile pages..."
        )
        all_rows: list[NormalizedPractitionerRow] = []
        for i, (url, stub) in enumerate(stubs_by_url.items(), start=1):
            member_id = _member_id_from_url(url)
            if member_id is None:
                # No id to fetch — keep the stub as-is.
                all_rows.append(stub)
                continue
            try:
                profile_html = fetch_profile_html(member_id, fetcher=fetcher)
            except Exception as e:  # pragma: no cover - live IO
                print(f"  WARN: profile fetch failed for {member_id}: {e}")
                all_rows.append(stub)
                continue
            profile_row = parse_profile_html(profile_html, member_id=member_id)
            if profile_row is None:
                all_rows.append(stub)
                continue
            all_rows.append(_merge_profile_into_stub(stub, profile_row))
            if i % 50 == 0:
                print(f"    enriched {i}/{len(stubs_by_url)}")

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
