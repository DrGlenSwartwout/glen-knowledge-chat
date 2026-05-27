"""One-shot migration: scrape NANP directory and load into practitioners table.

Run via:
  doppler run --project remedy-match --config prd -- \\
    python3 -m scrapers.practitioner_finder.migrate_nanp

NANP (mynanp.nanp.org) uses the SAME YourMembership / AssociationVoice CMS
as AANP (naturopathic.org). The two are sister deployments on the same
vendor and the iframe-card scrape pattern is byte-for-byte identical:

  1. Warm-up GET ``/search/`` to set the session cookies. (NANP's
     warm-up URL is the bare ``/search/`` path, not the AANP-style
     ``/search/custom.asp?id=NNN`` form.)
  2. GET ``/search/newsearch.asp`` which returns a shell with an iframe
     pointing at ``/searchserver/people2.aspx?id=<one-shot-session-uuid>``.
  3. GET the iframe URL: 24 cards / page, ``<span id="DocCount">N</span>``
     total, ``Page X of Y`` pagination, ASP.NET WebForms ``__doPostBack``
     replay for pages 2..N. The pagination control names are the same
     ``SearchResultsGrid$ctlNN$ctlMM`` form as AANP.
  4. For each unique ``/members/?id=<numeric>`` URL collected across all
     pages, fetch the profile page and merge profile fields (street,
     phone, email, website, credentials, BCHN, practice_name) into the
     stub row from the list page.

The mynanp.nanp.org subdomain is Cloudflare-protected. All HTTP goes
through Playwright so the cf_clearance cookie is granted once and reused
across the whole run.

The live NANP form doesn't filter by state via URL params either — the
``txt_state`` query param is a no-op (same as AANP). We do a single
unfiltered walk that returns all ~673 active practitioners.

Idempotent — re-running upserts by source_url. After load, runs the
shared geocoder over any rows still lacking lat/lng.
"""
import re
import sys
from typing import Optional
from urllib.parse import urlencode

from scrapers.practitioner_finder.nanp import (
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


_IFRAME_UUID_RE = re.compile(
    r'/searchserver/people2\.aspx\?id=([0-9A-Fa-f-]+)', re.I
)

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
_DOPOSTBACK_BUTTON_RE = re.compile(
    r"""<button[^>]*onclick=["']javascript:__doPostBack\(\s*&#39;([^&]+)&#39;\s*,\s*&#39;([^&]*)&#39;\s*\)["'][^>]*>(.*?)</button>""",
    re.I | re.S,
)
_DOPOSTBACK_BUTTON_RAW_RE = re.compile(
    r"""<button[^>]*onclick=["']javascript:__doPostBack\(\s*'([^']+)'\s*,\s*'([^']*)'\s*\)["'][^>]*>(.*?)</button>""",
    re.I | re.S,
)
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
    """Strip inner <i>...</i> / whitespace from a button label."""
    s = re.sub(r"<[^>]+>", "", raw or "")
    return s.strip()


def _is_next_arrow_html(raw: str) -> bool:
    """Detect the right-arrow forward-paging button."""
    return "fa-arrow-right" in (raw or "").lower()


def _find_doPostBack_target_for_page(
    html: str, page_number: int
) -> Optional[tuple[str, str]]:
    """Locate the (control_name, event_argument) for the page-number
    pagination control whose rendered button text matches ``page_number``.

    Only ~10 page-number buttons render at once on the live layout. For
    pages outside the visible numeric window, use the right-arrow control
    (``_find_next_arrow_target``) to step forward one page at a time.
    """
    want = str(page_number)

    for pattern in (_DOPOSTBACK_BUTTON_RAW_RE, _DOPOSTBACK_BUTTON_RE, _DOPOSTBACK_ANCHOR_RE):
        for m in pattern.finditer(html):
            control = m.group(1)
            argument = m.group(2)
            text = _strip_button_text(m.group(3))
            if text == want:
                return control, argument
            if argument == f"Page${page_number}":
                return control, argument
    return None


def _find_next_arrow_target(html: str) -> Optional[tuple[str, str]]:
    """Locate the (control_name, event_argument) for the forward-arrow
    pagination button. None if absent (e.g. on the last page)."""
    for pattern in (_DOPOSTBACK_BUTTON_RAW_RE, _DOPOSTBACK_BUTTON_RE):
        for m in pattern.finditer(html):
            if _is_next_arrow_html(m.group(3)):
                return m.group(1), m.group(2)
    return None


def _build_postback_form(html: str, control: str, argument: str) -> Optional[dict]:
    """Build the form_data payload for a __doPostBack continuation.

    Returns None if any of the three required ASP.NET hidden fields are
    missing (in which case the caller should abort the pagination walk —
    the page can't be replayed without them)."""
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

def fetch_search_shell_html(
    fetcher: Optional[PlaywrightFetcher] = None,
) -> str:
    """Fetch the rendered search-shell HTML via Playwright.

    Hits ``/search/newsearch.asp`` so the Cloudflare challenge is solved
    once and the cookie persists. Returns the raw rendered HTML body —
    the iframe-uuid extractor reads the iframe src out of this page.
    """
    if fetcher is not None:
        return fetcher.get(SEARCH_URL)
    with playwright_session() as f:
        return f.get(SEARCH_URL)


def fetch_iframe_results_html(
    session_uuid: str, fetcher: Optional[PlaywrightFetcher] = None
) -> str:
    """Fetch the page-1 iframe results page for a previously-issued search."""
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
# Single-walk pagination walker with multi-page __doPostBack replay
# ---------------------------------------------------------------------------

def fetch_all_stubs(
    fetcher: Optional[PlaywrightFetcher] = None,
) -> list[NormalizedPractitionerRow]:
    """Walk all pages of the NANP directory in a single unfiltered pass.

    Returns the deduplicated list of NormalizedPractitionerRow records
    (deduplication by source_url). The NANP form doesn't filter by state
    via URL params — a single unfiltered walk yields the full ~673-row
    directory across ~29 pages.

    Page 2..N use ``__doPostBack`` POST replay: scrape ``__VIEWSTATE`` +
    ``__EVENTVALIDATION`` from page N-1 and POST them back to the same
    iframe URL with ``__EVENTTARGET`` set to the page-N pagination
    control. The Playwright session keeps cf_clearance warm across all
    of these requests.
    """
    if fetcher is None:
        with playwright_session() as f:
            return fetch_all_stubs(fetcher=f)

    shell_html = fetch_search_shell_html(fetcher=fetcher)
    uuid = _extract_iframe_uuid(shell_html)
    if uuid is None:
        print("  WARN: no iframe uuid found in NANP search shell, skipping")
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
    record_count = parse_record_count(page_html)
    if record_count is not None:
        print(f"  DocCount: {record_count} record(s) across {total_pages} pages")

    iframe_url = _iframe_results_url(uuid)
    current_html = page_html
    for page_num in range(2, total_pages + 1):
        target = _find_doPostBack_target_for_page(current_html, page_num)
        if target is None:
            target = _find_next_arrow_target(current_html)
        if target is None:
            print(
                f"  WARN: page {page_num} pagination target not found "
                f"(no number button and no next-arrow); stopping at "
                f"page {page_num - 1}"
            )
            break
        form = _build_postback_form(current_html, target[0], target[1])
        if form is None:
            print(
                f"  WARN: page {page_num} missing __VIEWSTATE / "
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
            print(f"  WARN: page {page_num} POST replay failed: {e}")
            break
        rows = parse_search_results_html(current_html)
        added = 0
        for r in rows:
            if r.source_url and r.source_url not in seen_urls:
                seen_urls.add(r.source_url)
                out.append(r)
                added += 1
        if added == 0:
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
    has access to the BCHN custom field). Other fields fall back to the
    stub when the profile field is None — this lets the list-page
    city/state cover cases where the profile employer block is blank.
    """
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
    print("Fetching NANP directory (Playwright-backed, single unfiltered walk)...")
    stubs_by_url: dict[str, NormalizedPractitionerRow] = {}
    with playwright_session() as fetcher:
        # Warm-up: the live NANP search shell needs the search-form
        # session cookie set by visiting /search/ first; without this
        # /search/newsearch.asp returns an empty body on the first hit
        # of a fresh Playwright session.
        print(f"  warm-up: {DIRECTORY_FORM_URL}")
        try:
            fetcher.get(DIRECTORY_FORM_URL)
        except Exception as e:  # pragma: no cover - live IO
            print(f"  WARN: warm-up failed: {e}")

        try:
            rows = fetch_all_stubs(fetcher=fetcher)
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
