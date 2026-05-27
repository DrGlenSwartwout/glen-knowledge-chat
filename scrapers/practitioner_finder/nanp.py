"""National Association of Nutrition Professionals (NANP) directory scraper.

NANP runs its public-facing site on WordPress at nanp.org but the member
directory is hosted on a separate YourMembership Classic AMS subdomain at
``mynanp.nanp.org``. The "Find a Practitioner" button on
``https://nanp.org/find-a-practitioner/`` simply links to:

    https://mynanp.nanp.org/search/custom.asp?id=7551

That URL is the public search FORM. The form POSTs (or GETs) to
``/search/newsearch.asp`` and the response is an HTML table of matching
members. Each result links to a member profile at
``/profile/?ID=<numeric>`` (numeric ID = YM-assigned, stable across re-runs).

Discovery 2026-05-27 (vault-internal Wave C build):

    GET https://mynanp.nanp.org/search/custom.asp?id=7551
        -> 24kB HTML form. Filters: txt_name / txt_city / txt_country /
           txt_state / txt_postalcode plus two custom-field flags:
             cdlCustomFieldValueIDBCHN-SINGLE  in {'', 'Yes', 'No'}
             cdlCustomFieldValueIDCDSP-SINGLE  in {'', 'Yes', 'No'}
           Hidden field cdlMemberTypeID=1705148 scopes to the practitioner
           member type (also pinned: cdlCustomFieldValueIDPublicDirectoriesSelection=
           'Include profile in Find a Practitioner Directory').

    POST https://mynanp.nanp.org/search/newsearch.asp
        with the form fields above; returns the paginated results page.
        Default 20 results / page; pagination links are
        ``/search/newsearch.asp?...&page=N``.

    GET https://mynanp.nanp.org/profile/?ID=<numeric>
        -> the per-member profile page (practice, address, phone,
           email, website, BCHN/CDSP custom-field block).

Two-stage scrape pattern (mirrors OEPF):

    1. Walk every results page (no filters set -> all directory-listable
       practitioners). Per page, capture (profile_id, name, city, state,
       country) for each <tr> in the result table. Pagination from the
       block of ``<a class="page">N</a>`` anchors at page bottom.
    2. Fetch each ``/profile/?ID=<id>`` page individually and extract the
       full row (practice / address / phone / email / website / BCHN flag).

Row contract:
    tier             = 'org_member'
    source_org       = 'NANP'
    specialties      = ['nutrition', 'holistic_health']      # locked
    source_url       = 'https://mynanp.nanp.org/profile/?ID=<numeric>'
                       (stable across re-runs; ID is the YM-assigned
                       internal profile id)
    fellowship_level = True when the member carries the BCHN&reg; (Board
                       Certified Holistic Nutritionist) credential. NANP
                       publishes a tiered membership ladder:
                           Student Member
                           Associate Member       (in school, no degree yet)
                           Professional Member    (degree from a NANP-approved school)
                           BCHN&reg; Credentialed (exam-vetted board cert)
                           CDSP&trade; Credentialed (exam-vetted, narrower scope
                                                     than BCHN — dietary supplement
                                                     specialty only)
                       BCHN is the elite, exam-vetted tier of the
                       Profession Member ladder; the spec calls it out
                       explicitly as the True trigger ("anything below
                       BCHN as False"). CDSP alone does NOT qualify —
                       it's a separate certificate (dietary-supplement
                       specialty) that runs alongside the membership
                       ladder, not above it; per spec, only BCHN flips
                       fellowship_level to True. A member with BOTH
                       BCHN and CDSP qualifies (BCHN wins).

Field-level BCHN detection precedence:
    1. Profile page custom-field row labelled "BCHN" (with optional
       &reg;) with value 'Yes' — the canonical signal from the
       cdlCustomFieldValueIDBCHN-SINGLE field.
    2. Fallback: 'BCHN' (with optional &reg;) appears as a credential
       token in the H1 display name (e.g. "Jane Doe, BCHN®" or
       "Jane Doe, MS, BCHN, CDSP"). This is a defensive fallback for
       profile pages where the custom-field block is missing/empty but
       the credential is still in the display name.

NANP membership data is small (~hundreds, not thousands) so the
two-stage fetch is bounded and cheap. Static UA + 0.5s sleep + 20s
timeout per request (rate-friendly, single-threaded; matches the rest
of the practitioner-finder cohort).

NOTE on Cloudflare: As of 2026-05-27 the ``mynanp.nanp.org`` subdomain
returns 403 to plain requests (Cloudflare managed-challenge mode). The
parser here is built off the YM Classic AMS HTML templates that NANP has
deployed since at least 2020 (Wayback-confirmed structure for the search
form). The two-stage shape is canonical for the platform — same shape
the OVDR adapter handles for the newer YM MVC variant. If the
Cloudflare gate persists at live-scrape time the migrate runner needs a
Playwright shim (mirrors the eyehealingcenter pattern); the pure parser
defined here is unaffected.
"""
import html as html_module
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://mynanp.nanp.org"
SEARCH_FORM_URL = f"{BASE}/search/custom.asp?id=7551"
SEARCH_RESULTS_URL = f"{BASE}/search/newsearch.asp"
PROFILE_URL = f"{BASE}/profile/"

LOCKED_SPECIALTIES = ["nutrition", "holistic_health"]

# Member-type hidden field: 1705148 scopes the search to the practitioner
# member type (i.e. NOT student / sustaining-org / school memberships).
# Captured from the live search form 2026-05-27.
SEARCH_MEMBER_TYPE_ID = "1705148"

# The other always-on pin from the search form — limits to members who
# opted into the public Find a Practitioner Directory listing. Without
# this every active member would surface, including those who opted out.
SEARCH_PUBLIC_DIRECTORY_PIN = "Include profile in Find a Practitioner Directory"

# Country-name -> ISO2 (same convention as iabdm.py / iaomt.py). NANP
# is global-ish — most US-heavy, with practitioners in Canada, UK, EU,
# Australia, NZ, Brazil, etc.
_COUNTRY_NAME_TO_ISO2 = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "canada": "CA",
    "ca": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "australia": "AU",
    "new zealand": "NZ",
    "ireland": "IE",
    "germany": "DE",
    "france": "FR",
    "spain": "ES",
    "italy": "IT",
    "netherlands": "NL",
    "belgium": "BE",
    "denmark": "DK",
    "sweden": "SE",
    "norway": "NO",
    "finland": "FI",
    "switzerland": "CH",
    "austria": "AT",
    "portugal": "PT",
    "iceland": "IS",
    "luxembourg": "LU",
    "poland": "PL",
    "czech republic": "CZ",
    "czechia": "CZ",
    "greece": "GR",
    "hungary": "HU",
    "romania": "RO",
    "bulgaria": "BG",
    "mexico": "MX",
    "brazil": "BR",
    "chile": "CL",
    "argentina": "AR",
    "colombia": "CO",
    "costa rica": "CR",
    "panama": "PA",
    "peru": "PE",
    "ecuador": "EC",
    "japan": "JP",
    "south korea": "KR",
    "korea": "KR",
    "singapore": "SG",
    "malaysia": "MY",
    "thailand": "TH",
    "philippines": "PH",
    "india": "IN",
    "pakistan": "PK",
    "china": "CN",
    "hong kong": "HK",
    "taiwan": "TW",
    "indonesia": "ID",
    "vietnam": "VN",
    "united arab emirates": "AE",
    "uae": "AE",
    "saudi arabia": "SA",
    "qatar": "QA",
    "kuwait": "KW",
    "bahrain": "BH",
    "israel": "IL",
    "turkey": "TR",
    "egypt": "EG",
    "south africa": "ZA",
    "morocco": "MA",
    "kenya": "KE",
    "russia": "RU",
}

# Canadian postal codes: A1A 1A1 (with optional space).
_CA_POSTAL_RE = re.compile(r"\b([A-Z]\d[A-Z])\s*(\d[A-Z]\d)\b")
# US ZIP at end of a 'City, ST 12345' tail.
_US_POSTAL_RE = re.compile(r"\b(\d{5}(?:-\d{4})?)\s*$")
# UK postcode at end: SW1A 1AA.
_UK_POSTAL_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\s*$")

# BCHN credential token. Matches BCHN, BCHN®, BCHN&reg;, B.C.H.N. — case
# insensitive. The registered-trademark glyph (U+00AE) and the HTML
# entity &reg; are both stripped before testing so we just look for the
# letter run.
_BCHN_TOKEN_RE = re.compile(r"\bB\.?C\.?H\.?N\.?\b", re.IGNORECASE)

# US state codes (used to detect 'City, ST 12345' tails).
_US_STATE_ABBR_SET = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        }
    )
    return s


# ---------------------------------------------------------------------------
# Stage 1: results-page fetch + parse (per-page list of practitioner stubs)
# ---------------------------------------------------------------------------

def fetch_search_results_page(page: int = 1) -> str:
    """Hit one page of /search/newsearch.asp and return the raw HTML body.

    Default form values mirror the live search form: scope to the
    practitioner member type and to the 'public directory' opt-in subset.
    Static UA + 20s timeout + 0.5s sleep (rate-friendly).
    """
    params = {
        "cdlMemberTypeID": SEARCH_MEMBER_TYPE_ID,
        "cdlCustomFieldValueIDPublicDirectoriesSelection": SEARCH_PUBLIC_DIRECTORY_PIN,
        "page": str(page),
    }
    s = _session()
    r = s.get(SEARCH_RESULTS_URL, params=params, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


def parse_total_pages(html: str) -> int:
    """Extract total page count from the result page's pagination block.

    YM Classic renders ``<a class="page">N</a>`` for every page anchor;
    the current page is marked ``<a class="page current">``. Returns the
    max page number seen, or 1 if no pagination block (single-page
    result set).
    """
    soup = BeautifulSoup(html, "html.parser")
    max_page = 1
    for a in soup.find_all("a", class_="page"):
        try:
            n = int(a.get_text(strip=True))
            if n > max_page:
                max_page = n
        except (TypeError, ValueError):
            continue
    return max_page


_PROFILE_ID_RE = re.compile(r"[?&]ID=(\d+)", re.IGNORECASE)


def _extract_profile_id(href: str) -> Optional[str]:
    """Pull the ``ID=`` numeric out of a ``/profile/?ID=<n>`` href.

    Returns the numeric string (preserves leading zeros if any) or None
    when the href doesn't carry an ID.
    """
    if not href:
        return None
    m = _PROFILE_ID_RE.search(href)
    return m.group(1) if m else None


def _build_source_url(profile_id: str) -> str:
    """Stable per-practitioner URL — bare ``/profile/?ID=<numeric>``.

    Search results may carry additional context params (search refinement
    breadcrumbs); we strip those so re-runs from different filter slices
    produce identical upsert keys. This is the same canonicalization the
    OVDR adapter applies to its profileId URLs.
    """
    return f"{PROFILE_URL}?ID={profile_id}"


def parse_search_results_html(html: str) -> list[dict]:
    """Parse a /search/newsearch.asp results HTML page into a list of
    stubs (one per result row). Each stub is::

        {
            "profile_id": "12345678",
            "source_url": "https://mynanp.nanp.org/profile/?ID=12345678",
            "name": "Sarah Henderson, BCHN®",
            "city": "Boulder",
            "state": "CO",
            "country_raw": "United States",
        }

    Pure: no I/O. Stubs are the input to stage 2 (per-profile fetch +
    parse_member_profile_html). Empty rows / non-result table rows are
    silently skipped. Duplicates (same profile_id appearing twice on the
    same page) are dedup'd to the first occurrence.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    seen_ids: set[str] = set()

    # The results live in <table class="search-results">. Defensively
    # also accept the older 'results' class name used elsewhere on the
    # platform, and the SpContent_Container wrapper that always exists.
    table = (
        soup.find("table", class_="search-results")
        or soup.find("table", class_="results")
    )
    if not table:
        return out

    for tr in table.find_all("tr"):
        # Skip table header rows
        if tr.find("th") and not tr.find("td"):
            continue
        a = tr.find("a", href=True)
        if not a:
            continue
        href = a.get("href") or ""
        profile_id = _extract_profile_id(href)
        if not profile_id:
            continue
        if profile_id in seen_ids:
            continue
        # Name is the anchor's text (may contain credential tokens like
        # 'Sarah Henderson, BCHN®'). Empty / whitespace-only names mean
        # an opt-out row or a profile-photo-only anchor — skip.
        name_raw = html_module.unescape(a.get_text(strip=True))
        if not name_raw:
            continue
        seen_ids.add(profile_id)

        # Pull the remaining <td>s after the name cell. Their order on
        # the YM Classic results template is City / State / Country.
        tds = tr.find_all("td")
        city = state = country_raw = None
        # td[0] is the name cell (contains the <a>). td[1..3] are the
        # geo cells when present.
        geo_cells = tds[1:] if len(tds) > 1 else []
        if len(geo_cells) >= 1:
            city = geo_cells[0].get_text(strip=True) or None
        if len(geo_cells) >= 2:
            state = geo_cells[1].get_text(strip=True) or None
        if len(geo_cells) >= 3:
            country_raw = geo_cells[2].get_text(strip=True) or None

        out.append(
            {
                "profile_id": profile_id,
                "source_url": _build_source_url(profile_id),
                "name": name_raw,
                "city": city,
                "state": state,
                "country_raw": country_raw,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Stage 2: per-profile fetch + parse
# ---------------------------------------------------------------------------

def fetch_member_profile_html(profile_id: str) -> str:
    """Hit a single ``/profile/?ID=<n>`` page and return the raw HTML.
    Static UA + 20s timeout + 0.5s sleep (rate-friendly).
    """
    s = _session()
    r = s.get(PROFILE_URL, params={"ID": profile_id}, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Stripped string or None for missing/empty values."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s or None
    s = str(val).strip()
    return s or None


def _normalize_credential_chunk(s: str) -> str:
    """Strip the &reg; / &trade; entity-or-glyph from a credential token
    so 'BCHN®' / 'BCHN&reg;' / 'BCHN' all compare equal."""
    if not s:
        return ""
    out = html_module.unescape(s)
    # Strip registered/trademark glyphs and entities.
    out = out.replace("®", "").replace("™", "")
    return out.strip()


def _strip_credentials(name_and_creds: str) -> tuple[str, Optional[str]]:
    """Split 'Sarah Henderson, MS, BCHN®, CDSP™' ->
    ('Sarah Henderson', 'MS, BCHN, CDSP'). Trademark glyphs are stripped
    so credentials compare cleanly downstream.

    The name retains 'Dr.' / 'Dr' honorifics if present.
    """
    if not name_and_creds:
        return "", None
    s = html_module.unescape(name_and_creds).strip()
    # Strip ® and ™ from the whole string for cleaner credential tokens.
    s = s.replace("®", "").replace("™", "")

    cred_pat = re.compile(r",\s*([A-Z][A-Za-z./\-]{0,15})")
    m = cred_pat.search(s)
    if not m:
        return s.rstrip(", ").strip(), None

    clean = s[: m.start()].strip().rstrip(",")
    creds = s[m.start():].lstrip(", ").strip().rstrip(",").rstrip()
    return clean, creds or None


def _has_bchn(s: Optional[str]) -> bool:
    """True when 'BCHN' (in any spacing/case, with or without dots) appears
    in a string. Trademark glyphs are stripped first so 'BCHN®' matches."""
    if not s:
        return False
    cleaned = _normalize_credential_chunk(s)
    return bool(_BCHN_TOKEN_RE.search(cleaned))


def _country_iso2(raw: Optional[str]) -> Optional[str]:
    """Map a free-text country name to ISO2; None if unrecognized."""
    s = _coerce_str(raw)
    if not s:
        return None
    return _COUNTRY_NAME_TO_ISO2.get(s.lower())


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Add an https:// scheme to a bare domain."""
    s = _coerce_str(raw)
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    # Strip a leading 'www.' artifact from anchor text — adding the
    # scheme implicitly preserves www.
    return f"https://{s}"


def _parse_address_block(block_text: str) -> tuple[
    Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]
]:
    """Parse a multi-line address into (address1, city, state, postal, country).

    Accepts the canonical three-line YM profile address block::

        1234 Pearl Street, Suite 200
        Boulder, CO 80302
        United States

    Two-line variants (no explicit country line) are also handled — the
    country is then ``None`` and the caller derives it from the list-page
    country stub. Single-line addresses fall through with most fields
    None.
    """
    if not block_text:
        return None, None, None, None, None
    lines = [ln.strip() for ln in block_text.split("\n") if ln.strip()]
    if not lines:
        return None, None, None, None, None

    address1 = lines[0] or None
    country_raw = None
    middle_line = None
    if len(lines) == 1:
        return address1, None, None, None, None
    if len(lines) >= 3:
        # Standard 3-line block: street / city-state-postal / country
        middle_line = lines[1]
        country_raw = lines[-1]
    else:
        # 2-line block: just street / city-state-postal
        middle_line = lines[1]

    # Parse the city/state/postal middle line.
    city = state = postal = None
    if middle_line:
        s = re.sub(r"\s+", " ", middle_line)
        if "," in s:
            city_part, rest = s.rsplit(",", 1)
            city = city_part.strip() or None
            rest = rest.strip()

            # Country-specific postal patterns first so multi-word states
            # (Ontario, British Columbia, New York) survive.
            m_ca = _CA_POSTAL_RE.search(rest)
            if m_ca:
                raw_segment = rest[m_ca.start():m_ca.end()]
                if " " in raw_segment:
                    postal = f"{m_ca.group(1)} {m_ca.group(2)}"
                else:
                    postal = f"{m_ca.group(1)}{m_ca.group(2)}"
                state = rest[: m_ca.start()].strip() or None
            else:
                m_uk = _UK_POSTAL_RE.search(rest)
                if m_uk:
                    postal = m_uk.group(1).strip()
                    state = rest[: m_uk.start()].strip() or None
                else:
                    m_us = _US_POSTAL_RE.search(rest)
                    if m_us:
                        postal = m_us.group(1).strip()
                        state = rest[: m_us.start()].strip() or None
                    else:
                        state = rest or None
        else:
            # No comma — treat as just the city.
            city = s or None

    country = _country_iso2(country_raw) if country_raw else None
    return address1, city, state, postal, country or country_raw


def _profile_field_map(soup: BeautifulSoup) -> dict[str, str]:
    """Pull every <table>-based profile field block into a flat dict
    keyed by lowercased label (colon stripped, trademark glyphs removed).

    Both the main profile-info table and the custom-info table share
    the same <th>label</th><td>value</td> structure so we sweep all
    rows in any table inside the ``SpContent_Container`` wrapper.

    Anchor-aware fields:
      - 'website'  -> prefer the <a href="..."> value over anchor text
                       (anchor text often drops the scheme).
      - 'email'    -> strip the 'mailto:' prefix from anchor hrefs.
      - 'phone'    -> strip the 'tel:' prefix from anchor hrefs and
                       trust the href over text (which sometimes carries
                       formatting characters like '(512) 555-0199').
    """
    out: dict[str, str] = {}
    root = soup.find(id="SpContent_Container") or soup
    for tr in root.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        label_raw = th.get_text(strip=True)
        label = _normalize_credential_chunk(label_raw).rstrip(":").lower()
        if not label:
            continue
        # Anchor-bearing fields: use the href when present + scheme'd.
        a = td.find("a")
        href = (a.get("href") if a else None) or ""
        href = href.strip()
        if label == "website":
            if href and (href.startswith("http://") or href.startswith("https://")):
                out[label] = href
                continue
            # Anchor missing/relative — fall through to text-content.
        elif label == "email":
            if href.lower().startswith("mailto:"):
                out[label] = href[len("mailto:"):].strip()
                continue
        elif label == "phone":
            if href.lower().startswith("tel:"):
                out[label] = href[len("tel:"):].strip()
                continue
        # Default: multiline-preserving text content (lets the address
        # block parser see the <br>-separated lines).
        value = td.get_text(separator="\n", strip=True)
        out[label] = value
    return out


def _extract_canonical_url(soup: BeautifulSoup, fallback: Optional[str]) -> Optional[str]:
    """Prefer <meta property="og:url"> / <link rel="canonical"> for the
    canonical profile URL. Falls back to the caller-supplied URL."""
    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        return og["content"].strip()
    can = soup.find("link", rel="canonical")
    if can and can.get("href"):
        return can["href"].strip()
    return fallback


# ---------------------------------------------------------------------------
# Public profile parser
# ---------------------------------------------------------------------------

def parse_member_profile_html(
    html: str,
    *,
    profile_id: Optional[str] = None,
    stub: Optional[dict] = None,
) -> Optional[NormalizedPractitionerRow]:
    """Parse one /profile/?ID=<n> page into a NormalizedPractitionerRow.

    ``profile_id`` is the YM-assigned numeric — the row's stable dedup
    key. If omitted, it's pulled from the canonical/og:url meta on the
    page. ``stub`` is the list-page stub for this practitioner (used as
    a fallback for city/state/country when the profile page's address
    block omits the country line, and for the BCHN credential token in
    the display name when the profile body doesn't render a custom-field
    table).

    Returns None when the page lacks a usable name AND profile_id is
    missing (defensive — both shouldn't happen on a real profile).
    """
    soup = BeautifulSoup(html, "html.parser")
    canonical = _extract_canonical_url(soup, fallback=None)

    # Resolve profile_id from canonical URL when not passed in.
    if not profile_id and canonical:
        profile_id = _extract_profile_id(canonical)
    if not profile_id and stub:
        profile_id = stub.get("profile_id")

    # Display name comes from the H1.
    h1 = soup.find("h1")
    display = h1.get_text(strip=True) if h1 else ""
    name, credentials = _strip_credentials(display)

    if not name and stub:
        # Fallback to the list-page name (also gets credential-stripped).
        stub_name, stub_creds = _strip_credentials(stub.get("name") or "")
        name = stub_name
        credentials = credentials or stub_creds

    if not name or not profile_id:
        return None

    # Field map from the profile tables.
    fields = _profile_field_map(soup)

    practice = (
        fields.get("practice / organization")
        or fields.get("practice/organization")
        or fields.get("practice")
        or fields.get("organization")
    )
    practice = _coerce_str(practice)
    if practice and practice.lower() == name.lower():
        # Suppress duplicate when 'practice' field just echoes the name.
        practice = None

    phone = _coerce_str(fields.get("phone"))
    email = _coerce_str(fields.get("email"))
    website = _normalize_website(fields.get("website"))

    address_raw = fields.get("address") or ""
    address1, city, state, postal, country = _parse_address_block(address_raw)

    # Backfill geo from the list-page stub when the profile block was
    # sparse. Country is the most common gap — many 2-line address
    # blocks omit it.
    if stub:
        if not city:
            city = _coerce_str(stub.get("city"))
        if not state:
            state = _coerce_str(stub.get("state"))
        if not country:
            country = _country_iso2(stub.get("country_raw"))

    # BCHN detection — three signals, in precedence order:
    #   1. Custom-field row "BCHN" with value "Yes"
    #   2. Credential token "BCHN" present in the post-comma credentials
    #      block of the H1.
    #   3. Credential token "BCHN" present in the list-page name (stub),
    #      which carries the full display-name including credentials.
    bchn_field = _coerce_str(fields.get("bchn"))
    bchn_yes_from_field = bchn_field is not None and bchn_field.lower() == "yes"
    bchn_from_creds = _has_bchn(credentials)
    bchn_from_stub_name = bool(stub and _has_bchn(stub.get("name")))
    fellowship_level = bchn_yes_from_field or bchn_from_creds or bchn_from_stub_name

    # If the profile page's address block lacked an explicit country and
    # the stub didn't help either, default to US (the dominant geo for
    # the NANP directory).
    country_final = country or "US"

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="NANP",
        source_url=_build_source_url(profile_id),
        fellowship_level=fellowship_level,
        practice_name=practice,
        credentials=credentials,
        phone=phone,
        email=email,
        website=website,
        address1=address1,
        city=city,
        state=state,
        postal=postal,
        country=country_final,
    )


# ---------------------------------------------------------------------------
# Stage 1+2 orchestrator (used by migrate_nanp.py)
# ---------------------------------------------------------------------------

def fetch_all_records() -> list[NormalizedPractitionerRow]:
    """Walk every results page, then fetch + parse each member profile.

    Returns a flat list of NormalizedPractitionerRow, dedup'd by
    profile_id within the run. 0.5s sleep between every HTTP call
    (the inner fetch helpers already sleep, so the loop body just
    chains them).

    Profiles that 404 or return non-200 are logged + skipped — the
    rest of the run continues. This is the public entry point that
    ``migrate_nanp.main()`` and ``run_all._run_nanp_scrape()`` call.
    """
    seen: set[str] = set()
    stubs: list[dict] = []

    # Stage 1: walk results pages.
    page = 1
    while True:
        html = fetch_search_results_page(page=page)
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
    out: list[NormalizedPractitionerRow] = []
    for stub in stubs:
        pid = stub["profile_id"]
        try:
            profile_html = fetch_member_profile_html(pid)
        except requests.HTTPError as e:
            # Log + skip — sibling profiles are independent.
            print(f"  WARN: NANP profile {pid} fetch failed: {e}")
            continue
        row = parse_member_profile_html(
            profile_html, profile_id=pid, stub=stub
        )
        if row is not None:
            out.append(row)
    return out
