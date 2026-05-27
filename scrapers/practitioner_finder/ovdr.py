"""Optometric Vision Development & Rehabilitation Association (OVDRA / formerly
COVD) directory scraper.

OVDRA is the rebrand of the College of Optometrists in Vision Development
(COVD). The legacy ``covd.org`` apex returns 403 behind Cloudflare WAF, and
``ovdr.org`` 301-redirects everywhere. The member locator lives on its own
subdomain that's WAF-free:

    https://locate.covd.org/

That subdomain is an ASP.NET MVC application backed by YourMembership.com
(YM) — visible in the page footer "Association Management Software Powered
by YourMembership". The locator form (``id="SearchForm"``) submits to
``/Search/DoSearch`` (GET) with these parameters:

    Country=US&State=NY                         # by state
    Country=CA                                  # by country (non-US)
    Country=US&State=NY&page=N                  # pagination (20/page)
    Country=US&ZipCodeOrAddress=...&SearchRadius=...   # radius search

The country dropdown enumerates 30 countries, with US states exposed only
when Country=US is selected. The cleanest full-coverage strategy is:

  1. For Country=US, walk all 50 states + DC + military codes (the locator
     dropdown lists them all).
  2. For every non-US country in the dropdown, walk Country=<ISO2> (state
     is unavailable for these and search returns the country's full list).

Each search results page (``<table class="results">``) contains 1 to 20
doctors. Each doctor occupies 1+ ``<tr>``: a "marker" row with the pin
+ lat/long + the ``<a class="doctor">`` link, plus zero or more trailing
"multiOffices" rows (one per additional office the doctor practices at).
For the simple US case, office data (practice name, address, phone) is
collapsed into the same TR as the doctor link. For Canadian and
international entries, the doctor TR contains only a hidden
``<address style="display: none;">`` and the visible office data lives
in a separate follow-up TR with ``<div class="multiOffices">``.

The list page carries every field we need (name + credentials in the
anchor text, practice in ``<p class="office">``, address in
``<address>``, phone in ``<div class="phone">``, lat/lng + canonical
profileId in the detail anchor's URL). No per-record detail fetch is
required for the bulk pass — this mirrors IABDM's single-stage shape
rather than OEPF's two-stage.

Each doctor maps to ONE NormalizedPractitionerRow using their FIRST
listed office (the primary practice the doctor registered). Doctors who
practice at multiple offices share one source_url (per-profile, not
per-office), which is the right granularity: the practitioner is the
entity, not the address.

Output row contract:
  tier         = 'org_member'
  source_org   = 'OVDR'
  specialties  = ['rehabilitation', 'eye_care']      # locked
  source_url   = https://locate.covd.org/Search/Detailed?profileId=<GUID>
                 (stable across re-runs; profileId is the YM-assigned UUID)
  fellowship_level = True when the credential string contains FCOVD,
                 FCOVD-A, or FOVDR (case- and spacing-insensitive). OVDRA
                 calls its fellowship designation "Fellow of COVD (FCOVD)"
                 historically and "Fellowship (FOVDR)" post-rebrand; both
                 abbreviations appear in the live directory (the legacy
                 FCOVD form for already-credentialed doctors, the FOVDR
                 form for newly-credentialed). FCOVD-A is the "Advanced"
                 tier above standard FCOVD and also qualifies.
"""
import html as html_module
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://locate.covd.org"
SEARCH_URL = f"{BASE}/Search/DoSearch"
DETAIL_URL = f"{BASE}/Search/Detailed"

LOCKED_SPECIALTIES = ["rehabilitation", "eye_care"]

# Non-US countries OVDRA's locator dropdown exposes. Two-letter ISO codes
# matching the dropdown's value attribute. Order is arbitrary but kept
# stable for reproducible run logs.
NON_US_COUNTRIES = [
    "AU", "CA", "CN", "DK", "EC", "DE", "GR", "HK", "HU", "IN",
    "IL", "IT", "JP", "KW", "MY", "MX", "NL", "NZ", "NG", "NO",
    "PH", "PL", "SG", "ZA", "KR", "ES", "CH", "AE", "GB",
]

# All US states + DC + military codes (the locator dropdown lists every
# USPS abbreviation including AA/AE/AP for armed forces addresses, but
# only the 50 states + DC will realistically return practitioners).
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

# Fellowship marker. Matches FCOVD, FCOVD-A, F.C.O.V.D., FOVDR, F.O.V.D.R.
# in any case/spacing. The "Advanced" FCOVD-A is the senior-most fellowship
# tier; the post-rebrand FOVDR is the same designation under the new name.
_FELLOWSHIP_RE = re.compile(
    r"\b(?:F\.?C\.?O\.?V\.?D\.?(?:-?A)?|F\.?O\.?V\.?D\.?R\.?)\b",
    re.IGNORECASE,
)

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
# Stage 1: paginated search HTML fetch
# ---------------------------------------------------------------------------

def fetch_search_page(
    *,
    country: str,
    state: Optional[str] = None,
    page: int = 1,
) -> str:
    """Hit one page of /Search/DoSearch for the given (country, [state], page).

    Static UA + 20s timeout + 0.5s sleep (rate-friendly, single-threaded).
    Returns the raw HTML body. Caller decides when to stop (parse pagination
    or rely on ``parse_search_html`` returning fewer than 20 results).
    """
    params = {"Country": country, "page": str(page)}
    if state:
        params["State"] = state
    s = _session()
    r = s.get(SEARCH_URL, params=params, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


def parse_total_pages(html: str) -> int:
    """Extract the total page count from the pagination block.

    The locator renders ``<a class="page">N</a>`` for each page and an
    ``<a class="page current">`` for the current page. Returns the max
    page number seen (default 1 if no pagination block — single page of
    results)."""
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


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s or None
    s = str(val).strip()
    return s or None


def _strip_credentials(name_and_creds: str) -> tuple[str, Optional[str]]:
    """Split 'Rebecca Marinoff, OD, FCOVD' -> ('Rebecca Marinoff', 'OD, FCOVD').

    Credentials are the trailing comma-separated short uppercase tokens
    (with optional dots, hyphens, or numbers). The name retains 'Dr.'
    honorifics if present. If no trailing credential pattern matches,
    the entire string is returned as the name with credentials=None.
    """
    if not name_and_creds:
        return "", None
    s = name_and_creds.strip()

    # First credential token after a comma: short uppercase abbrev allowing
    # internal dots, hyphens, semicolons, slashes (covers FCOVD-A, OD;,
    # FAAO, F.C.O.V.D., Doctor of Optometry, MS, MBA, PhD, COVT).
    # We also accept Title-case "Doctor of Optometry" / "Optometrist" style
    # tails by treating any first comma as the credential boundary when the
    # text after it isn't sentence-y (i.e. all-uppercase OR a known role).
    cred_pat = re.compile(r",\s*([A-Z][A-Za-z./\-;]*)")
    m = cred_pat.search(s)
    if not m:
        return s.rstrip(", "), None

    clean = s[: m.start()].strip().rstrip(",")
    creds = s[m.start():].lstrip(", ").strip().rstrip(",").rstrip()
    return clean, creds or None


def _extract_lat_lng_from_href(href: str) -> tuple[Optional[float], Optional[float]]:
    """Pull lat= / lng= query params out of a /Search/Detailed URL.

    Returned as floats (or None when absent / unparseable). The locator
    embeds Google-Maps-geocoded coordinates in every detail link, so we
    can extract lat/lng without a separate geocoder pass — but we follow
    the cross-adapter convention of leaving lat/lng to the shared geocoder
    so geocode_quality stays consistent. These helpers exist for tests
    and future use only.
    """
    if not href:
        return None, None
    m_lat = re.search(r"[?&]lat=([-\d.]+)", href)
    m_lng = re.search(r"[?&]lng=([-\d.]+)", href)
    try:
        lat = float(m_lat.group(1)) if m_lat else None
    except (TypeError, ValueError):
        lat = None
    try:
        lng = float(m_lng.group(1)) if m_lng else None
    except (TypeError, ValueError):
        lng = None
    return lat, lng


def _extract_profile_id(href: str) -> Optional[str]:
    """Pull the profileId GUID out of a /Search/Detailed URL. The profileId
    is the stable YM-assigned UUID for the practitioner and is our dedup
    key. Returns None if the URL is malformed."""
    if not href:
        return None
    m = re.search(r"[?&]profileId=([A-Za-z0-9\-]+)", href)
    if not m:
        return None
    return m.group(1)


def _build_source_url(profile_id: Optional[str]) -> Optional[str]:
    """Stable per-practitioner URL — bare /Search/Detailed?profileId=GUID.

    The page also requires ``&address=...&Country=...&State=...`` to render
    correctly, but the ``profileId`` alone is the dedup key. We keep
    source_url canonical (bare profileId only) so re-runs from different
    search contexts produce identical upsert keys."""
    if not profile_id:
        return None
    return f"{DETAIL_URL}?profileId={profile_id}"


# Canadian postal codes are ANA NAN, optionally separated by a space:
# 'N0H2C1', 'N0H 2C1', 'M9A 4S4'. Use to detect end-of-line postal even
# when the state is a multi-word name like 'British Columbia'.
_CA_POSTAL_RE = re.compile(r"\b([A-Z]\d[A-Z])\s*(\d[A-Z]\d)\b")

# US ZIP at the very end of the line: '10036' or '10036-8005'.
_US_POSTAL_RE = re.compile(r"\b(\d{5}(?:-\d{4})?)\s*$")

# UK postcode pattern at end of line: 'SW1A 1AA' or 'SW112PJ' (no space).
# Loose match — the leading letters + digits + optional space + trailing digits/letters.
_UK_POSTAL_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\s*$")


def _split_city_state_postal(
    line2: str, country: str
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Split a 'City, State Postal' / 'City, Province Postal' line.

    US patterns:
        'New York, NY 10036'           -> ('New York', 'NY', '10036')
        'New York, NY 10036-8005'      -> ('New York', 'NY', '10036-8005')
        'Commack, New York 11725'      -> ('Commack', 'New York', '11725')
    Canadian (any province name, any postal spacing):
        'Port Elgin, Ontario N0H2C1'   -> ('Port Elgin', 'Ontario', 'N0H2C1')
        'Etobicoke, Ontario M9A 4S4'   -> ('Etobicoke', 'Ontario', 'M9A 4S4')
        'Kelowna, BC V1Y 0H8'          -> ('Kelowna', 'BC', 'V1Y 0H8')
    UK / EU:
        'London, England SW112PJ'      -> ('London', 'England', 'SW112PJ')

    Returns (city, state_or_province, postal) — any may be None.
    """
    if not line2:
        return None, None, None
    s = re.sub(r"\s+", " ", line2.strip())
    if "," not in s:
        return s or None, None, None

    # Split on the LAST comma to separate city from state+postal.
    city, rest = s.rsplit(",", 1)
    city = city.strip() or None
    rest = rest.strip()

    # Try country-specific postal patterns FIRST so multi-word state names
    # (British Columbia, New York, etc.) survive postal extraction intact.
    # Canadian postal — anchored at end of rest.
    m_ca = _CA_POSTAL_RE.search(rest)
    if m_ca:
        # Postal = the two groups, with a normalizing space between them
        # to match the on-page form (Canada Post canonical is space-separated).
        postal = f"{m_ca.group(1)} {m_ca.group(2)}"
        # If the on-page form had no space, preserve that (it's still a valid
        # postal code).
        raw_postal_segment = rest[m_ca.start():m_ca.end()]
        if " " not in raw_postal_segment:
            postal = f"{m_ca.group(1)}{m_ca.group(2)}"
        state = rest[: m_ca.start()].strip() or None
        return city, state, postal

    # UK postcode — anchored at end.
    m_uk = _UK_POSTAL_RE.search(rest)
    if m_uk:
        postal = m_uk.group(1).strip()
        state = rest[: m_uk.start()].strip() or None
        return city, state, postal

    # US ZIP — anchored at end.
    m_us = _US_POSTAL_RE.search(rest)
    if m_us:
        postal = m_us.group(1).strip()
        state = rest[: m_us.start()].strip() or None
        return city, state, postal

    # No recognizable postal: rest is just the state/province name.
    return city, rest or None, None


def _parse_address_block(address_html: str, country: str) -> tuple[
    Optional[str], Optional[str], Optional[str], Optional[str]
]:
    """Parse an <address> block into (address1, city, state, postal).

    The block is two lines separated by <br>:
        <address>
            33 W 42nd St
            <br />
            New York, NY 10036
        </address>
    """
    if not address_html:
        return None, None, None, None
    # Replace <br> tags with newlines, then collapse whitespace per line.
    txt = re.sub(r"<br\s*/?>", "\n", address_html, flags=re.IGNORECASE)
    txt = re.sub(r"<[^>]+>", "", txt)  # strip remaining tags
    txt = html_module.unescape(txt)
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    if not lines:
        return None, None, None, None
    address1 = lines[0]
    if len(lines) >= 2:
        city, state, postal = _split_city_state_postal(lines[1], country)
    else:
        city, state, postal = None, None, None
    return address1, city, state, postal


def _is_fellowship(credentials_or_name: Optional[str]) -> bool:
    """True if FCOVD / FCOVD-A / FOVDR (in any case/spacing) appears."""
    if not credentials_or_name:
        return False
    return bool(_FELLOWSHIP_RE.search(credentials_or_name))


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def _row_to_dict(row_html: str) -> dict:
    """Slice a single <tr>...</tr> into named fields. Internal helper used
    by the main parser; broken out for testability."""
    soup = BeautifulSoup(row_html, "html.parser")
    out: dict = {}

    # Doctor anchor (only present in the "primary" doctor row)
    a = soup.find("a", class_="doctor")
    if a:
        href = a.get("href") or ""
        out["doctor_href"] = href
        out["doctor_label"] = a.get_text(strip=True)
        out["profile_id"] = _extract_profile_id(href)
        out["lat"], out["lng"] = _extract_lat_lng_from_href(href)

    # Practice name(s) — <p class="office">
    offices = [p.get_text(strip=True) for p in soup.find_all("p", class_="office")]
    # Filter out empty / "Office N:" sentinel lines that appear in
    # multiOffices blocks
    real = [o for o in offices if o and not re.match(r"^Office\s+\d+:?$", o, re.I)]
    if real:
        out["practice_name"] = real[0]

    # Address block — <address>
    addr = soup.find("address")
    if addr:
        out["address_html"] = str(addr)

    # Phone — <div class="phone">
    phone = soup.find("div", class_="phone")
    if phone:
        out["phone"] = phone.get_text(strip=True) or None

    return out


def parse_search_html(
    html: str, *, country: str = "US"
) -> list[NormalizedPractitionerRow]:
    """Parse a /Search/DoSearch results page into 0+ NormalizedPractitionerRow.

    Each doctor's data may span 1+ ``<tr>``s: the "marker" TR with the
    ``<a class="doctor">`` link, optionally followed by "multiOffices" TRs
    (one per additional office). When the doctor's first office data is
    collapsed into the marker TR (US single-office case), we emit one row
    from that TR alone. When the marker TR has a hidden address (Canadian /
    international case), we look at the next TR — the first
    ``multiOffices`` block — for practice/address/phone.

    Returns one NormalizedPractitionerRow per doctor; multi-office doctors
    yield ONE row (using their first office) because source_url is per
    practitioner, not per address. Pure: no I/O.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="results")
    if not table:
        return []

    rows_html: list[str] = []
    for tr in table.find_all("tr"):
        rows_html.append(str(tr))

    results: list[NormalizedPractitionerRow] = []
    i = 0
    while i < len(rows_html):
        row_dict = _row_to_dict(rows_html[i])
        if not row_dict.get("doctor_href"):
            # Header row or orphan multiOffices row (skip — these are
            # consumed in lookahead below).
            i += 1
            continue

        profile_id = row_dict.get("profile_id")
        source_url = _build_source_url(profile_id)
        # Without a profileId we have no stable dedup key — skip rather than
        # synthesize, because re-runs would otherwise create duplicates.
        if not source_url:
            i += 1
            continue

        # The label is "Name, Cred1, Cred2, ...". Split.
        label = row_dict.get("doctor_label") or ""
        name, credentials = _strip_credentials(label)
        if not name:
            i += 1
            continue

        # Practice / address / phone may live in THIS row (US single-office
        # case) OR in the NEXT row's multiOffices block (Canadian /
        # international case where the doctor TR holds only a HIDDEN
        # ``<address style="display: none;">`` and the visible data lives
        # in a follow-up multiOffices TR).
        addr_html = row_dict.get("address_html")
        practice = row_dict.get("practice_name")
        phone = row_dict.get("phone")

        # Peek ahead: collect every consecutive multiOffices TR. If any
        # exists, it carries the visible office data and we prefer it over
        # the doctor TR's hidden address. Multi-office doctors yield ONE
        # row using their FIRST office (per the per-practitioner dedup
        # contract).
        if i + 1 < len(rows_html):
            nxt = _row_to_dict(rows_html[i + 1])
            if nxt.get("address_html") and not nxt.get("doctor_href"):
                # The doctor TR's address (if any) was the hidden duplicate;
                # the multiOffices TR is authoritative for THIS doctor.
                addr_html = nxt.get("address_html") or addr_html
                practice = nxt.get("practice_name") or practice
                phone = nxt.get("phone") or phone
                i += 1
                # Skip over any additional multiOffices TRs (extra offices
                # for the SAME doctor — we only emit one row per doctor).
                while i + 1 < len(rows_html):
                    after = _row_to_dict(rows_html[i + 1])
                    if after.get("address_html") and not after.get("doctor_href"):
                        i += 1
                    else:
                        break

        address1, city, state, postal = _parse_address_block(addr_html or "", country)

        # If practice matches practitioner name verbatim (solo entry where
        # the office name field was filled with the doctor's name),
        # suppress the duplicate.
        if practice and name and practice.strip().lower() == name.strip().lower():
            practice = None

        row = NormalizedPractitionerRow(
            tier="org_member",
            name=name,
            specialties=list(LOCKED_SPECIALTIES),
            source_org="OVDR",
            source_url=source_url,
            fellowship_level=_is_fellowship(credentials) or _is_fellowship(label),
            practice_name=practice,
            credentials=credentials,
            phone=phone,
            email=None,        # not present on the list page
            website=None,      # not present on the list page
            address1=address1,
            city=city,
            state=state,
            postal=postal,
            country=country if country else "US",
        )
        results.append(row)
        i += 1

    return results


def fetch_all_records() -> list[NormalizedPractitionerRow]:
    """Walk every (country, state?) tuple OVDR's locator exposes and return
    a flat list of NormalizedPractitionerRow. Dedups by source_url within
    the run — the same practitioner can theoretically appear in multiple
    search slices (e.g. Country=US&State=NY plus Country=US with no state),
    though in practice the dropdown's mutual-exclusion enforces single
    appearance.

    Used by migrate_ovdr.main(). 0.5s sleep between every HTTP call (the
    fetch_search_page helper already sleeps internally)."""
    seen: set[str] = set()
    out: list[NormalizedPractitionerRow] = []

    # US states
    for st in US_STATES:
        page = 1
        while True:
            html = fetch_search_page(country="US", state=st, page=page)
            rows = parse_search_html(html, country="US")
            if not rows:
                break
            new_rows = [r for r in rows if r.source_url and r.source_url not in seen]
            for r in new_rows:
                seen.add(r.source_url)
                out.append(r)
            # Determine pagination — total pages from the page itself
            total = parse_total_pages(html)
            if page >= total:
                break
            page += 1

    # Non-US countries
    for c in NON_US_COUNTRIES:
        page = 1
        while True:
            html = fetch_search_page(country=c, page=page)
            rows = parse_search_html(html, country=c)
            if not rows:
                break
            new_rows = [r for r in rows if r.source_url and r.source_url not in seen]
            for r in new_rows:
                seen.add(r.source_url)
                out.append(r)
            total = parse_total_pages(html)
            if page >= total:
                break
            page += 1

    return out
