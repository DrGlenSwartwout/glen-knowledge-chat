"""American College of Functional Neurology (ACFN) directory scraper.

ACFN publishes its Fellows directory at https://acfn.org/directory/ as a
single static WordPress page. Discovery 2026-05-29: a plain ``requests``
GET (no login, no JS render, no Cloudflare) returns the entire roster
inline as one HTML ``<table class="posts-data-table">`` — 151 fellows as
of capture, all on one page (no pagination; ``page-numbers`` markers are
absent).

Table shape (header row is ``<th>``; data rows are ``<td>``):

    Name | Fellowships | Clinic Name | Clinic City | Clinic State |
    Countries | Certification Years | hf:categories | hf:tax:country |
    hf:tax:certification_year

The ``Name`` cell wraps an ``<a href="https://acfn.org/fellows/<slug>/">``
linking to the fellow's profile page — that per-fellow URL is the stable,
unique ``source_url`` (and the dedup / ON CONFLICT key) for each row.

The ``Fellowships`` cell holds one or more credential tokens, each in its
own ``<span data-slug="...">`` separated by commas, e.g.::

    <td><span data-slug="fabvr">FABVR</span>,
        <span data-slug="facfn">FACFN</span></td>

These tokens ARE the practitioner's ACFN credentials and the sole source
of the fellowship marker. Observed tokens: FACFN (Fellow), the
sub-fellowships FABBIR / FABCDD / FABES / FABNN / FABVR (and FABHP per
spec, not yet seen live), plus the non-fellow markers CABCDD (a
*Candidate*, not a Fellow), 'Uncategorized', and 'Retired'.

Fellowship rule: ``fellowship_level=True`` when the credentials carry
FACFN OR any sub-fellowship token (FABBIR / FABCDD / FABES / FABNN /
FABVR / FABHP), matched on word boundaries so CABCDD does NOT trip
FABCDD. Rows whose only tokens are CABCDD / Uncategorized / Retired are
plain members -> False.

Output rows have tier='org_member', source_org='ACFN',
specialties=['functional_neurology', 'holistic_health']. The listing
table carries name, credentials, practice (Clinic Name), city, state,
and country; phone / email / website are NOT in the listing table (they
live on the individual profile pages). The default scrape is a single
GET of the listing — phone/website enrichment via the profile pages is
opt-in (``fetch_with_profiles`` / ``parse_profile_html``) to keep the
default run rate-friendly. lat / lng / photo_url / bio are always None
(portal- and geocoder-managed).
"""
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://acfn.org"
DIRECTORY_URL = f"{BASE}/directory/"

LOCKED_SPECIALTIES = ["functional_neurology", "holistic_health"]

# The seven Fellowship credential tokens. FABHP is in the spec but not
# yet present in the live capture; included so it flags correctly if it
# ever appears. CABCDD (Candidate) is deliberately NOT here.
_FELLOWSHIP_TOKENS = ("FACFN", "FABBIR", "FABCDD", "FABES", "FABNN", "FABVR", "FABHP")

# Word-boundary alternation so 'CABCDD' does not match 'FABCDD', etc.
_FELLOWSHIP_RE = re.compile(
    r"\b(?:" + "|".join(_FELLOWSHIP_TOKENS) + r")\b"
)

# Country name (as the ACFN directory writes it) -> ISO2. The directory's
# Countries column is free text; this keeps `country` consistent with the
# sibling adapters' 2-letter convention. Unknown / empty -> default 'US'.
_COUNTRY_NAME_TO_ISO2 = {
    "united states": "US",
    "usa": "US",
    "canada": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "ireland": "IE",
    "australia": "AU",
    "new zealand": "NZ",
    "germany": "DE",
    "netherlands": "NL",
    "the netherlands": "NL",
    "belgium": "BE",
    "switzerland": "CH",
    "austria": "AT",
    "spain": "ES",
    "portugal": "PT",
    "italy": "IT",
    "france": "FR",
    "norway": "NO",
    "sweden": "SE",
    "denmark": "DK",
    "finland": "FI",
    "malta": "MT",
    "poland": "PL",
    "mexico": "MX",
    "brazil": "BR",
    "china": "CN",
    "japan": "JP",
    "south korea": "KR",
    "korea": "KR",
    "singapore": "SG",
    "india": "IN",
    "south africa": "ZA",
    "israel": "IL",
    "united arab emirates": "AE",
    "uae": "AE",
}

# US full-state-name -> 2-letter abbr. The directory's Clinic State column
# is a free-text full name (occasionally misspelled, e.g. "New Jeresey");
# we canonicalize to the abbr when recognized, else keep the raw text.
_US_STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "district of columbia": "DC", "florida": "FL",
    "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY",
    "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}

# Listing-table header labels (in capture order). Used to locate columns
# by name rather than by fixed index, so the parser survives column
# re-ordering / added taxonomy columns.
_COL_NAME = "name"
_COL_FELLOWSHIPS = "fellowships"
_COL_CLINIC = "clinic name"
_COL_CITY = "clinic city"
_COL_STATE = "clinic state"
_COL_COUNTRY = "countries"


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
# Stage 1: fetch the directory page (single GET, no pagination)
# ---------------------------------------------------------------------------

def fetch_directory_page() -> str:
    """Download the full ACFN Fellows directory page HTML.

    Single GET — the entire roster ships in one ``<table>`` response.
    Static UA + 20s timeout + 0.5s courtesy sleep (rate-friendly, per
    feedback_e4l_portal_no_concurrency.md)."""
    s = _session()
    r = s.get(DIRECTORY_URL, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _clean(text: Optional[str]) -> Optional[str]:
    """Collapse internal whitespace; return None for empty."""
    if text is None:
        return None
    s = re.sub(r"\s+", " ", text).strip()
    return s or None


def _has_fellowship(credentials: Optional[str]) -> bool:
    """True when the credential string carries FACFN or any sub-fellowship
    token (matched on word boundaries so CABCDD does not match FABCDD)."""
    if not credentials:
        return False
    return bool(_FELLOWSHIP_RE.search(credentials.upper()))


def _country_iso2(name: Optional[str]) -> Optional[str]:
    """Map a free-text country name to ISO2; None if blank/unrecognized."""
    s = _clean(name)
    if not s:
        return None
    return _COUNTRY_NAME_TO_ISO2.get(s.lower())


def _normalize_state(raw: Optional[str], country: Optional[str]) -> Optional[str]:
    """Canonicalize a US full-state-name to its 2-letter abbr. For non-US
    (or unrecognized) values, return the cleaned raw text unchanged so the
    geocoder still has a region hint."""
    s = _clean(raw)
    if not s:
        return None
    if country == "US" or country is None:
        abbr = _US_STATE_NAME_TO_ABBR.get(s.lower())
        if abbr:
            return abbr
    return s


def _fellowships_from_cell(cell) -> Optional[str]:
    """Extract the credential string from the Fellowships <td>.

    Prefers the per-token ``<span data-slug>`` elements (clean, no stray
    whitespace around commas); falls back to the cell text when the markup
    differs. Returns a comma-joined token string (e.g. 'FABVR, FACFN') or
    None when empty / a pure non-fellow placeholder is the only content."""
    if cell is None:
        return None
    spans = cell.find_all("span", attrs={"data-slug": True})
    if spans:
        tokens = [_clean(sp.get_text()) for sp in spans]
        tokens = [t for t in tokens if t]
        joined = ", ".join(tokens)
        return joined or None
    # Fallback: split the raw cell text on commas and re-join cleanly.
    txt = _clean(cell.get_text(" "))
    if not txt:
        return None
    tokens = [t.strip() for t in txt.split(",") if t.strip()]
    return ", ".join(tokens) or None


def _build_source_url(profile_href: Optional[str], name: Optional[str]) -> str:
    """Stable, unique per-fellow URL.

    Prefers the profile <a href> from the Name cell (canonical, unique).
    Falls back to a slugged directory anchor when a row somehow lacks a
    link, so the dedup layer still gets a deterministic key instead of a
    collision."""
    href = _clean(profile_href)
    if href:
        if href.startswith("//"):
            return f"https:{href}"
        if href.startswith("/"):
            return f"{BASE}{href}"
        if href.startswith("http://") or href.startswith("https://"):
            return href
        return f"{BASE}/{href.lstrip('/')}"
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "unknown").lower()).strip("-") or "unknown"
    return f"{DIRECTORY_URL}#fellow-{slug}"


def _header_index(table) -> dict:
    """Map normalized header label -> column index from the first row."""
    rows = table.find_all("tr")
    if not rows:
        return {}
    header_cells = rows[0].find_all(["th", "td"])
    index = {}
    for i, c in enumerate(header_cells):
        label = (_clean(c.get_text(" ")) or "").lower()
        if label and label not in index:
            index[label] = i
    return index


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def parse_directory_html(html: str) -> list[NormalizedPractitionerRow]:
    """Pure parser: directory page HTML -> list[NormalizedPractitionerRow].

    Locates the ``<table class="posts-data-table">`` roster, maps columns
    by header label, and emits one row per fellow. Rows with no usable
    name are skipped. Phone / email / website are left None (not in the
    listing table); use ``parse_profile_html`` to enrich them per fellow.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="posts-data-table")
    if table is None:
        # Defensive: directory plugin swapped. Fall back to the first table
        # whose header contains both 'name' and 'fellowships'.
        for t in soup.find_all("table"):
            idx = _header_index(t)
            if _COL_NAME in idx and _COL_FELLOWSHIPS in idx:
                table = t
                break
    if table is None:
        return []

    idx = _header_index(table)
    if _COL_NAME not in idx:
        return []

    name_i = idx[_COL_NAME]
    fel_i = idx.get(_COL_FELLOWSHIPS)
    clinic_i = idx.get(_COL_CLINIC)
    city_i = idx.get(_COL_CITY)
    state_i = idx.get(_COL_STATE)
    country_i = idx.get(_COL_COUNTRY)

    rows: list[NormalizedPractitionerRow] = []
    all_tr = table.find_all("tr")
    for tr in all_tr[1:]:  # skip header
        cells = tr.find_all("td")
        if not cells or len(cells) <= name_i:
            continue

        name_cell = cells[name_i]
        name = _clean(name_cell.get_text(" "))
        if not name:
            continue

        link = name_cell.find("a")
        href = link.get("href") if link else None

        credentials = (
            _fellowships_from_cell(cells[fel_i])
            if fel_i is not None and len(cells) > fel_i
            else None
        )

        practice = (
            _clean(cells[clinic_i].get_text(" "))
            if clinic_i is not None and len(cells) > clinic_i
            else None
        )
        if practice and practice.lower() == name.lower():
            practice = None

        country_raw = (
            _clean(cells[country_i].get_text(" "))
            if country_i is not None and len(cells) > country_i
            else None
        )
        country_iso = _country_iso2(country_raw) or "US"

        city = (
            _clean(cells[city_i].get_text(" "))
            if city_i is not None and len(cells) > city_i
            else None
        )
        state_raw = (
            cells[state_i].get_text(" ")
            if state_i is not None and len(cells) > state_i
            else None
        )
        state = _normalize_state(state_raw, country_iso)

        rows.append(
            NormalizedPractitionerRow(
                tier="org_member",
                name=name,
                specialties=list(LOCKED_SPECIALTIES),
                source_org="ACFN",
                source_url=_build_source_url(href, name),
                fellowship_level=_has_fellowship(credentials),
                practice_name=practice,
                credentials=credentials,
                phone=None,
                email=None,
                website=None,
                address1=None,
                city=city,
                state=state,
                postal=None,
                country=country_iso,
            )
        )
    return rows


def fetch_all_directory_rows() -> list[NormalizedPractitionerRow]:
    """Fetch + parse the directory in one shot (single GET). The default,
    rate-friendly entry point for run_all."""
    html = fetch_directory_page()
    return parse_directory_html(html)


# ---------------------------------------------------------------------------
# Optional per-profile enrichment (opt-in — extra HTTP per fellow)
# ---------------------------------------------------------------------------

# Profile "Clinic Information:" block lines (label: value), e.g.
#   Phone Number: 225-271-4083
#   Website: www.MyEliteChiro.com
#   City: Livingston / State/Province: Louisiana / Country: United States
_PROFILE_PHONE_RE = re.compile(r"Phone\s*Number\s*:?\s*([+(\d][\d\s().+\-/]{5,})", re.I)
_PROFILE_MAILTO_RE = re.compile(
    r"mailto:([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.I
)


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Enforce an http(s) scheme on a bare domain; None for empties."""
    s = _clean(raw)
    if not s or s.lower() in {"none", "n/a", "null"}:
        return None
    if not (s.startswith("http://") or s.startswith("https://")):
        s = f"https://{s}"
    return s


def parse_profile_html(html: str) -> dict:
    """Pure parser for a single fellow profile page. Returns a dict with
    any of 'phone', 'email', 'website' that could be extracted from the
    'Clinic Information:' block. Missing fields are omitted.

    The ACFN org's own contact line (secretary@acfn.org) is excluded so a
    fellow's email isn't polluted by the footer."""
    out: dict = {}
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")

    m = _PROFILE_PHONE_RE.search(text)
    if m:
        phone = re.split(r"[/\n]", m.group(1), maxsplit=1)[0].strip().rstrip(".,-")
        if phone:
            out["phone"] = phone

    # Website appears on its own line after a "Website:" label.
    wm = re.search(r"Website\s*:?\s*\n?\s*([^\s\n]+)", text, re.I)
    if wm:
        site = _normalize_website(wm.group(1))
        if site:
            out["website"] = site

    for em in _PROFILE_MAILTO_RE.findall(html):
        if em.lower() != "secretary@acfn.org":
            out["email"] = em.lower()
            break

    return out


def fetch_with_profiles(
    rows: Optional[list[NormalizedPractitionerRow]] = None,
    *,
    sleep: float = 0.5,
) -> list[NormalizedPractitionerRow]:
    """Enrich listing rows with phone/email/website by fetching each
    fellow's profile page. OPT-IN: this issues one extra GET per fellow
    (~150 requests). Mutates and returns the rows in place.

    Pass ``rows`` to enrich an existing parse; omit to fetch the directory
    first. Profile fetch failures are swallowed (the listing row stays as
    is) so one bad page never aborts the batch."""
    if rows is None:
        rows = fetch_all_directory_rows()
    s = _session()
    for row in rows:
        url = row.source_url
        if not url or "/fellows/" not in url:
            continue
        try:
            r = s.get(url, timeout=20)
            r.raise_for_status()
            fields = parse_profile_html(r.text)
        except requests.RequestException:
            fields = {}
        if fields.get("phone") and not row.phone:
            row.phone = fields["phone"]
        if fields.get("email") and not row.email:
            row.email = fields["email"]
        if fields.get("website") and not row.website:
            row.website = fields["website"]
        time.sleep(sleep)
    return rows
