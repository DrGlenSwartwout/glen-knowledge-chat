"""eyehealingcenter.com scraper for the Phase 1 migration.

Two source pages:
- /practitioner-finder-by-state (groups practitioners under state headers,
  then city sub-headers)
- /practitioner-finder-by-city  (groups practitioners under city headers,
  alphabetical across all states)

HTML structure (discovered 2026-05-26):
  Both pages embed all practitioner data in a single
  div.gp-text-container — the widest-body text block on the page.
  All content is <p class="gp-component-id-..."> tags inside that div.

  By-state:
    - State headers: <p><strong>StateName</strong></p> (strong text = bare state name)
    - City sub-headers: <p> (no strong) whose text matches "City, State (XX)"
    - Practitioner data: one or more non-header <p> tags that follow a city line.
      All practitioner data for a city is packed into one or a few consecutive lines.

  By-city:
    - City headers: <p><strong>...</strong></p> where the strong text starts with
      "-" followed by "City, State (XX)", OR is a short "City ST" abbreviation
      (secondary section at end of page), OR is "• PractitionerName" (bullet-
      denoted practitioners embedded directly in a strong tag — these belong to
      the current city block).
    - Non-strong <p> tags after a city header are practitioner data lines.

Both produce NormalizedPractitionerRow with tier='eyehealing' and
specialties=['eye_care']. Sub-tags get layered in by the post-migration
classification sweep — see normalize.py docstring.
"""
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from scrapers.practitioner_finder.models import NormalizedPractitionerRow
from scrapers.practitioner_finder.normalize import infer_eyehealing_specialties


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://eyehealingcenter.com"
BY_STATE_URL = f"{BASE}/practitioner-finder-by-state"
BY_CITY_URL = f"{BASE}/practitioner-finder-by-city"

# Matches: "Birmingham, Alabama (AL)" or "Colorado Springs, (CO)"
# Also handles period separator typos: "Allentown. Pennsylvania (PA)"
_CITY_STATE_LONG_RE = re.compile(
    r"^([A-Z][^,.()\n]+?)[,.]\s+([A-Z][A-Za-z\s']+?)\s+\(([A-Z]{2})\)"
)
# Matches: "Tampa FL" or "St. Louis MO" (abbreviated, no parens)
_CITY_STATE_SHORT_RE = re.compile(
    r"^([\w\s./'-]+?)\s+([A-Z]{2})$"
)
# Phone patterns
_PHONE_RE = re.compile(
    r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"
)
# Website: token that looks like a hostname or URL path (no spaces, has a dot)
_WEBSITE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?"
)
# "Looking for" placeholder lines — skip these
_PLACEHOLDER_RE = re.compile(
    r"(?:we'?re? looking|i'?m looking|looking for|please let me know)",
    re.IGNORECASE,
)

# Known US state names → abbreviation lookup (for by-state page)
_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "District of Criminals": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
_STATE_NAMES = set(_STATE_ABBR.keys())
_STATE_ABBR_SET = set(_STATE_ABBR.values())


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    return s


def fetch_by_state_html() -> str:
    s = _session()
    r = s.get(BY_STATE_URL, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


def fetch_by_city_html() -> str:
    s = _session()
    r = s.get(BY_CITY_URL, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_main_container(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """Return the single large gp-text-container that holds the practitioner
    list.  It is identified by being the longest one in the page."""
    containers = soup.find_all("div", class_="gp-text-container")
    if not containers:
        return None
    return max(containers, key=lambda c: len(c.get_text()))


def _is_state_header(p_tag) -> Optional[str]:
    """Return state name string if this <p> is a state header, else None.

    State headers: the <strong> child contains exactly a bare state name
    (possibly with trailing whitespace/punctuation) and nothing else in the <p>
    outside the strong tag matters."""
    strong = p_tag.find("strong")
    if not strong:
        return None
    txt = strong.get_text(strip=True)
    # Strip trailing punctuation
    txt = txt.rstrip(".,: \t")
    if txt in _STATE_NAMES:
        return txt
    return None


def _parse_city_state_from_text(txt: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (city, state_name, state_abbr) from a city line, or (None,None,None)."""
    m = _CITY_STATE_LONG_RE.match(txt)
    if m:
        city = m.group(1).strip()
        state_name = m.group(2).strip()
        abbr = m.group(3)
        return city, state_name, abbr
    m = _CITY_STATE_SHORT_RE.match(txt)
    if m:
        city = m.group(1).strip()
        abbr = m.group(2)
        if abbr in _STATE_ABBR_SET:
            return city, None, abbr
    return None, None, None


def _is_city_line_by_state(p_tag) -> Optional[tuple[str, str, str]]:
    """On the by-state page a city line is a <p> WITHOUT a strong tag whose
    text matches the city+state pattern.  Returns (city, state_name, abbr) or None."""
    if p_tag.find("strong"):
        return None
    txt = p_tag.get_text(separator=" ", strip=True)
    city, state_name, abbr = _parse_city_state_from_text(txt)
    if city:
        return city, state_name or "", abbr
    return None


def _is_city_header_by_city(p_tag) -> Optional[tuple[str, str]]:
    """On the by-city page a city header is a <p><strong>...</strong></p> whose
    strong text (after stripping leading '-' and whitespace) matches a city+state
    pattern.  Returns (city, state_abbr) or None."""
    strong = p_tag.find("strong")
    if not strong:
        return None
    txt = strong.get_text(strip=True)
    # Strip leading dash separators like "- " or "-\n"
    txt = re.sub(r"^-+\s*", "", txt).strip()
    if not txt:
        return None
    city, _, abbr = _parse_city_state_from_text(txt)
    if city and abbr:
        return city, abbr
    return None


def _extract_name_from_line(line: str) -> str:
    """Best-effort: take the name portion from a practitioner data line.
    The name is typically the first comma-delimited field before credentials
    (OD, MS, FCOVD, etc.) or parenthetical notes.
    Strip leading bullet '•' characters."""
    # Remove bullet prefix
    line = re.sub(r"^[•\-\s]+", "", line).strip()
    # Strip " Note: ..." suffixes
    line = re.sub(r"\s+Note:.*$", "", line, flags=re.IGNORECASE)
    # Remove trailing phone, website, email fragments
    line = _PHONE_RE.sub("", line)
    line = _WEBSITE_RE.sub("", line).strip()
    # The name is before the first comma or credential abbreviation
    # Credentials pattern: 2-5 uppercase letters, possibly with dots (O.D., FCOVD)
    cred_pat = re.compile(r",\s*(?:[A-Z][A-Z.]{1,6})")
    m = cred_pat.search(line)
    if m:
        name = line[:m.start()].strip()
    else:
        # Use full line if no credential separator found
        name = line.strip()
    # Remove parenthetical location hints like "(Tulsa OK)" at end
    name = re.sub(r"\s*\([^)]+\)\s*$", "", name).strip()
    return name


def _extract_phone(text: str) -> Optional[str]:
    m = _PHONE_RE.search(text)
    return m.group(0).strip() if m else None


def _extract_website(text: str) -> Optional[str]:
    """Extract the first website-looking token from a line."""
    # Don't pick up email addresses
    clean = re.sub(r"\S+@\S+", "", text)
    m = _WEBSITE_RE.search(clean)
    if m:
        site = m.group(0).strip().rstrip(".,/")
        # Reject if it looks like a phone fragment
        if _PHONE_RE.match(site):
            return None
        return site
    return None


def _slug(name: str, idx: int) -> str:
    """Create a URL-safe slug from a practitioner name and index."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{slug}-{idx}"


def _build_row(
    name: str,
    data_lines: list[str],
    city: Optional[str],
    state: Optional[str],
    idx: int,
    page: str,  # "by-state" or "by-city"
) -> Optional[NormalizedPractitionerRow]:
    """Build a NormalizedPractitionerRow from parsed fragments."""
    if not name:
        return None
    # Best-effort field extraction from data_lines
    phone: Optional[str] = None
    website: Optional[str] = None
    for line in data_lines:
        if not phone:
            phone = _extract_phone(line)
        if not website:
            website = _extract_website(line)

    source_url = (
        f"{BASE}/practitioner-finder-{page}#{_slug(name, idx)}"
    )
    return NormalizedPractitionerRow(
        tier="eyehealing",
        name=name,
        specialties=infer_eyehealing_specialties(name),
        source_org="eyehealingcenter",
        source_url=source_url,
        city=city,
        state=state,
        phone=phone,
        website=website,
        country="US",
    )


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------

def parse_by_state_html(html: str) -> list[NormalizedPractitionerRow]:
    """Parse the by-state listing page into normalized rows.

    Walk the page top-down tracking (state, city).  State headers are <p>
    tags with a bare state name in <strong>.  City lines are plain <p> tags
    matching 'City, State (XX)'.  Practitioner data lines immediately follow
    each city line and continue until the next city or state header.

    Tolerant: skips blocks that lack an extractable name, or are placeholder
    'looking for' lines."""
    soup = BeautifulSoup(html, "html.parser")
    container = _find_main_container(soup)
    if not container:
        return []

    paras = container.find_all("p")
    rows: list[NormalizedPractitionerRow] = []
    current_state: Optional[str] = None
    current_city: Optional[str] = None
    pending_data: list[str] = []   # lines collected for current practitioner block
    idx = 0

    def flush_block(data_lines: list[str]) -> None:
        nonlocal idx
        if not data_lines:
            return
        # The first line should contain the practitioner name
        first = data_lines[0]
        if _PLACEHOLDER_RE.search(first):
            return
        name = _extract_name_from_line(first)
        if not name:
            return
        row = _build_row(name, data_lines[1:], current_city, current_state, idx, "by-state")
        if row:
            rows.append(row)
            idx += 1

    for p in paras:
        # Check for state header
        state_name = _is_state_header(p)
        if state_name:
            # Flush any pending data before state transition
            flush_block(pending_data)
            pending_data = []
            current_state = _STATE_ABBR.get(state_name, state_name)
            current_city = None
            continue

        # Check for city sub-header (no strong, city pattern)
        city_info = _is_city_line_by_state(p)
        if city_info and current_state is not None:
            city, _, abbr = city_info
            # If city text is actually a practitioner line with the city embedded
            # (happens when they pack city + data into one line), skip as city header
            # only if it cleanly matches the city pattern.
            flush_block(pending_data)
            pending_data = []
            current_city = city
            continue

        if current_state is None:
            # Skip preamble before the first state header
            continue

        # Data line
        txt = p.get_text(separator=" ", strip=True)
        if not txt:
            continue

        # Some practitioners are on single packed lines with multiple names
        # (e.g. "Gary Etting OD...  encino: (818)... LA: (310)...")
        # We just keep the whole thing as one block — name extraction will
        # grab the first name, phone grabs the first phone.
        pending_data.append(txt)

    # Flush final block
    flush_block(pending_data)
    return rows


def parse_by_city_html(html: str) -> list[NormalizedPractitionerRow]:
    """Parse the by-city listing page into normalized rows.

    City headers are <p> tags with <strong> text matching "- City, State (XX)"
    or short "City ST" format.  Some practitioners are denoted with bullet (•)
    in their own strong tags — these count as the practitioner name line.
    Non-strong lines are contact data.

    Walk top-down, tracking current city/state.  Flush a practitioner block
    each time a new name-bearing line appears within a city section."""
    soup = BeautifulSoup(html, "html.parser")
    container = _find_main_container(soup)
    if not container:
        return []

    paras = container.find_all("p")
    rows: list[NormalizedPractitionerRow] = []
    current_city: Optional[str] = None
    current_state: Optional[str] = None
    pending_name: Optional[str] = None
    pending_data: list[str] = []
    idx = 0

    def flush_practitioner() -> None:
        nonlocal idx, pending_name, pending_data
        if not pending_name:
            pending_data = []
            return
        if _PLACEHOLDER_RE.search(pending_name):
            pending_name = None
            pending_data = []
            return
        name = _extract_name_from_line(pending_name)
        if name:
            row = _build_row(name, pending_data, current_city, current_state, idx, "by-city")
            if row:
                rows.append(row)
                idx += 1
        pending_name = None
        pending_data = []

    for p in paras:
        # Check if this is a city header
        city_info = _is_city_header_by_city(p)
        if city_info:
            flush_practitioner()
            city, abbr = city_info
            current_city = city
            current_state = abbr
            # Also check if the <strong> has additional content after the city header
            # (on the first paragraph "Alphabetical by City ... Akron, Ohio (OH)")
            # that case is handled by the Akron data paragraphs following.
            continue

        if current_city is None:
            # Skip preamble
            continue

        strong = p.find("strong")
        txt = p.get_text(separator=" ", strip=True)
        if not txt:
            continue

        if strong:
            # A strong tag inside a non-city-header para = bullet-denoted practitioner
            # name.  Flush previous block and start a new one.
            strong_txt = strong.get_text(strip=True)
            # Skip pure separator lines like "•" or "-" alone
            clean = re.sub(r"^[•\-\s]+", "", strong_txt).strip()
            if not clean:
                continue
            flush_practitioner()
            pending_name = txt  # full text of the <p> as the name line
        else:
            # Non-strong = data line (phone, website, address, etc.)
            if pending_name is None:
                # First non-strong line after city header = practitioner name
                pending_name = txt
            else:
                pending_data.append(txt)

    # Flush final block
    flush_practitioner()
    return rows
