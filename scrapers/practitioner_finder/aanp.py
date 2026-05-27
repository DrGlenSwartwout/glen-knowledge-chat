"""American Association of Naturopathic Physicians (AANP) scraper.

AANP (naturopathic.org — NOT to be confused with the unrelated American
Association of *Nurse* Practitioners at aanp.org) publishes its public
"Find an ND" directory through a YourMembership / AssociationVoice CMS.

Live structure (re-discovered 2026-05-27 after the site migrated off
the original table-grid layout some time after the 2022 Wayback
captures):

- The public search form at ``/search/custom.asp?id=5613`` POSTs into
  ``/search/search.asp`` (or accepts the same params as GET, e.g.
  ``?txt_state=California``).
- ``/search/search.asp`` returns a shell page that embeds an iframe
  pointing at ``/searchserver/people2.aspx?id=<session-uuid>``. The
  iframe URL carries the actual paginated practitioner card list. The
  session UUID is one-shot per search.
- The whole site is behind Cloudflare bot mitigation, so the live
  migrate runner uses Playwright (see ``migrate_aanp.py``).

Each iframe response page contains:

  <span id="DocCount">N</span> Records Found     # may be "1000+" for unbounded queries
  <ul id="search-results">
    <li>
      <div class="memb-result-item">
        <div class="memb-img-wrap">
          <a href="/members/?id=NNNN">...</a>
        </div>
        <div class="memb-info-wrap">
          <p class="name">
            <span><a href="/members/?id=NNNN" class="normalName">Dr. First Last</a></span>
          </p>
          <p class="address">City<br>State<br>Postal<br></p>
          <p class="distance_radius"></p>
        </div>
      </div>
    </li>
    ... (24 cards / page in production) ...
  </ul>

  Page <current> of <total>

The list page is sparse — only name + member_id + (city, state, postal).
Street, phone, email, website, credentials, practice_name all live on
the per-member profile page at ``/members/?id=<numeric>``. So this is
a two-stage scrape pattern: list-page card -> per-row profile fetch.
The profile page structure (``tdEmployerName`` + ``tdWorkPhone`` +
``CstmFldLbl/CstmFldVal`` custom field rows) is unchanged from the
original layout and ``parse_profile_html`` continues to work against it.

Pagination on the live iframe is ASP.NET WebForms ``__doPostBack``
targeted at ``SearchResultsGrid$ctlNN$ctlMM`` (NOT the older
``Page$N`` event-argument style). The migrate runner replays the
__VIEWSTATE / __EVENTVALIDATION hidden fields to walk the page set.

Fellowship rule
---------------
Naturopathy doesn't have a clean "Fellow" tier analogous to OEPF/IAOMT's
fellowship grade. AANP's ladder is just "Member" — there is no public
"Fellow of AANP" or "Diplomate" credential exposed in the public
directory; the AANP "Active Member" status is a single-tier credential
gated only on holding a valid state ND license. We therefore default
ALL AANP rows to ``fellowship_level=False`` UNLESS the per-row name
string carries one of the recognized vetted post-nominals (FAANP,
FABNO, FACO, FABNE — board specialty certifications from naturopathic
specialty boards, the closest structural analogue we have in this
profession to OD/MD fellow status). The list-grid rows don't carry
credentials; this only matters when the parser is fed a profile dict
that includes the Credentials field — see ``_detect_fellowship_creds``.

The per-practitioner ``source_url`` is the canonical member-detail URL
``https://naturopathic.org/members/?id=<id>`` and is stable across
re-runs (the numeric ``id`` is the YourMembership account id).
"""
import html as html_module
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://naturopathic.org"
DIRECTORY_FORM_URL = f"{BASE}/search/custom.asp?id=5613"
SEARCH_URL = f"{BASE}/search/search.asp"

LOCKED_SPECIALTIES = ["naturopathy", "holistic_health"]

# Recognized vetted post-nominals that mark a naturopathic Fellow / Diplomate
# tier. FAANP = Fellow of the American Association of Naturopathic
# Physicians (honorary, by election); FABNO = Fellow of the American
# Board of Naturopathic Oncology; FACO = Fellow American College
# Of (specialty boards, e.g. FACO-NM); FABNE = Fellow of the American
# Board of Naturopathic Endocrinology. "Diplomate" appears occasionally
# as a long-form credential — we match it case-insensitively.
_FELLOWSHIP_CRED_TOKENS = {
    "FAANP",
    "FABNO",
    "FACO",
    "FABNE",
    "DIPLOMATE",
}

# US state full-name set used as a *signal*: if a row div sequence has
# fewer than 5 trailing divs, the city/state/postal/country mapping
# needs to fall back to inference. We keep this list small — it's only
# the ND-licensure jurisdictions (the AANP directory predominantly lists
# US practitioners + a small Canadian fringe). Anything outside this set
# is treated as either Canadian province (full names) or "other".
_US_STATES = {
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
    # Territories occasionally seen
    "Puerto Rico", "U.S. Virgin Islands", "Guam",
}

_CA_PROVINCES = {
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan",
    "Northwest Territories", "Nunavut", "Yukon",
}

# Country-name -> ISO2 (same convention as iabdm.py / iaomt.py).
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
    "australia": "AU",
    "new zealand": "NZ",
    "ireland": "IE",
    "germany": "DE",
    "france": "FR",
    "mexico": "MX",
    "japan": "JP",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


# ---------------------------------------------------------------------------
# Stage 1: paginated state-by-state fetch (live)
# ---------------------------------------------------------------------------

def fetch_state_directory_html(state: str) -> str:
    """Fetch the rendered search-result HTML for a single US state.

    Hits ``/search/search.asp?txt_state=<State>`` with a static Mozilla UA,
    20s timeout, 0.5s polite sleep. Returns the raw HTML body. Note:
    naturopathic.org is Cloudflare-protected so a static-UA request can
    return HTTP 403 — callers should handle ``requests.HTTPError`` and
    fall back to a Playwright session if needed. The fixture-driven
    parser doesn't touch this function.
    """
    s = _session()
    r = s.get(
        SEARCH_URL,
        params={"txt_state": state},
        timeout=20,
    )
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


def fetch_iframe_results_html(session_uuid: str) -> str:
    """Fetch the iframe results page for a previously-issued search session.

    The YourMembership search splits the form (``/search/search.asp``)
    from the results grid (``/searchserver/people2.aspx?id=<uuid>``).
    The uuid is one-shot per search submission — extracted from the
    iframe ``src`` attribute on the search.asp response page.
    """
    s = _session()
    url = f"{BASE}/searchserver/people2.aspx"
    r = s.get(
        url,
        params={
            "id": session_uuid,
            "cdbid": "",
            "canconnect": "0",
            "canmessage": "0",
            "map": "True",
            "toggle": "True",
            "hhSearchTerms": "",
        },
        timeout=20,
    )
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
        # Strip stray nbsp left over from inline HTML extraction.
        s = s.replace("\xa0", " ").strip()
        return s or None
    s = str(val).strip()
    return s or None


def _strip_html_tags(s: str) -> str:
    """Drop all HTML tags from a snippet; collapse whitespace.

    Block-level tags (``<br>``, ``<div>``, ``<p>``, ``<td>``, ``<tr>``,
    ``<li>``, ``<h1-6>``) are replaced with a single space so adjacent
    block contents don't smash together. Inline tags (``<a>``,
    ``<span>``, ``<b>``, etc.) are removed cleanly so that anchor-
    delimited comma lists like ``<a>LAc</a>, <a>NMD</a>, <a>Other</a>``
    yield ``"LAc, NMD, Other"`` and not ``"LAc , NMD , Other"``.
    """
    if not s:
        return ""
    # Block-level open + close tags -> space.
    out = re.sub(r"<(br|p|div|td|tr|li|ul|ol|h\d)[^>]*>", " ", s, flags=re.I)
    out = re.sub(r"</(br|p|div|td|tr|li|ul|ol|h\d)>", " ", out, flags=re.I)
    # Anything else (inline tags + their attrs) -> removed.
    out = re.sub(r"<[^>]+>", "", out)
    out = html_module.unescape(out)
    # Tidy up spacing around commas (anchor-stripped lists leave bare
    # ", " sequences with stray double-spaces).
    out = re.sub(r"\s*,\s*", ", ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _strip_credentials(name: str) -> tuple[str, Optional[str]]:
    """Split 'Dr. Jane Doe, ND, FABNO' into (clean_name, credentials).

    Mirrors the IABDM helper's intent: credentials are the trailing
    comma-separated short uppercase abbreviations (with optional dots).
    Honorifics (``Dr.``, ``Dra.``) are preserved on the name."""
    if not name:
        return "", None
    s = name.strip()
    paren = re.match(r"^(.*?)\s*\(([A-Za-z][A-Za-z.,\s/-]*)\)\s*$", s)
    if paren:
        s = f"{paren.group(1).strip()}, {paren.group(2).strip()}"

    # Credential tokens are 2+ short upper-case letters (with optional
    # internal dots/slashes); 2-char minimum lets us catch "ND" — the
    # default ND credential is the most common in the AANP directory.
    cred_pat = re.compile(r",\s*([A-Za-z][A-Za-z./]*[A-Za-z])")
    m = cred_pat.search(s)
    if not m:
        return s.rstrip(", "), None
    clean = s[: m.start()].strip().rstrip(",")
    creds = s[m.start():].lstrip(", ").strip().rstrip(",").rstrip()
    return clean, creds or None


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Add an https:// scheme to a bare domain; reject obviously bad URLs."""
    s = _coerce_str(raw)
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    # Reject malformed shell entries like 'mailto:' or javascript:.
    if s.startswith("mailto:") or s.startswith("javascript:") or s.startswith("#"):
        return None
    return f"https://{s}"


def _country_iso2_from_name(raw: Optional[str]) -> Optional[str]:
    """Map a free-text country name to ISO2; None if unrecognized."""
    s = _coerce_str(raw)
    if not s:
        return None
    return _COUNTRY_NAME_TO_ISO2.get(s.lower())


def _infer_country_from_state(state: Optional[str]) -> str:
    """When the row doesn't include an explicit country, infer from state.

    US states + territories -> ``US``; Canadian provinces -> ``CA``;
    everything else (including ``None``) -> ``US`` (the AANP directory
    is dominated by US practitioners; this is the safe default)."""
    s = _coerce_str(state)
    if not s:
        return "US"
    if s in _US_STATES:
        return "US"
    if s in _CA_PROVINCES:
        return "CA"
    return "US"


def _detect_fellowship_creds(credentials: Optional[str]) -> bool:
    """True when the credential string contains FAANP / FABNO / FACO /
    FABNE / Diplomate (case-insensitive token match).

    See the module docstring for the rule rationale: AANP has no
    public "Fellow" tier the way the eye/dental orgs do, so we treat
    only specialty-board fellow / diplomate credentials as qualifying."""
    s = _coerce_str(credentials)
    if not s:
        return False
    for chunk in re.split(r"[,;/]\s*|\s+", s):
        token = chunk.strip().rstrip(".").upper()
        token = re.sub(r"[^A-Z]", "", token)
        if token in _FELLOWSHIP_CRED_TOKENS:
            return True
    return False


def _build_source_url(member_id: str) -> str:
    """Canonical detail-page URL for a member id."""
    return f"{BASE}/members/?id={member_id}"


# ---------------------------------------------------------------------------
# Search-results card extraction (live ul#search-results layout)
# ---------------------------------------------------------------------------

# Card-name anchor: <a href="/members/?id=NNN" ... class="normalName">Name</a>.
# Accepts both relative and (defensively) absolute hrefs.
_CARD_NAME_ANCHOR_RE = re.compile(
    r'<a\s+href="(?:https?://[^"]*naturopathic\.org)?/members/\?id=(\d+)"'
    r'[^>]*class="normalName"[^>]*>(.*?)</a>',
    re.S | re.I,
)

# Card address paragraph: <p class="address">City<br>State<br>Postal<br></p>.
# Captures the inner HTML (whose <br>-delimited tokens we split on).
_CARD_ADDRESS_RE = re.compile(
    r'<p\s+class="address"[^>]*>(.*?)</p>',
    re.S | re.I,
)


def _parse_card(card_html: str) -> Optional[dict]:
    """Pull a single member's data out of one ``<li>`` / ``memb-result-item`` card.

    Returns a dict with keys ``id``, ``name``, ``addr_tokens`` (the
    list of address-line tokens lifted from ``<p class="address">``,
    in document order) or None if no name anchor is present.
    """
    name_m = _CARD_NAME_ANCHOR_RE.search(card_html)
    if not name_m:
        return None
    member_id = name_m.group(1)
    name_raw = _strip_html_tags(name_m.group(2))
    if not name_raw:
        return None

    addr_tokens: list[str] = []
    addr_m = _CARD_ADDRESS_RE.search(card_html)
    if addr_m:
        addr_inner = addr_m.group(1)
        # Split on <br> tags; each fragment becomes a stripped token.
        for piece in re.split(r"<br\s*/?>", addr_inner, flags=re.I):
            text = _strip_html_tags(piece)
            if text:
                addr_tokens.append(text)

    return {"id": member_id, "name": name_raw, "addr_tokens": addr_tokens}


def _card_to_normalized_row(card: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: parsed card dict -> NormalizedPractitionerRow stub.

    The list-page card carries ONLY name + member_id + (city, state, postal).
    Street, country, phone, email, website, credentials, practice_name all
    live on the profile page and are filled in by ``parse_profile_html``
    when the migrate runner does the per-row enrichment fetch.

    Returns None when no usable name is recovered. Address tokens are
    matched by content type (state-name match wins) so missing fields
    don't shift positional indexing.
    """
    name_raw = card.get("name")
    if not name_raw:
        return None

    clean_name, credentials = _strip_credentials(name_raw)
    if not clean_name:
        return None

    tokens = list(card.get("addr_tokens", []))

    # Find the state token (full US state name OR Canadian province);
    # then the city is whatever comes before it and the postal is
    # whatever comes after. Postal must match a US ZIP / Canadian
    # postcode shape; anything else trailing the state is discarded.
    state_idx: Optional[int] = None
    for i, t in enumerate(tokens):
        if t in _US_STATES or t in _CA_PROVINCES:
            state_idx = i
            break

    city = state = postal = None
    if state_idx is not None:
        state = tokens[state_idx] or None
        if state_idx >= 1:
            city = tokens[state_idx - 1] or None
        # Postal: next token after the state, if it shape-matches.
        if state_idx + 1 < len(tokens):
            cand = tokens[state_idx + 1]
            cand_compact = cand.replace(" ", "")
            if (
                re.match(r"^\d{5}(?:-\d{4})?$", cand)
                or re.match(r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$", cand, re.I)
                or re.match(r"^[A-Z]\d[A-Z]\d[A-Z]\d$", cand_compact, re.I)
            ):
                postal = cand

    country_iso = _infer_country_from_state(state) or "US"

    return NormalizedPractitionerRow(
        tier="org_member",
        name=clean_name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="AANP",
        source_url=_build_source_url(card["id"]),
        fellowship_level=_detect_fellowship_creds(credentials),
        practice_name=None,
        credentials=credentials,
        phone=None,
        email=None,
        website=None,
        address1=None,   # street is profile-only
        city=city,
        state=state,
        postal=postal,
        country=country_iso,
    )


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------

def parse_search_results_html(html: str) -> list[NormalizedPractitionerRow]:
    """Pure parser: takes a ``/searchserver/people2.aspx`` response HTML
    (the iframe page) and returns one NormalizedPractitionerRow stub per
    ``<li><div class="memb-result-item">`` card. No I/O.

    The card markup is stable across page-1..N — pagination is handled
    by the caller (``__doPostBack`` replay against the iframe URL).
    This parser is page-scoped: it returns whatever is on the page handed
    to it.

    The list-page rows are stubs: only ``name``, ``source_url``, and
    (when present) ``city``/``state``/``postal`` are populated. The
    migrate runner enriches each stub by fetching the corresponding
    profile page and merging fields.
    """
    if not isinstance(html, str):
        return []

    # Isolate the search-results UL so we don't accidentally pull member
    # IDs from the Google Maps marker JS, the "Featured Members" sidebar,
    # or the original ``<table id="SearchResultsGrid">`` (the live JS
    # transforms the table into the UL on render, but the server-side
    # HTML always carries the UL).
    ul_marker = 'id="search-results"'
    u_idx = html.find(ul_marker)
    if u_idx < 0:
        return []
    # Walk forward to the closing </ul> tag so we don't accidentally
    # consume the next UL after this one.
    end_idx = html.find("</ul>", u_idx)
    list_html = html[u_idx : end_idx if end_idx > u_idx else len(html)]

    rows: list[NormalizedPractitionerRow] = []
    # Split on the start of each memb-result-item card; skip the first
    # split (preamble before first card).
    parts = list_html.split('<div class="memb-result-item"')
    for chunk in parts[1:]:
        card = _parse_card(chunk)
        if card is None:
            continue
        normalized = _card_to_normalized_row(card)
        if normalized is not None:
            rows.append(normalized)
    return rows


def parse_record_count(html: str) -> Optional[int]:
    """Extract the ``DocCount`` (total results across all pages) from a
    search-result page. Returns None if not found.

    The live site renders ``1000+`` as a soft cap for unbounded queries;
    we strip the trailing ``+`` and return the numeric portion (the
    actual total is then established by walking pages until exhausted).
    """
    if not isinstance(html, str):
        return None
    m = re.search(r'<span\s+id="DocCount"[^>]*>(\d+)\+?</span>', html)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def parse_page_info(html: str) -> Optional[tuple[int, int]]:
    """Extract ``(current_page, total_pages)`` from the result page.

    Returns None if the page-of-N text isn't present."""
    if not isinstance(html, str):
        return None
    m = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', html)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Profile-page parser (used to enrich list-grid rows with phone / website /
# credentials / practice_name when the migration script chooses to do per-
# row fetches).
# ---------------------------------------------------------------------------

def _extract_title_name(html: str) -> Optional[str]:
    """Pull the practitioner name from the <title> tag.

    The detail page title is consistently ``Dr. First Last - American
    Association of Naturopathic Physicians`` — split on the trailing
    em-dash + suffix."""
    m = re.search(r"<title>([^<]+)</title>", html, re.I)
    if not m:
        return None
    t = html_module.unescape(m.group(1)).strip()
    # Strip the trailing " - American Association of Naturopathic Physicians"
    t = re.sub(r"\s*-\s*American Association of Naturopathic Physicians\s*$", "", t).strip()
    return t or None


_EMPLOYER_TD_RE = re.compile(r'id="tdEmployerName"[^>]*>(.*?)</td>', re.S | re.I)
_WORKPHONE_TD_RE = re.compile(r'id="tdWorkPhone"[^>]*>(.*?)</td>', re.S | re.I)
_CUSTOM_FIELD_RE = re.compile(
    r'class="CstmFldLbl"[^>]*>([^<:]+?)\s*:\s*</label>'
    r'[^<]*</td>\s*<td[^>]*class="CstmFldVal"[^>]*>(.*?)</td>',
    re.S,
)


def _parse_employer_block(td_html: str) -> dict:
    """Pull (practice_name, address1, city, state, postal, country) out of
    the ``tdEmployerName`` cell.

    The cell has a deterministic structure where each anchor link points
    to a search-by-field URL: the first anchor (if its href is
    ``txt_employName=...``) is the practice name; subsequent anchors are
    city, state, country. Plain text between anchors is the street.
    """
    out: dict = {}
    if not td_html:
        return out

    # Find all anchor (href, text) pairs.
    anchors = re.findall(r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>', td_html, re.S | re.I)
    # And all stray text fragments (after stripping tags).
    # Build a positional decomposition: iterate piece-by-piece.
    pieces: list[tuple[str, str, Optional[str]]] = []
    for chunk in re.split(r'(<a[^>]*>.*?</a>|<br[^>]*/?>|<[^>]+>)', td_html, flags=re.S | re.I):
        chunk = chunk or ""
        if not chunk.strip():
            continue
        if chunk.startswith("<a"):
            m = re.search(r'href="([^"]+)"[^>]*>(.*?)</a>', chunk, re.S | re.I)
            if m:
                pieces.append(("a", _strip_html_tags(m.group(2)), m.group(1)))
        elif chunk.startswith("<"):
            continue
        else:
            txt = _strip_html_tags(chunk).replace("[", "").replace("]", "").strip()
            if txt and txt != "Map":
                pieces.append(("text", txt, None))

    practice_name = None
    address_lines: list[str] = []
    city = state = postal = country = None

    for kind, val, href in pieces:
        if val == "Map":
            continue
        if kind == "a":
            href_l = (href or "").lower()
            if "txt_employname=" in href_l:
                practice_name = val
            elif "txt_city=" in href_l:
                city = val
            elif "txt_state=" in href_l:
                state = val
            elif "txt_country=" in href_l:
                country = val
        else:
            # Plain-text fragment: either a street line or a postal code.
            # Postal pattern: US ZIP (12345 or 12345-6789) or Canadian
            # postcode (A1A 1A1 / A1A1A1). Anything else is treated as
            # a street line — the AANP profiles list 1-2 street lines
            # ("95 S Main St", "Fl 2") between the practice-name anchor
            # and the city anchor.
            postal_token = val.replace(" ", "")
            if (
                re.match(r"^\d{5}(?:-\d{4})?$", val)
                or re.match(r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$", val, re.I)
                or re.match(r"^[A-Z]\d[A-Z]\d[A-Z]\d$", postal_token, re.I)
            ):
                postal = val
            elif val:
                address_lines.append(val)

    out["practice_name"] = practice_name
    if address_lines:
        out["address1"] = ", ".join(address_lines)
    if city:
        out["city"] = city
    if state:
        out["state"] = state
    if postal:
        out["postal"] = postal
    if country:
        out["country"] = country
    return out


def _parse_workphone_block(td_html: str) -> dict:
    """Pull phone + website out of the ``tdWorkPhone`` cell.

    Structure varies: ``203 288-8283 (Phone)`` for offices with a phone,
    or a ``Visit Website`` anchor (or both). We pick the first
    ``(Phone)``-tagged number and the first http anchor labeled
    ``Visit Website``."""
    out: dict = {}
    if not td_html:
        return out
    phone_m = re.search(r"([\d\.\-\(\)\s]{7,})\s*<span[^>]*>\(Phone\)</span>", td_html, re.I)
    if phone_m:
        ph = _strip_html_tags(phone_m.group(1))
        ph = re.sub(r"\s+", " ", ph).strip()
        if ph:
            out["phone"] = ph
    site_m = re.search(r'<a\s+href="(https?://[^"]+)"[^>]*>\s*Visit Website', td_html, re.I)
    if site_m:
        out["website"] = site_m.group(1)
    return out


def _parse_custom_fields(html: str) -> dict:
    """Pull the labeled custom fields (Credentials, Clinic Email, ...) out
    of a member profile page."""
    out: dict = {}
    for m in _CUSTOM_FIELD_RE.finditer(html):
        label = m.group(1).strip()
        value_html = m.group(2)
        value = _strip_html_tags(value_html)
        if not value:
            continue
        out[label] = value
    return out


def parse_profile_html(
    html: str, member_id: Optional[str] = None
) -> Optional[NormalizedPractitionerRow]:
    """Pure parser: takes a ``/members/?id=<id>`` detail page HTML and
    returns a fully-populated NormalizedPractitionerRow. Returns None
    if no usable name is found.

    The detail page carries: name (from <title>), practice name +
    address (from tdEmployerName), phone + website (from tdWorkPhone),
    and Credentials / Clinic Email / Practice Focus / Treatment Modalities
    (from the CstmFld custom-field table).

    ``member_id`` (optional) is used to construct ``source_url``. The
    live profile pages no longer carry a ``/members/?id=<n>`` anchor in
    the body, so the migrate runner must pass the id it used to fetch
    the page. If omitted we still try to recover one from any in-page
    anchor for backwards compatibility with the older fixtures.
    """
    if not isinstance(html, str):
        return None

    name_raw = _extract_title_name(html)
    if not name_raw:
        return None
    clean_name, title_creds = _strip_credentials(name_raw)
    if not clean_name:
        return None

    # Caller-supplied id wins; else look for any /members/?id=NNN anchor
    # on the page (the older fixtures carry one in the edit-profile link).
    if member_id is None:
        id_m = re.search(r'/members/\?id=(\d+)', html)
        member_id = id_m.group(1) if id_m else None

    employer_m = _EMPLOYER_TD_RE.search(html)
    employer = _parse_employer_block(employer_m.group(1) if employer_m else "")

    phone_m = _WORKPHONE_TD_RE.search(html)
    workphone = _parse_workphone_block(phone_m.group(1) if phone_m else "")

    fields = _parse_custom_fields(html)

    # Credentials come from the Credentials custom field first, falling
    # back to the title-extracted creds.
    credentials = _coerce_str(fields.get("Credentials")) or title_creds
    email = _coerce_str(fields.get("Clinic Email"))

    # Practice name: prefer the employer-block link; if it equals the
    # practitioner name (solo practice, the employer link is just the
    # name) suppress it. Match against both the exact name and the
    # honorific-stripped form ("Dr. Joshua Levitt" vs "Joshua Levitt").
    practice_name = _coerce_str(employer.get("practice_name"))
    if practice_name:
        bare_name = re.sub(r"^(?:Dr|Dra|Mr|Mrs|Ms|Mx)\.?\s+", "", clean_name, flags=re.I).strip()
        if (
            practice_name.lower() == clean_name.lower()
            or practice_name.lower() == bare_name.lower()
        ):
            practice_name = None

    state = _coerce_str(employer.get("state"))
    country_iso = (
        _country_iso2_from_name(employer.get("country"))
        or _infer_country_from_state(state)
        or "US"
    )

    return NormalizedPractitionerRow(
        tier="org_member",
        name=clean_name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="AANP",
        source_url=_build_source_url(member_id) if member_id else None,
        fellowship_level=_detect_fellowship_creds(credentials),
        practice_name=practice_name,
        credentials=credentials,
        phone=_coerce_str(workphone.get("phone")),
        email=email,
        website=_normalize_website(workphone.get("website")),
        address1=_coerce_str(employer.get("address1")),
        city=_coerce_str(employer.get("city")),
        state=state,
        postal=_coerce_str(employer.get("postal")),
        country=country_iso,
    )
