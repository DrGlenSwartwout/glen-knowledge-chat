"""National Association of Nutrition Professionals (NANP) scraper.

NANP publishes its public "Find a Practitioner" directory through the same
YourMembership / AssociationVoice CMS as AANP (naturopathic.org). NANP and
AANP are sister deployments on the same vendor — the iframe-card pattern,
the profile-page selectors, and the ASP.NET ``__doPostBack`` pagination
are all identical. This adapter is a near-mirror of ``aanp.py`` with the
NANP-specific URLs / source_org / specialties / fellowship rule swapped in.

Live structure (re-discovered 2026-05-27 against ``mynanp.nanp.org``):

- The public search form at ``/search/`` is the warm-up GET. The actual
  search-shell endpoint is ``/search/newsearch.asp`` (NOT
  ``/search/search.asp`` — that's the AANP path). Both paths return a
  shell page that embeds an iframe pointing at
  ``/searchserver/people2.aspx?id=<session-uuid>``.
- ``/searchserver/people2.aspx?id=<uuid>`` returns the paginated
  practitioner card list — ``<ul id="search-results">`` of
  ``<li><div class="memb-result-item">`` cards (24 per page on the live
  capture, DocCount=673 total, page 1 of 29).
- The whole site is behind Cloudflare bot mitigation, so the live migrate
  runner uses Playwright (see ``migrate_nanp.py``).
- The iframe response page is structurally IDENTICAL to AANP — the same
  card markup, the same ``<span id="DocCount">N</span>`` total, the same
  ``Page X of Y`` text. The list-page parser is byte-for-byte the same as
  the AANP parser; only the locked-invariant constants change.

Each iframe card carries only: name + member_id + (city, state). The
postal column is empty on NANP's live capture (the synthesized fixture
included postal in the card, but live cards do not). Street, country,
phone, email, website, credentials, practice_name all live on the
per-member profile page at ``/members/?id=<numeric>``.

The profile page structure (``tdEmployerName`` + ``tdWorkPhone`` +
``CstmFldLbl/CstmFldVal`` custom field rows) is the standard YourMembership
template — same as AANP. Two NANP-specific differences worth flagging:

  - NANP profiles often expose the phone via ``tdHomePhone`` instead of
    (or in addition to) ``tdWorkPhone``. We accept either.
  - NANP profiles do NOT use the ``txt_employName=`` anchor for the
    practice-name link. Instead the practice name (when present) is the
    first plain-text fragment inside ``tdEmployerName``, ABOVE the city
    anchor. We treat the first non-postal text fragment that doesn't
    start with a digit as the practice name; numeric-leading fragments
    are treated as a street line (per AANP behaviour). This also
    correctly handles the (common) case where the employer block opens
    with a bare city anchor (no practice, no street).
  - NANP renders the practitioner email via a JS-decrypted ``mailto:``
    anchor in the right-column header (not via the AANP "Clinic Email"
    custom field). We pick the first non-empty mailto whose value is a
    real email and isn't the ``info@nanp.org`` site-footer link.

Fellowship rule (NANP-specific)
-------------------------------
NANP publishes a tiered membership ladder:

    Student Member
    Associate Member        (in school, no degree yet)
    Professional Member     (degree from a NANP-approved school)
    BCHN(R) Credentialed    (Board Certified Holistic Nutritionist —
                             exam-vetted board certification, the elite
                             tier of the Professional Member ladder)
    CDSP(TM) Credentialed   (Certified Dietary Supplement Professional —
                             a narrower certificate, NOT a higher tier)

BCHN is the elite, exam-vetted Board Certified Holistic Nutritionist
credential. ``fellowship_level=True`` iff one of the following signals
fires (in precedence order):

    1. The profile's ``CstmFldLbl`` / ``CstmFldVal`` block carries a row
       labelled ``BCHN`` (with optional (R) glyph or HTML entity) whose
       value reads ``Yes`` — the canonical signal from the
       ``cdlCustomFieldValueIDBCHN`` field on the live YM form. This is
       the highest-confidence signal and dominates.
    2. The credential string (from the title's trailing post-nominals or
       the ``Credentials`` custom field) contains the ``BCHN`` token,
       matched case-insensitively, with the (R) trademark glyph /
       ``&reg;`` entity / interspersed dots all collapsed before
       comparison. This catches profiles where the custom-field block is
       missing/empty but the credential is still in the title.

CDSP alone does NOT qualify — it's a separate certificate that runs
alongside the membership ladder, not above it. A member with BOTH BCHN
and CDSP qualifies (BCHN wins).

The per-practitioner ``source_url`` is the canonical member-detail URL
``https://mynanp.nanp.org/members/?id=<id>`` and is stable across re-runs
(the numeric ``id`` is the YourMembership account id).
"""
import html as html_module
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://mynanp.nanp.org"
# Warm-up GET: the live NANP search-form page lives at the bare
# ``/search/`` path (no ``?id=NNN`` query param, unlike AANP). Hitting
# this once at the start of a Playwright session primes the session
# cookies so the iframe-extraction GET on ``/search/newsearch.asp``
# returns a populated shell.
DIRECTORY_FORM_URL = f"{BASE}/search/"
# Shell page that embeds the iframe. NANP uses ``newsearch.asp`` where
# AANP uses ``search.asp``; otherwise identical.
SEARCH_URL = f"{BASE}/search/newsearch.asp"

LOCKED_SPECIALTIES = ["nutrition", "holistic_health"]

# US state full-name set used as a *signal*: when a row's address-token
# sequence has fewer than 3 trailing tokens, city/state mapping needs to
# fall back to inference. Same list shape as AANP — NANP's directory is
# similarly US-dominant with a small Canadian / international fringe.
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
    "Puerto Rico", "U.S. Virgin Islands", "Guam",
}

_CA_PROVINCES = {
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan",
    "Northwest Territories", "Nunavut", "Yukon",
}

# Country-name -> ISO2. NANP has more international reach than AANP so
# the table is broader.
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
    "israel": "IL",
    "south africa": "ZA",
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
# Stage 1: live fetch helpers (kept for parity with AANP; the migrate
# runner does the Cloudflare-clearing fetches through Playwright)
# ---------------------------------------------------------------------------

def fetch_search_shell_html() -> str:
    """Fetch the rendered search-shell HTML (a single unfiltered walk).

    Hits ``/search/newsearch.asp`` with a static Mozilla UA, 20s timeout,
    0.5s polite sleep. Returns the raw HTML body. ``mynanp.nanp.org`` is
    Cloudflare-protected so static-UA requests may return HTTP 403 —
    callers should handle ``requests.HTTPError`` and fall back to a
    Playwright session if needed. The fixture-driven parser doesn't
    touch this function.
    """
    s = _session()
    r = s.get(SEARCH_URL, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


def fetch_iframe_results_html(session_uuid: str) -> str:
    """Fetch the iframe results page for a previously-issued search session.

    The YourMembership search splits the form shell (``/search/newsearch.asp``)
    from the results grid (``/searchserver/people2.aspx?id=<uuid>``).
    The uuid is one-shot per search submission — extracted from the
    iframe ``src`` attribute on the search shell response page.
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
        s = s.replace("\xa0", " ").strip()
        return s or None
    s = str(val).strip()
    return s or None


def _strip_html_tags(s: str) -> str:
    """Drop all HTML tags from a snippet; collapse whitespace.

    Block-level tags become a single space so adjacent block contents
    don't smash together; inline tags vanish so anchor-delimited
    comma lists yield clean comma-separated values.
    """
    if not s:
        return ""
    out = re.sub(r"<(br|p|div|td|tr|li|ul|ol|h\d)[^>]*>", " ", s, flags=re.I)
    out = re.sub(r"</(br|p|div|td|tr|li|ul|ol|h\d)>", " ", out, flags=re.I)
    out = re.sub(r"<[^>]+>", "", out)
    out = html_module.unescape(out)
    out = re.sub(r"\s*,\s*", ", ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _strip_credentials(name: str) -> tuple[str, Optional[str]]:
    """Split 'Sarah Henderson, MS, BCHN' into (clean_name, credentials).

    Mirrors AANP's helper. NANP credential strings can carry (R)/(TM)
    trademark glyphs (``BCHN(R)``, ``CDSP(TM)``); ``_normalize_credential_chunk``
    strips those at compare time, but they're left in place on the
    credentials string itself so downstream consumers see the original
    decoration."""
    if not name:
        return "", None
    s = name.strip()
    # Strip ® / ™ glyphs and entities up front so the credential split
    # is clean. We compare on letter-only tokens later.
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"&reg;|&trade;", "", s, flags=re.I)

    paren = re.match(r"^(.*?)\s*\(([A-Za-z][A-Za-z.,\s/-]*)\)\s*$", s)
    if paren:
        s = f"{paren.group(1).strip()}, {paren.group(2).strip()}"

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
    everything else -> ``US`` (the NANP directory is US-dominant; this is
    the safe default).
    """
    s = _coerce_str(state)
    if not s:
        return "US"
    if s in _US_STATES:
        return "US"
    if s in _CA_PROVINCES:
        return "CA"
    return "US"


# BCHN credential token — matches BCHN with optional dots between letters,
# case-insensitively. The trademark glyph (U+00AE) and the HTML entity
# &reg; are stripped before this regex runs so we just look for the
# letter run. Word boundaries prevent false matches on tokens like
# ``ABCHNYZ``.
_BCHN_TOKEN_RE = re.compile(r"\bB\.?C\.?H\.?N\.?\b", re.IGNORECASE)


def _normalize_credential_chunk(s: str) -> str:
    """Strip the (R) / (TM) entity-or-glyph from a credential token so
    ``BCHN(R)`` / ``BCHN&reg;`` / ``BCHN`` all compare equal."""
    if not s:
        return ""
    out = html_module.unescape(s)
    out = out.replace("®", "").replace("™", "")
    return out.strip()


def _detect_fellowship_creds(credentials: Optional[str]) -> bool:
    """True when the credential string contains the BCHN token.

    Trademark decorations are stripped before the test so ``BCHN(R)``,
    ``BCHN&reg;``, ``B.C.H.N.``, ``bchn`` all match. The credential
    string can be a comma-separated list (``ND, BCHN, MS``) or a free
    text run (``Sarah Henderson, MS, BCHN, CDSP``)."""
    s = _coerce_str(credentials)
    if not s:
        return False
    cleaned = _normalize_credential_chunk(s)
    return bool(_BCHN_TOKEN_RE.search(cleaned))


def _build_source_url(member_id: str) -> str:
    """Canonical detail-page URL for a member id."""
    return f"{BASE}/members/?id={member_id}"


# ---------------------------------------------------------------------------
# Search-results card extraction (live ul#search-results layout)
# ---------------------------------------------------------------------------

# Card-name anchor: <a href="/members/?id=NNN" ... class="normalName">Name</a>.
# Accepts both relative and (defensively) absolute hrefs against either
# nanp.org or mynanp.nanp.org.
_CARD_NAME_ANCHOR_RE = re.compile(
    r'<a\s+href="(?:https?://[^"]*nanp\.org)?/members/\?id=(\d+)"'
    r'[^>]*class="normalName"[^>]*>(.*?)</a>',
    re.S | re.I,
)

# Card address paragraph: <p class="address">City<br>State<br>...<br></p>.
_CARD_ADDRESS_RE = re.compile(
    r'<p\s+class="address"[^>]*>(.*?)</p>',
    re.S | re.I,
)


def _parse_card(card_html: str) -> Optional[dict]:
    """Pull a single member's data out of one ``<li>`` / ``memb-result-item`` card.

    Returns a dict with keys ``id``, ``name``, ``addr_tokens`` (the list
    of address-line tokens lifted from ``<p class="address">``, in
    document order) or None if no name anchor is present.
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
        for piece in re.split(r"<br\s*/?>", addr_inner, flags=re.I):
            text = _strip_html_tags(piece)
            if text:
                addr_tokens.append(text)

    return {"id": member_id, "name": name_raw, "addr_tokens": addr_tokens}


def _card_to_normalized_row(card: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: parsed card dict -> NormalizedPractitionerRow stub.

    NANP list-page cards carry name + member_id + (city, state). Unlike
    AANP, the live capture's ``<p class="address">`` cards do NOT include
    a postal token — postal is profile-only on NANP. Street, country,
    phone, email, website, credentials, practice_name are all profile-
    only and merged in by the migrate runner.
    """
    name_raw = card.get("name")
    if not name_raw:
        return None

    clean_name, credentials = _strip_credentials(name_raw)
    if not clean_name:
        return None

    tokens = list(card.get("addr_tokens", []))

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
        source_org="NANP",
        source_url=_build_source_url(card["id"]),
        fellowship_level=_detect_fellowship_creds(credentials),
        practice_name=None,
        credentials=credentials,
        phone=None,
        email=None,
        website=None,
        address1=None,
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

    Identical card markup to AANP — see the AANP module docstring for the
    full structural notes. The migrate runner enriches each stub by
    fetching the corresponding ``/members/?id=<id>`` profile page and
    merging fields.
    """
    if not isinstance(html, str):
        return []

    ul_marker = 'id="search-results"'
    u_idx = html.find(ul_marker)
    if u_idx < 0:
        return []
    end_idx = html.find("</ul>", u_idx)
    list_html = html[u_idx : end_idx if end_idx > u_idx else len(html)]

    rows: list[NormalizedPractitionerRow] = []
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

    The live NANP iframe renders the exact count (e.g. ``673``) without a
    trailing ``+``; the regex defensively accepts the trailing ``+`` form
    (which the platform uses on truly-unbounded queries elsewhere) so the
    parser stays portable across sister deployments.
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

    Returns None if the ``Page X of Y`` text isn't present.
    """
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
# credentials / practice_name)
# ---------------------------------------------------------------------------

def _extract_title_name(html: str) -> Optional[str]:
    """Pull the practitioner name from the <title> tag.

    NANP profile titles are consistently ``First Last - National Association
    of Nutrition Professionals (NANP)``. We strip the trailing suffix.
    """
    m = re.search(r"<title>([^<]+)</title>", html, re.I)
    if not m:
        return None
    t = html_module.unescape(m.group(1)).strip()
    t = re.sub(
        r"\s*-\s*National Association of Nutrition Professionals.*$",
        "",
        t,
    ).strip()
    return t or None


_EMPLOYER_TD_RE = re.compile(r'id="tdEmployerName"[^>]*>(.*?)</td>', re.S | re.I)
_WORKPHONE_TD_RE = re.compile(r'id="tdWorkPhone"[^>]*>(.*?)</td>', re.S | re.I)
_HOMEPHONE_TD_RE = re.compile(r'id="tdHomePhone"[^>]*>(.*?)</td>', re.S | re.I)
_CUSTOM_FIELD_RE = re.compile(
    r'class="CstmFldLbl"[^>]*>([^<:]+?)\s*:\s*</label>'
    r'[^<]*</td>\s*<td[^>]*class="CstmFldVal"[^>]*>(.*?)</td>',
    re.S,
)
# Catches any non-empty mailto: anchor in the page body. We filter out
# the NANP site-footer ``info@nanp.org`` link and the JS-template
# ``'+ strEmail + '`` placeholders in the consumer.
_MAILTO_ANCHOR_RE = re.compile(
    r'<a\s+href="mailto:([^"]+)"', re.I
)
_EMAIL_SHAPE_RE = re.compile(
    r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$'
)


def _parse_employer_block(td_html: str) -> dict:
    """Pull (practice_name, address1, city, state, postal, country) out of
    the ``tdEmployerName`` cell.

    NANP differs from AANP here: AANP renders the practice name as the
    first anchor with a ``txt_employName=`` href; NANP NEVER uses that
    anchor. Instead, the first plain-text fragment INSIDE the cell is
    the practice name (if it doesn't start with a digit and doesn't
    match a postal pattern). Numeric-leading fragments are treated as
    street lines (the canonical case for Charity Allen's profile, where
    the cell opens with ``424 Breckenridge Way``).

    Subsequent anchors carry the city / state / country (each pointed at
    ``/search/search.asp?txt_city=...`` etc).
    """
    out: dict = {}
    if not td_html:
        return out

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
    seen_first_geo_anchor = False

    for kind, val, href in pieces:
        if val == "Map":
            continue
        if kind == "a":
            href_l = (href or "").lower()
            # AANP-style txt_employName= practice anchor (rare on NANP
            # but kept for robustness — sister deployments may use it).
            if "txt_employname=" in href_l:
                practice_name = val
            elif "txt_city=" in href_l:
                city = val
                seen_first_geo_anchor = True
            elif "txt_state=" in href_l:
                state = val
                seen_first_geo_anchor = True
            elif "txt_country=" in href_l:
                country = val
                seen_first_geo_anchor = True
        else:
            postal_token = val.replace(" ", "")
            if (
                re.match(r"^\d{5}(?:-\d{4})?$", val)
                or re.match(r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$", val, re.I)
                or re.match(r"^[A-Z]\d[A-Z]\d[A-Z]\d$", postal_token, re.I)
            ):
                postal = val
            elif val:
                # NANP-specific heuristic: text fragments before any geo
                # anchor that DON'T start with a digit are practice names
                # (e.g. "Certified Holistic Nutrition Consultant"). Text
                # fragments that DO start with a digit are street lines
                # ("424 Breckenridge Way"). After the first geo anchor,
                # everything else is street/extra address detail.
                if (
                    not seen_first_geo_anchor
                    and practice_name is None
                    and not re.match(r"^\d", val)
                ):
                    practice_name = val
                else:
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


def _parse_phone_block(td_html: str) -> dict:
    """Pull phone + website out of a phone-cell (tdWorkPhone or
    tdHomePhone).

    Structure varies: ``931 2205391 (Phone)`` for offices with a phone,
    or a ``Visit Website`` anchor (or both, or neither). We pick the
    first ``(Phone)``-tagged number and the first http anchor labeled
    ``Visit Website``. Same shape as AANP — NANP just sometimes puts the
    phone in ``tdHomePhone`` instead of (or in addition to) ``tdWorkPhone``.
    """
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
    """Pull the labeled custom fields (BCHN, CDSP, Area of Expertise, ...)
    out of a member profile page."""
    out: dict = {}
    for m in _CUSTOM_FIELD_RE.finditer(html):
        label_raw = m.group(1).strip()
        # Normalize the label (strip ® / ™ / &reg; / &trade;) so callers
        # can look up by 'BCHN' / 'CDSP' regardless of trademark glyph.
        label = _normalize_credential_chunk(label_raw)
        value_html = m.group(2)
        value = _strip_html_tags(value_html)
        if not value:
            continue
        out[label] = value
    return out


def _extract_practitioner_email(html: str) -> Optional[str]:
    """Pick the practitioner's email out of the page.

    NANP profiles render the practitioner email via a JS-decrypted
    ``mailto:`` anchor in the right-column header. The static HTML has
    BOTH the rendered ``<a href="mailto:real@addr">`` (when Playwright
    has run the JS) AND a JS-template placeholder ``<a href="mailto:'+
    strEmail + '">``. We accept only mailto values that:

      - shape-match an email (single @, valid TLD), AND
      - are NOT the ``info@nanp.org`` site-footer link.
    """
    if not isinstance(html, str):
        return None
    for m in _MAILTO_ANCHOR_RE.finditer(html):
        raw = m.group(1).strip()
        if not raw or raw.lower() == "info@nanp.org":
            continue
        if not _EMAIL_SHAPE_RE.match(raw):
            continue
        return raw
    return None


def parse_profile_html(
    html: str, member_id: Optional[str] = None
) -> Optional[NormalizedPractitionerRow]:
    """Pure parser: takes a ``/members/?id=<id>`` detail page HTML and
    returns a fully-populated NormalizedPractitionerRow. Returns None
    if no usable name is found.

    The detail page carries: name (from <title>), optional practice name
    + address (from tdEmployerName), optional phone (from tdWorkPhone or
    tdHomePhone), website (from either phone cell's Visit-Website
    anchor), email (from the JS-decrypted mailto anchor in the right
    column), and credentials / BCHN / CDSP / etc (from the CstmFld
    custom-field table).

    ``member_id`` (optional) is used to construct ``source_url``. The live
    profile pages no longer carry a ``/members/?id=<n>`` anchor in the
    body, so the migrate runner must pass the id it used to fetch the
    page. If omitted we try to recover one from any in-page anchor.
    """
    if not isinstance(html, str):
        return None

    name_raw = _extract_title_name(html)
    if not name_raw:
        return None
    clean_name, title_creds = _strip_credentials(name_raw)
    if not clean_name:
        return None

    if member_id is None:
        id_m = re.search(r'/members/\?id=(\d+)', html)
        member_id = id_m.group(1) if id_m else None

    employer_m = _EMPLOYER_TD_RE.search(html)
    employer = _parse_employer_block(employer_m.group(1) if employer_m else "")

    # Phone + website come from EITHER tdWorkPhone OR tdHomePhone (NANP
    # sometimes only populates one). Work takes precedence when both
    # carry data; we merge website + phone with work winning.
    work_m = _WORKPHONE_TD_RE.search(html)
    work = _parse_phone_block(work_m.group(1) if work_m else "")
    home_m = _HOMEPHONE_TD_RE.search(html)
    home = _parse_phone_block(home_m.group(1) if home_m else "")
    phone_val = work.get("phone") or home.get("phone")
    website_val = work.get("website") or home.get("website")

    fields = _parse_custom_fields(html)

    # Credentials: prefer the explicit Credentials custom field (rare on
    # NANP — most members have no such row); fall back to the title's
    # post-nominals.
    credentials = _coerce_str(fields.get("Credentials")) or title_creds

    # Email: prefer the body-decrypted mailto; fall back to a Clinic
    # Email custom field if present.
    email = _extract_practitioner_email(html) or _coerce_str(fields.get("Clinic Email"))

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

    # Fellowship: precedence is (1) custom-field "BCHN" == "Yes", then
    # (2) BCHN token in credentials/title.
    bchn_field = _coerce_str(fields.get("BCHN"))
    bchn_yes_from_field = bchn_field is not None and bchn_field.lower() == "yes"
    fellowship_level = bchn_yes_from_field or _detect_fellowship_creds(credentials)

    return NormalizedPractitionerRow(
        tier="org_member",
        name=clean_name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="NANP",
        source_url=_build_source_url(member_id) if member_id else None,
        fellowship_level=fellowship_level,
        practice_name=practice_name,
        credentials=credentials,
        phone=_coerce_str(phone_val),
        email=email,
        website=_normalize_website(website_val),
        address1=_coerce_str(employer.get("address1")),
        city=_coerce_str(employer.get("city")),
        state=state,
        postal=_coerce_str(employer.get("postal")),
        country=country_iso,
    )


# ---------------------------------------------------------------------------
# Public orchestrator (used by run_all._run_nanp_scrape and migrate_nanp)
# ---------------------------------------------------------------------------

def fetch_all_records() -> list[NormalizedPractitionerRow]:
    """Walk the NANP iframe directory and enrich every stub with its
    profile page. Returns the merged NormalizedPractitionerRow list.

    Thin wrapper around the migrate_nanp helpers: keeps the public
    ``fetch_all_records`` import compatible with the
    ``run_all._run_nanp_scrape`` invocation pattern (every adapter in
    the cohort exposes this name).
    """
    # Import lazily so the pure parser module doesn't drag Playwright in
    # at import time (the test suite parses fixtures without a browser).
    from scrapers.practitioner_finder.migrate_nanp import (
        fetch_all_stubs,
        fetch_profile_html,
        _member_id_from_url,
        _merge_profile_into_stub,
    )
    from scrapers.practitioner_finder.playwright_fetch import playwright_session

    out: list[NormalizedPractitionerRow] = []
    with playwright_session() as fetcher:
        try:
            fetcher.get(DIRECTORY_FORM_URL)
        except Exception as e:  # pragma: no cover - live IO
            print(f"  WARN: NANP warm-up failed: {e}")

        try:
            stubs = fetch_all_stubs(fetcher=fetcher)
        except Exception as e:  # pragma: no cover - live IO
            print(f"  ERROR fetching NANP directory: {e}")
            stubs = []

        for stub in stubs:
            member_id = _member_id_from_url(stub.source_url)
            if member_id is None:
                out.append(stub)
                continue
            try:
                profile_html = fetch_profile_html(member_id, fetcher=fetcher)
            except Exception as e:  # pragma: no cover - live IO
                print(f"  WARN: NANP profile {member_id} fetch failed: {e}")
                out.append(stub)
                continue
            profile_row = parse_profile_html(profile_html, member_id=member_id)
            if profile_row is None:
                out.append(stub)
                continue
            out.append(_merge_profile_into_stub(stub, profile_row))
    return out
