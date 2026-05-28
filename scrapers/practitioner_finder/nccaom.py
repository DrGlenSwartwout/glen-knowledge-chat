"""National Board for Acupuncture and Herbal Medicine (NCBAHM) — formerly
NCCAOM — "Find a Practitioner" directory scraper.

Rebrand note (2026)
-------------------
The organization formerly known as the National Certification Commission
for Acupuncture and Oriental Medicine (NCCAOM) rebranded to the National
Board for Acupuncture and Herbal Medicine (NCBAHM). The directory now
lives at ``https://directory.ncbahm.org/`` (old ``directory.nccaom.org``
no longer resolves).

The diplomate cert codes were renamed in step with the brand change:

    Dipl. Ac.   (NCBAHM)   — Diplomate of Acupuncture
    Dipl. CH.   (NCBAHM)   — Diplomate of Chinese Herbology
    Dipl. OM.   (NCBAHM)   — Diplomate of Oriental Medicine
    Dipl. AHM   (NCBAHM)   — Diplomate of Acupuncture and Herbal Medicine
    Dipl. ABT   (NCBAHM)   — Diplomate of Asian Bodywork Therapy

The directory's cert-box still emits the short codes ``AC`` / ``CH`` /
``OM`` (and historically ``ABT``); we expand them into the canonical
``Dipl. <X> (NCBAHM)`` form. The newer ``AHM`` code is included in the
mapping defensively even though no fixture row carries it yet.

The ``source_org`` field is intentionally kept as ``"NCCAOM"`` (not
``"NCBAHM"``) — orchestrator, UI, GHL tags, and historical DB rows all
key off that string. We document the rebrand here and emit the new
``Dipl. ... (NCBAHM)`` credential strings; the ``source_org`` rename
is a separate (future) coordinated migration.

Live architecture (locked 2026-05-27)
-------------------------------------
- Real host: ``https://directory.ncbahm.org/``
- Search form: POSTs to ``/FAP/SearchPractitioners``. The POST is
  Cloudflare-Turnstile-gated and CANNOT be passed with headless
  Playwright — the migrate runner uses ``playwright_session(headless=False)``
  to drive a visible window through the Turnstile challenge.
- Stable per-state URL: the POST handler 302s to a GET URL we can replay
  directly (no token round-trip needed after the Turnstile cookie is in
  place):

    GET /FAP/SearchResultWithoutMap
        ?Radius=0
        &CountryCode=USA
        &StateCode=<XX>
        &SearchType=2
        &Latitude=0
        &Longitude=0
        &SortBy=DisplayName
        &SortDirection=DESC
        &SearchFormType=FAP
        &PageNo=<N>
        &PageSize=20

  ``PageSize=20`` is fixed by the back-end (no larger page renders).

- Pagination: bump ``PageNo`` 1..N. The hidden
  ``<input id="hdnlastpage" value="N">`` field on every response carries
  the last page number; the ``"<N> Practitioners found"`` banner gives
  the total record count.

- Card structure (one practitioner = one ``<div class="result-card__item">``):

    <div class="result-card__item">
      <div class="info-box">
        <p class="name">
          <a href="/FAP/PractitionerDetail?AgencyClientId=<b64id>="
             title="View the profile of Aaron Bullington"> Aaron Bullington</a>
        </p>
        <p class="gendar">Certified Diplomate | Male</p>
        <div class="iconic-callout">
          <div class="iconic-callout__item">
            <i class="icon-call-end"></i>
            <p class="copy"><em>808-934-9858</em></p>           # phone
          </div>
          <div class="iconic-callout__item">
            <i class="icon-globe"></i>
            <p class="copy">
              <span>www.example.com</span>
              <a href="http://www.example.com" title="...">Visit ...</a>
            </p>                                                # website
          </div>
          <div class="iconic-callout__item">
            <i class="icon-location-pin"></i>
            <p class="copy">
              <em id="addressdata_N">82 Keaa St , Hilo, HI, USA</em>
            </p>                                                # address
          </div>
        </div>
      </div>
      <div class="cert-box">
        <div class="cerfapfap-init slick-initialized slick-slider ...">
          <div class="slick-list ..."><div class="slick-track ...">
            <div class="cert-box__item slick-slide ...">
              <div class="content"><p class="copy">AC Certification</p></div>
              ...
            </div>
            <div class="cert-box__item ...">
              <div class="content"><p class="copy">CH Certification</p></div>
              ...
            </div>
          </div></div>
        </div>
      </div>
    </div>

  The cert-box may carry 1..N badges per practitioner (Adam J. French
  L.Ac. on the NY page-1 fixture has AC + CH + OM all three). Card
  fields phone / website / address all use the same ``Not Available``
  sentinel when the practitioner didn't supply a value.

Fellowship rule
---------------
Every practitioner in the NCBAHM public directory is a board-certified
diplomate by definition (NCBAHM is the credentialing body — the only
people listed hold the Dipl. <X> designation). So the default is
``fellowship_level=True``.

Exception: the per-card ``<p class="gendar">"<Status> | <Gender>"``
header can carry override states like ``Expired``, ``Inactive``,
``Retired``, ``Recertification Pending``, ``Suspended``, ``Revoked``.
We detect those and downgrade to False. The production fixtures only
contain ``Certified Diplomate`` (the public directory filters
inactive entries by default), so the downgrade branch is exercised via
synthetic fixtures in the test suite.

Source URL
----------
The per-practitioner ``source_url`` is the canonical detail-page URL:

    https://directory.ncbahm.org/FAP/PractitionerDetail?AgencyClientId=<b64id>=

The AgencyClientId is an opaque URL-safe base64 string emitted by the
back-end and is stable across re-runs — it's the dedup key for upsert.
Cards on the result page link to this URL natively.

Notes
-----
- The parser is pure-functional and decoupled from fetch. The live
  fetch (Cloudflare Turnstile + non-headless Playwright) lives in
  ``migrate_nccaom.py``.
- ``PageSize`` is a back-end constant of 20 regardless of any
  ``PageSize=`` param hint we send (the server ignores it).
- Pagination is 1-indexed; ``hdnlastpage`` carries the last page number.
"""
import html as html_module
import re
from typing import Optional

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


BASE = "https://directory.ncbahm.org"
SEARCH_URL = f"{BASE}/FAP/SearchResultWithoutMap"
DETAIL_URL = f"{BASE}/FAP/PractitionerDetail"

LOCKED_SPECIALTIES = ["acupuncture_tcm", "holistic_health"]

# Cert-code -> human credential. The list-view cert-box still emits the
# short codes; we expand them to the canonical Dipl. <X> (NCBAHM) form.
# AHM is included defensively for future listings (no fixture row carries
# the new code yet).
_CERT_CODE_TO_CREDENTIAL = {
    "AC":  "Dipl. Ac. (NCBAHM)",
    "CH":  "Dipl. CH. (NCBAHM)",
    "OM":  "Dipl. OM. (NCBAHM)",
    "AHM": "Dipl. AHM (NCBAHM)",
    "ABT": "Dipl. ABT (NCBAHM)",
}

# Status tokens (case-insensitive) that DOWNGRADE fellowship_level to
# False. NCBAHM production fixtures only carry "Certified Diplomate",
# but the parser checks defensively for these future-proof variants.
_INACTIVE_STATUS_TOKENS = {
    "EXPIRED",
    "INACTIVE",
    "RETIRED",
    "RECERTIFICATION PENDING",
    "SUSPENDED",
    "REVOKED",
}

# US state two-letter codes the CountryCode=USA dropdown exposes. NCBAHM
# uses the StateCode dropdown for US states only (50 + DC + the five
# inhabited territories: PR, GU, VI, AS, MP). Order is stable for
# reproducible run logs.
US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "PR", "GU", "VI", "AS", "MP",
]

# Country-name -> ISO2 (same convention as iabdm.py / iaomt.py / aanp.py).
# NCBAHM's directory is dominantly US — international entries are a small
# fringe set, and the rebrand-era directory exposes only the US dropdown.
# Kept here for the address parser to ISO2-normalize the country token.
_COUNTRY_NAME_TO_ISO2 = {
    "usa": "US",
    "us": "US",
    "united states": "US",
    "united states of america": "US",
    "canada": "CA",
    "ca": "CA",
    "australia": "AU",
    "new zealand": "NZ",
    "united kingdom": "GB",
    "uk": "GB",
    "germany": "DE",
    "japan": "JP",
    "china": "CN",
    "korea, republic of": "KR",
    "south korea": "KR",
    "korea": "KR",
    "hong kong": "HK",
    "taiwan": "TW",
    "singapore": "SG",
    "mexico": "MX",
    "israel": "IL",
    "switzerland": "CH",
    "netherlands": "NL",
    "ireland": "IE",
    "italy": "IT",
    "france": "FR",
    "spain": "ES",
    "puerto rico": "PR",
    "india": "IN",
    "philippines": "PH",
    "thailand": "TH",
    "vietnam": "VN",
    "malaysia": "MY",
    "indonesia": "ID",
    "brazil": "BR",
    "argentina": "AR",
    "chile": "CL",
    "south africa": "ZA",
    "united arab emirates": "AE",
    "uae": "AE",
}


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Stripped string or None for missing/empty values."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().replace("\xa0", " ").strip()
        return s or None
    s = str(val).strip()
    return s or None


def _strip_html_tags(s: str) -> str:
    """Drop all HTML tags from a snippet; collapse whitespace.

    Block-level tags become a single space so adjacent block contents
    don't smash together; inline tags are removed cleanly. ``&amp;`` /
    ``&#160;`` and friends are unescaped."""
    if not s:
        return ""
    out = re.sub(r"<(br|p|div|td|tr|li|ul|ol|h\d|section|em|span)[^>]*>", " ", s, flags=re.I)
    out = re.sub(r"</(br|p|div|td|tr|li|ul|ol|h\d|section|em|span)>", " ", out, flags=re.I)
    out = re.sub(r"<[^>]+>", "", out)
    out = html_module.unescape(out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _strip_credentials(name: str) -> tuple[str, Optional[str]]:
    """Split 'Youl Park L.Ac.' / 'Jane Doe, L.Ac., DAOM' into
    (clean_name, credentials).

    Mirrors the IABDM / AANP / OVDR helpers. The NCBAHM name field
    occasionally includes the practitioner's degree (``L.Ac.``,
    ``DAOM``, ``DACM``) tacked onto the end with or without a comma —
    we accept both ``Name, Cred`` and ``Name Cred`` forms when the
    trailing token matches a known credential pattern."""
    if not name:
        return "", None
    s = name.strip()

    paren = re.match(r"^(.*?)\s*\(([A-Za-z][A-Za-z.,\s/-]*)\)\s*$", s)
    if paren:
        s = f"{paren.group(1).strip()}, {paren.group(2).strip()}"

    # Trailing degree credentials (L.Ac., LAc, DAOM, DACM, PhD, MS, MD, ND, OMD)
    # may be attached without a comma. Detect a trailing token of the
    # form ``<dot-letters>`` at the end of the name and split on it.
    trail = re.match(
        r"^(.*?)\s+("
        r"L\.?Ac\.?|DAOM|DACM|OMD|DAc|PhD|MD|ND|DC|LMT|MAOM|MSAOM|MSTCM|MSOM"
        r")\.?$",
        s,
    )
    if trail:
        clean = trail.group(1).strip().rstrip(",").strip()
        creds = trail.group(2).strip()
        return clean or s, creds or None

    cred_pat = re.compile(r",\s*([A-Za-z][A-Za-z./]*[A-Za-z])")
    m = cred_pat.search(s)
    if not m:
        return s.rstrip(", "), None
    clean = s[: m.start()].strip().rstrip(",")
    creds = s[m.start():].lstrip(", ").strip().rstrip(",").rstrip()
    return clean, creds or None


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Add a scheme to a bare domain; reject mailto/javascript/anchors."""
    s = _coerce_str(raw)
    if not s:
        return None
    if s.lower() in {"not available", "n/a", "none"}:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    if s.startswith("mailto:") or s.startswith("javascript:") or s.startswith("#"):
        return None
    return f"https://{s}"


def _normalize_phone(raw: Optional[str]) -> Optional[str]:
    """Trim whitespace and reject 'Not Available' / empty sentinels."""
    s = _coerce_str(raw)
    if not s:
        return None
    if s.lower() in {"not available", "n/a", "none"}:
        return None
    return s


def _country_iso2(raw: Optional[str]) -> Optional[str]:
    """Map a free-text country to ISO2; falls back to the input
    truncated to two characters when it looks like a USPS-style code
    (USA -> US, US -> US, GB -> GB)."""
    s = _coerce_str(raw)
    if not s:
        return None
    key = s.lower()
    if key in _COUNTRY_NAME_TO_ISO2:
        return _COUNTRY_NAME_TO_ISO2[key]
    # Already a 2-letter ISO code?
    if re.fullmatch(r"[A-Za-z]{2}", s):
        return s.upper()
    return None


def _extract_agency_client_id(href: str) -> Optional[str]:
    """Pull the opaque AgencyClientId out of any practitioner link.

    The list page uses ``/FAP/PractitionerDetail?AgencyClientId=<b64>=``
    while the older name-search results used
    ``/FAPPractitionerProfile/<b64>=``; both encode the same opaque key.
    The trailing ``=`` is part of the URL-safe base64 padding and is
    preserved verbatim — re-runs must yield identical IDs for stable
    dedup."""
    if not href:
        return None
    # AgencyClientId query-param form.
    m = re.search(r"[?&]AgencyClientId=([^&\"'#]+)", href)
    if m:
        return m.group(1)
    # /FAPPractitionerProfile/<id> path-style form (legacy).
    m = re.search(r"/FAPPractitionerProfile/([^?&\"'#/]+)", href)
    if m:
        return m.group(1)
    return None


def _build_source_url(agency_client_id: Optional[str]) -> Optional[str]:
    """Canonical per-practitioner URL — bare AgencyClientId form.

    The list-page link form ``/FAP/PractitionerDetail?AgencyClientId=<id>``
    is the canonical source_url. Both list-page and legacy name-search
    pages resolve to this profile."""
    if not agency_client_id:
        return None
    return f"{DETAIL_URL}?AgencyClientId={agency_client_id}"


# Address-line regex: NCBAHM renders addresses as a single comma-joined
# line, e.g.:
#     "513 N Morrison Rd , Vancouver, WA, USA"
#     "1301 Spring St #7F , Seattle, WA, USA"
#     "143 Piccadilly Downs , Lynbrook, NY 11563-3117, USA"
# Last comma-separated token is the country; second-to-last is the
# state (which may include a zip after the state code).
_US_POSTAL_TAIL_RE = re.compile(r"\b(\d{5}(?:-\d{4})?)\s*$")
_CA_POSTAL_TAIL_RE = re.compile(r"\b([A-Z]\d[A-Z])\s*(\d[A-Z]\d)\s*$")


def _parse_address_line(line: str) -> dict:
    """Parse 'street, city, state[ postal], country' into the address fields.

    Returns a dict with keys address1 / city / state / postal / country.
    Any field may be absent (returned as None / missing key).

    Defensive: returns empty dict for the literal sentinel
    ``"Not Available"`` (frequent in the HI page-1 fixture for solo
    practitioners who didn't supply a street address)."""
    out: dict = {}
    if not line:
        return out
    s = re.sub(r"\s+", " ", line.strip()).strip().rstrip(",")
    if not s:
        return out
    if s.lower() in {"not available", "n/a", "none"}:
        return out

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return out

    # Last token is country.
    country = None
    if len(parts) >= 2:
        country = parts[-1]
        parts = parts[:-1]
    else:
        # No country supplied — leave the whole line as address1.
        out["address1"] = s
        return out

    # The middle/last token may be 'STATE POSTAL' (US) or 'PROVINCE POSTAL'
    # (Canada). Strip the postal off if present, then the remaining word(s)
    # are the state/province.
    state = None
    postal = None
    if parts:
        last = parts[-1]
        m_us = _US_POSTAL_TAIL_RE.search(last)
        m_ca = _CA_POSTAL_TAIL_RE.search(last)
        if m_ca:
            postal = f"{m_ca.group(1)} {m_ca.group(2)}"
            state = last[: m_ca.start()].strip() or None
        elif m_us:
            postal = m_us.group(1)
            state = last[: m_us.start()].strip() or None
        else:
            state = last or None
        parts = parts[:-1]

    # Remaining: street(s) then city. If exactly one part remains, treat
    # it as the city (the directory routinely omits a separate street
    # line for solo practitioners).
    city = None
    address1 = None
    if len(parts) == 1:
        city = parts[0]
    elif len(parts) >= 2:
        city = parts[-1]
        address1 = ", ".join(parts[:-1])

    if address1:
        out["address1"] = address1
    if city:
        out["city"] = city
    if state:
        out["state"] = state
    if postal:
        out["postal"] = postal
    if country:
        out["country"] = country
    return out


def _detect_inactive_status(status_text: Optional[str]) -> bool:
    """True when the 'gendar' status line contains an inactive marker.

    NCBAHM renders the practitioner status as ``"<Status> | <Gender>"``
    in the ``<p class="gendar">`` block. Current production exclusively
    shows ``Certified Diplomate``, but the parser checks defensively
    for Expired / Inactive / Retired / Recertification Pending /
    Suspended / Revoked."""
    s = _coerce_str(status_text)
    if not s:
        return False
    # Status is the segment before the first " | ".
    head = s.split("|", 1)[0].strip().upper()
    if head in _INACTIVE_STATUS_TOKENS:
        return True
    # Allow inactive-token substring match too (e.g. "EXPIRED CERTIFICATION").
    for token in _INACTIVE_STATUS_TOKENS:
        if token in head:
            return True
    return False


# ---------------------------------------------------------------------------
# List-grid card parser
# ---------------------------------------------------------------------------

# Match the practitioner-card anchor in both list-view layouts.
# Production list-page form: <a href="/FAP/PractitionerDetail?AgencyClientId=...=" title="...">Name</a>
# Legacy name-search form:   <a href="/FAPPractitionerProfile/...=" title="...">Name</a>
_NAME_ANCHOR_RE = re.compile(
    r'<a\s+href="((?:/FAP/PractitionerDetail\?AgencyClientId=|/FAPPractitionerProfile/)[^"]+)"'
    r'[^>]*>(.*?)</a>',
    re.S | re.I,
)

# Cert-code extractor. The live HTML wraps each cert-box__item in
# ``class="cert-box__item slick-slide ..."`` — i.e. the class attribute
# carries additional tokens after the cert-box__item class. We anchor on
# the <p class="copy">XX Certification</p> marker inside the cert-box
# slick-track instead of trying to thread the class-token regex, which is
# both simpler and more robust to slick-slider re-shuffling.
#
# We narrow the search to AFTER the first ``<div class="cert-box"`` marker
# so the practitioner's name / status text can't accidentally match
# (defensive — none of the production fixtures actually contain a stray
# "XX Certification" string outside the cert-box, but a future
# practitioner with that phrase in their name would otherwise pollute the
# cert-codes list).
_CERT_BOX_RE = re.compile(
    r'<div[^>]*class="[^"]*\bcert-box\b[^"]*"[^>]*>(.*)$',
    re.S | re.I,
)
_CERT_CODE_INSIDE_BOX_RE = re.compile(
    r'<p[^>]*class="copy"[^>]*>\s*([A-Za-z]+)\s+Certification',
    re.S | re.I,
)


def _parse_card(card_html: str) -> Optional[dict]:
    """Pull one practitioner-card's data out of a result-card HTML chunk.

    Returns a dict with keys ``href``, ``name``, ``status``, ``phone``,
    ``website``, ``address``, ``cert_codes`` — or None if no anchor was
    found in the chunk."""
    if not card_html:
        return None

    m = _NAME_ANCHOR_RE.search(card_html)
    if not m:
        return None
    href = m.group(1)
    name = _strip_html_tags(m.group(2))
    if not name:
        return None

    out: dict = {"href": href, "name": name}

    # Status from <p class="gendar">...</p> (free text before "|").
    g = re.search(r'<p\s+class="gendar"[^>]*>(.*?)</p>', card_html, re.S | re.I)
    if g:
        out["status"] = _strip_html_tags(g.group(1))

    # Address: <em id="addressdata_N">...</em> (the canonical id-anchored
    # form on the live list page). Fall back to any <em> following an
    # icon-location-pin marker for forward-compat with the legacy
    # name-search layout.
    a = re.search(
        r'<em[^>]*id="addressdata_\d+"[^>]*>(.*?)</em>',
        card_html,
        re.S | re.I,
    )
    if not a:
        a = re.search(
            r'icon-location-pin[^<]*</i>\s*<p[^>]*class="copy"[^>]*>\s*<em[^>]*>(.*?)</em>',
            card_html,
            re.S | re.I,
        )
    if not a:
        a = re.search(
            r'icon-location-pin[^<]*</i>\s*</span>\s*<em[^>]*>(.*?)</em>',
            card_html,
            re.S | re.I,
        )
    if a:
        out["address"] = _strip_html_tags(a.group(1))

    # Phone: <i class="icon-call-end"></i> ... <em>...</em>.
    p = re.search(
        r'icon-call-end[^<]*</i>\s*<p[^>]*class="copy"[^>]*>\s*<em[^>]*>(.*?)</em>',
        card_html,
        re.S | re.I,
    )
    if not p:
        p = re.search(
            r'icon-call-end[^<]*</i>\s*</span>\s*<em[^>]*>(.*?)</em>',
            card_html,
            re.S | re.I,
        )
    if p:
        out["phone"] = _strip_html_tags(p.group(1))

    # Website: the globe block carries BOTH a truncated display <span>
    # AND a full anchor with the href. Take the anchor href when
    # present; fall back to the <em> text (which may be "Not Available")
    # otherwise. Cards without a website link omit the <a> entirely.
    w = re.search(
        r'icon-globe[^<]*</i>\s*<p[^>]*class="copy"[^>]*>(.*?)</p>',
        card_html,
        re.S | re.I,
    )
    website = None
    if w:
        chunk = w.group(1)
        href_m = re.search(r'<a[^>]+href="([^"]+)"', chunk, re.I)
        if href_m:
            website = href_m.group(1)
    if website:
        out["website"] = website

    # Cert codes: confine the search to AFTER the first <div class="cert-box"
    # marker so the card-header status / name / address can't pollute the
    # cert-codes list.
    cert_codes: list[str] = []
    box_m = _CERT_BOX_RE.search(card_html)
    if box_m:
        for code_m in _CERT_CODE_INSIDE_BOX_RE.finditer(box_m.group(1)):
            cert_codes.append(code_m.group(1).upper())
    out["cert_codes"] = cert_codes

    return out


def _card_to_row(card: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: parsed card dict -> NormalizedPractitionerRow.

    Returns None when the card has no extractable AgencyClientId
    (without it, the source_url isn't stable and re-runs would dup)."""
    name_raw = card.get("name")
    if not name_raw:
        return None
    name, name_creds = _strip_credentials(name_raw)
    if not name:
        return None

    agency_id = _extract_agency_client_id(card.get("href") or "")
    source_url = _build_source_url(agency_id)
    if not source_url:
        return None

    # Build credentials: cert-code expansions + any trailing degree
    # pulled off the name. Dedupe preserving first-seen order so the
    # credentials string reads as "Dipl. Ac. (NCBAHM), Dipl. CH. (NCBAHM), L.Ac.".
    cred_parts: list[str] = []
    for code in card.get("cert_codes", []) or []:
        cred = _CERT_CODE_TO_CREDENTIAL.get(code)
        if cred and cred not in cred_parts:
            cred_parts.append(cred)
    if name_creds:
        for chunk in re.split(r",\s*", name_creds):
            chunk = chunk.strip()
            if chunk and chunk not in cred_parts:
                cred_parts.append(chunk)
    credentials = ", ".join(cred_parts) if cred_parts else None

    address_fields = _parse_address_line(card.get("address") or "")
    country_raw = address_fields.get("country")
    country_iso = _country_iso2(country_raw) or country_raw or "US"

    inactive = _detect_inactive_status(card.get("status"))

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        # source_org intentionally kept as the historical brand string
        # (NCCAOM); see module docstring for the rebrand decision.
        source_org="NCCAOM",
        source_url=source_url,
        # Every listed NCBAHM diplomate is board-certified by definition;
        # the only exception is the explicit inactive-status downgrade.
        fellowship_level=not inactive,
        practice_name=None,  # not in list-grid cards
        credentials=credentials,
        phone=_normalize_phone(card.get("phone")),
        email=None,           # not in list-grid cards
        website=_normalize_website(card.get("website")),
        address1=_coerce_str(address_fields.get("address1")),
        city=_coerce_str(address_fields.get("city")),
        state=_coerce_str(address_fields.get("state")),
        postal=_coerce_str(address_fields.get("postal")),
        country=country_iso,
    )


# ---------------------------------------------------------------------------
# Page-scoped public parsers
# ---------------------------------------------------------------------------

def parse_search_html(html: str) -> list[NormalizedPractitionerRow]:
    """Pure parser: takes a ``/FAP/SearchResultWithoutMap`` response HTML
    and returns one NormalizedPractitionerRow per practitioner card on
    the page. No I/O.

    Handles the production list layout
    (``<div class="result-card__item">``) and the legacy name-search
    layout (``<div class="citySearchList__content">``) — both feed
    through the same per-card parser. Pagination is the caller's
    responsibility — this parser is page-scoped.
    """
    if not isinstance(html, str):
        return []

    rows: list[NormalizedPractitionerRow] = []
    seen_urls: set[str] = set()

    def _walk(marker: str) -> None:
        for chunk in re.split(r'<div\s+class="' + re.escape(marker) + r'"', html)[1:]:
            card_html = '<div class="' + marker + '"' + chunk
            nxt = card_html.find('<div class="' + marker + '"', 1)
            if nxt > 0:
                card_html = card_html[:nxt]
            parsed = _parse_card(card_html)
            if not parsed:
                continue
            row = _card_to_row(parsed)
            if row is None:
                continue
            if row.source_url in seen_urls:
                continue
            seen_urls.add(row.source_url)
            rows.append(row)

    _walk("result-card__item")
    _walk("citySearchList__content")

    return rows


def parse_total_pages(html: str) -> int:
    """Extract the total page count from the hidden ``hdnlastpage`` input.

    Returns 0 when the field is absent (defensive: callers should treat 0
    as "no pagination available — use the empty-page break instead")."""
    if not isinstance(html, str):
        return 0
    m = re.search(r'id="hdnlastpage"\s+value="(\d+)"', html)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return 0


def parse_total_count(html: str) -> Optional[int]:
    """Extract the "N Practitioners found" total from the result banner.

    Returns None if the banner isn't present (e.g. error page or
    fragmented response)."""
    if not isinstance(html, str):
        return None
    m = re.search(r'>(\d+)\s+Practitioners\s+found', html, re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None
