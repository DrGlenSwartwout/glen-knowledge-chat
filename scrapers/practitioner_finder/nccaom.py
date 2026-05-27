"""National Certification Commission for Acupuncture and Oriental Medicine
(NCCAOM) "Find a Practitioner" directory scraper.

NCCAOM is the credentialing body for acupuncture and Oriental medicine
practitioners in the US — every diplomate listed in the public directory
holds at least one of:

    Dipl. Ac.   (NCCAOM)   — Diplomate of Acupuncture
    Dipl. C.H.  (NCCAOM)   — Diplomate of Chinese Herbology
    Dipl. O.M.  (NCCAOM)   — Diplomate of Oriental Medicine
    Dipl. ABT   (NCCAOM)   — Diplomate of Asian Bodywork Therapy

Discovery 2026-05-27:

The public web UI ``https://www.nccaom.org/find-a-practitioner/`` is
behind Cloudflare bot mitigation, but the actual practitioner search
tool lives on a separate ASP.NET MVC subdomain at
``https://directory.nccaom.org/`` (also gated by Cloudflare in production
— a real run will need a recycled browser session or Playwright
fallback; the fixture-driven parser doesn't care). The search form
``<form action="/FAP/SearchPractitioners" method="post">`` accepts three
SearchType modes:

    SearchType=1  — Search by Radius   (Radius + PinCode/Latitude/Longitude)
    SearchType=2  — Search by City, State, Country (CountryCode + StateCode + CityName)
    SearchType=3  — Search by Practitioner Name (FirstName + LastName)

The POST handler 302-redirects to a GET endpoint that renders the
results page directly — the GET URL works for re-issuing the same query
without going back through the POST + CSRF token round-trip:

    GET /FAP/SearchResultWithoutMap
        ?Radius=0
        &CountryCode=USA
        &StateCode=WA
        &SearchType=2
        &Latitude=0
        &Longitude=0
        &SortBy=DisplayName
        &SortDirection=ASC
        &SearchFormType=FAP
        &PageNo=1

This is the cleanest full-coverage approach: walk every US state +
every non-US country in the CountryCode dropdown. The page renders
20 practitioners per page; the total page count is in the hidden
``<input id="hdnlastpage" value="N" />`` field — walk PageNo=1..N.

Card markup (one practitioner = one ``<div class="result-card__item">``):

    <div class="result-card__item">
      <div class="info-box">
        <p class="name"><a href="/FAP/PractitionerDetail?AgencyClientId=<b64id>="
                          title="View the profile of Zhenbo Li"> Zhenbo Li</a></p>
        <p class="gendar">Certified Diplomate | Female</p>
        <div class="iconic-callout">
          <div class="iconic-callout__item">
            <i class="icon-call-end"></i>
            <p class="copy"><em>360-984-6489</em></p>          # phone
          </div>
          <div class="iconic-callout__item">
            <i class="icon-globe"></i>
            <p class="copy">
              <span>www.example.com ...</span>
              <a href="http://www.example.com">Visit ...</a>   # website (or "Not Available")
            </p>
          </div>
          <div class="iconic-callout__item">
            <i class="icon-location-pin"></i>
            <p class="copy">
              <em id="addressdata_N">513 N Morrison Rd , Vancouver, WA, USA</em>
            </p>
          </div>
        </div>
      </div>
      <div class="cert-box">
        <div class="cert-box__item">
          <div class="content"><p class="copy">AC Certification</p></div>
        </div>
        <div class="cert-box__item">
          <div class="content"><p class="copy">CH Certification</p></div>
        </div>
        ...
      </div>
    </div>

The cert-box codes map to diplomate credentials:
    AC -> Dipl. Ac. (NCCAOM)
    CH -> Dipl. C.H. (NCCAOM)
    OM -> Dipl. O.M. (NCCAOM)
    ABT -> Dipl. ABT (NCCAOM)
We combine the codes (e.g. "Dipl. Ac., Dipl. C.H.") into the credentials
string; the practitioner's degree credentials (L.Ac., DAOM, etc.) are
only present in the displayed ``name`` field when the practitioner added
them voluntarily, in which case we lift them via the standard
``_strip_credentials`` helper.

There's also a Name-Search result page (``SearchType=3``) with a
different layout — ``<section class="fap--citySearchList">`` containing
``<div class="citySearchList__content">`` cards. That layout is
supported by the same parser via secondary card-locator markers; the
name-search page's per-row anchor is
``/FAPPractitionerProfile/<b64id>=`` (just the relative variant of
``/FAP/PractitionerDetail?AgencyClientId=<b64id>=`` — both resolve to
the same profile and share the same opaque AgencyClientId).

Fellowship rule
---------------
**Every NCCAOM-listed practitioner is board-certified by definition.**
NCCAOM is a credentialing body — the only people in this directory hold
the Dipl. Ac. / Dipl. C.H. / Dipl. O.M. / Dipl. ABT designation. So the
default is ``fellowship_level=True`` for any row produced from the
public directory.

Exception: the per-card status text (rendered in the ``<p class="gendar">``
line as ``"<Status> | <Gender>"``) can carry override states like
``Expired``, ``Inactive``, ``Retired``, or ``Recertification Pending``.
We detect those and downgrade fellowship_level to False. The production
fixtures captured 2026-05-27 only contain ``Certified Diplomate`` (the
public directory filters expired/inactive entries by default), so the
override branch is exercised only via synthetic fixtures in the test
suite — but it's in place for any future hidden-status listings that
slip through.

The per-practitioner ``source_url`` is the canonical detail-page URL:

    https://directory.nccaom.org/FAP/PractitionerDetail?AgencyClientId=<b64id>=

The AgencyClientId is an opaque URL-safe base64 string stable across
re-runs — it's the dedup key for upsert.

Notes / surprises
-----------------
- Cloudflare gating: ``directory.nccaom.org`` serves a JS challenge to
  static-UA curl in production. A live scrape will need either a recycled
  browser session, a Playwright fallback, or a residential-proxy
  pass-through. The parser is fully decoupled from fetch — the migration
  script can swap fetch backends without parser changes.
- Voluntary directory: the on-page disclaimer reads "this directory is
  voluntary, not all certified Diplomates will be listed." Total
  certified Diplomate count is ~25,000+ (NCCAOM public reporting), but
  only the opted-in subset is visible — closer to 15,000-20,000 based
  on per-state spot-counts (WA alone has 789).
- Page size: 20 per page, regardless of any per_page hint. The result-info
  banner reports total count; the hidden ``hdnlastpage`` carries total
  pages. Pagination is 1-indexed.
"""
import html as html_module
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://directory.nccaom.org"
SEARCH_URL = f"{BASE}/FAP/SearchResultWithoutMap"
DETAIL_URL = f"{BASE}/FAP/PractitionerDetail"

LOCKED_SPECIALTIES = ["acupuncture_tcm", "holistic_health"]

# Cert-code -> human credential. The NCCAOM directory list page emits
# short codes (AC/CH/OM/ABT) in the cert-box; we expand them into the
# canonical Dipl. <X> (NCCAOM) form, dropping the trademark suffix to
# keep the credentials string compact and unambiguous.
_CERT_CODE_TO_CREDENTIAL = {
    "AC": "Dipl. Ac. (NCCAOM)",
    "CH": "Dipl. C.H. (NCCAOM)",
    "OM": "Dipl. O.M. (NCCAOM)",
    "ABT": "Dipl. ABT (NCCAOM)",
}

# Status flags that DOWNGRADE fellowship_level to False. NCCAOM itself
# isn't the source of these in the current production fixtures (the
# public directory pre-filters to Certified Diplomate only), but the
# parser checks the status text defensively in case any of these slip
# through to the rendered list page in the future.
_INACTIVE_STATUS_TOKENS = {
    "EXPIRED",
    "INACTIVE",
    "RETIRED",
    "RECERTIFICATION PENDING",
    "SUSPENDED",
    "REVOKED",
}

# US state two-letter codes the CountryCode=USA dropdown exposes. NCCAOM
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

# Non-US countries NCCAOM's CountryCode dropdown exposes. NCCAOM mostly
# lists US practitioners — the international set is small. We walk the
# countries known to actually have diplomates in the directory; the rest
# of the dropdown's full ~250-country list returns empty pages on every
# query. Keeping this list conservative reduces unnecessary HTTP load
# at scrape time.
NON_US_COUNTRIES = [
    "CANADA",
    "AUSTRALIA",
    "NEW ZEALAND",
    "UNITED KINGDOM",
    "GERMANY",
    "JAPAN",
    "CHINA",
    "KOREA, REPUBLIC OF",
    "HONG KONG",
    "TAIWAN",
    "SINGAPORE",
    "MEXICO",
    "ISRAEL",
    "SWITZERLAND",
    "NETHERLANDS",
    "IRELAND",
    "ITALY",
    "FRANCE",
    "SPAIN",
    "PUERTO RICO",
]

# Country-name -> ISO2 (same convention as iabdm.py / iaomt.py / aanp.py).
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
# Stage 1: paginated search HTML fetch (live)
# ---------------------------------------------------------------------------

def fetch_search_page(
    *,
    country: str = "USA",
    state: Optional[str] = None,
    city: Optional[str] = None,
    page: int = 1,
) -> str:
    """Fetch one page of ``/FAP/SearchResultWithoutMap`` for the given
    (country, [state], [city], page) tuple.

    Static UA + 20s timeout + 0.5s sleep (rate-friendly, single-threaded).
    Retries once on connection error before re-raising. Returns the raw
    HTML body — caller parses it with ``parse_search_html``.

    Note: ``directory.nccaom.org`` is Cloudflare-protected. Static-UA
    requests can return HTTP 403 ("Just a moment..." challenge page).
    Production runs should wrap this in a Playwright session or recycle
    a browser cookie jar.
    """
    params = {
        "Radius": "0",
        "CountryCode": country,
        "SearchType": "2",
        "Latitude": "0",
        "Longitude": "0",
        "SortBy": "DisplayName",
        "SortDirection": "ASC",
        "SearchFormType": "FAP",
        "PageNo": str(page),
    }
    if state:
        params["StateCode"] = state
    if city:
        params["CityName"] = city
    s = _session()
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            r = s.get(SEARCH_URL, params=params, timeout=20)
            r.raise_for_status()
            time.sleep(0.5)
            return r.text
        except requests.RequestException as e:
            last_exc = e
            time.sleep(1.0)
            continue
    # Both attempts failed — re-raise the last error.
    if last_exc is not None:
        raise last_exc
    return ""  # unreachable; satisfies type checker


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

    Mirrors the IABDM / AANP / OVDR helpers. The NCCAOM name field
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
    while the name-search results use ``/FAPPractitionerProfile/<b64>=``;
    both encode the same opaque key. The trailing ``=`` is part of the
    URL-safe base64 padding and is preserved verbatim — re-runs must
    yield identical IDs for stable dedup."""
    if not href:
        return None
    # AgencyClientId query param style.
    m = re.search(r"[?&]AgencyClientId=([^&\"'#]+)", href)
    if m:
        return m.group(1)
    # /FAPPractitionerProfile/<id> path style.
    m = re.search(r"/FAPPractitionerProfile/([^?&\"'#/]+)", href)
    if m:
        return m.group(1)
    return None


def _build_source_url(agency_client_id: Optional[str]) -> Optional[str]:
    """Canonical per-practitioner URL — bare AgencyClientId form.

    The list-page link form ``/FAP/PractitionerDetail?AgencyClientId=<id>``
    is the canonical source_url we use across both the list parser
    (which emits this form natively) and the name-search parser (which
    sees the alternate ``/FAPPractitionerProfile/<id>`` form). Both
    target the same profile."""
    if not agency_client_id:
        return None
    return f"{DETAIL_URL}?AgencyClientId={agency_client_id}"


# Address-line regex: NCCAOM renders addresses as a single comma-joined
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
    Any field may be absent (returned as None / missing key)."""
    out: dict = {}
    if not line:
        return out
    s = re.sub(r"\s+", " ", line.strip()).strip().rstrip(",")
    if not s:
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

    NCCAOM renders the practitioner status as ``"<Status> | <Gender>"``
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
# Name-search form:          <a href="/FAPPractitionerProfile/...=" title="...">Name</a>
_NAME_ANCHOR_RE = re.compile(
    r'<a\s+href="((?:/FAP/PractitionerDetail\?AgencyClientId=|/FAPPractitionerProfile/)[^"]+)"'
    r'[^>]*>(.*?)</a>',
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

    # Address: <em id="addressdata_N">...</em> OR <em>...</em> directly
    # following an icon-location-pin marker. The id-anchored form is
    # the list-view style; the name-search layout uses the same em id
    # convention.
    a = re.search(
        r'<em[^>]*id="addressdata_\d+"[^>]*>(.*?)</em>',
        card_html,
        re.S | re.I,
    )
    if not a:
        # Fall back: any <em> following the location-pin icon.
        a = re.search(
            r'icon-location-pin[^<]*</i>\s*</span>\s*<em[^>]*>(.*?)</em>',
            card_html,
            re.S | re.I,
        )
    if not a:
        a = re.search(
            r'icon-location-pin[^<]*</i>\s*<p[^>]*class="copy"[^>]*>\s*<em[^>]*>(.*?)</em>',
            card_html,
            re.S | re.I,
        )
    if a:
        out["address"] = _strip_html_tags(a.group(1))

    # Phone: <i class="icon-call-end"></i> ... <em>...</em> (or 'Not Available').
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

    # Website: <i class="icon-globe"></i> ... <a href="..." title="...">.
    # The card carries BOTH a truncated display <span> and a full anchor.
    # Take the anchor href.
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

    # Cert codes: every <div class="cert-box__item">... <p class="copy">XX Certification</p> ...
    certs = re.findall(
        r'class="cert-box__item"[^>]*>.*?<p[^>]*class="copy"[^>]*>\s*([A-Za-z]+)\s+Certification',
        card_html,
        re.S | re.I,
    )
    out["cert_codes"] = [c.upper() for c in certs] if certs else []

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
    # pulled off the name.
    cred_parts: list[str] = []
    for code in card.get("cert_codes", []) or []:
        cred = _CERT_CODE_TO_CREDENTIAL.get(code)
        if cred and cred not in cred_parts:
            cred_parts.append(cred)
    if name_creds:
        # name-side creds (L.Ac., DAOM, ...) come AFTER the Dipl. codes
        # so the credential string reads as "Dipl. Ac. (NCCAOM), L.Ac.".
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
        source_org="NCCAOM",
        source_url=source_url,
        # Every listed NCCAOM diplomate is board-certified by definition;
        # the exception is the explicit inactive status downgrades.
        fellowship_level=not inactive,
        practice_name=None,  # not in list-grid cards; only in detail page
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

    Handles BOTH the production list layout
    (``<div class="result-card__item">`` inside
    ``<div id="pageListDetail">``) AND the name-search layout
    (``<div class="citySearchList__content">`` cards). Pagination is the
    caller's responsibility — this parser is page-scoped.
    """
    if not isinstance(html, str):
        return []

    rows: list[NormalizedPractitionerRow] = []
    seen_urls: set[str] = set()

    # List-view cards live in <div class="result-card__item">...</div>.
    # Split on the opening tag and parse each chunk.
    for chunk in re.split(r'<div\s+class="result-card__item"', html)[1:]:
        # Re-prepend the marker so the chunk is self-contained for regex,
        # then trim at the start of the NEXT result-card__item if any
        # leaked through (shouldn't, since split consumed it).
        card_html = '<div class="result-card__item"' + chunk
        # Trim trailing junk past the next card boundary if present.
        nxt = card_html.find('<div class="result-card__item"', 1)
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

    # Name-search layout: <div class="citySearchList__content"> ... </div>.
    # Iterate the same way, but use the alternative card marker.
    for chunk in re.split(r'<div\s+class="citySearchList__content"', html)[1:]:
        card_html = '<div class="citySearchList__content"' + chunk
        nxt = card_html.find('<div class="citySearchList__content"', 1)
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


# ---------------------------------------------------------------------------
# Full-coverage fetch
# ---------------------------------------------------------------------------

def fetch_all_records() -> list[NormalizedPractitionerRow]:
    """Walk every (Country, [State]) tuple the NCCAOM dropdown exposes
    and return a flat list of NormalizedPractitionerRow. Dedups by
    source_url within the run.

    Strategy:
      1. For Country=USA, walk all 50 states + DC + 5 territories.
      2. For every non-US country in ``NON_US_COUNTRIES``, walk
         Country=<NAME> (state is unavailable for these and search
         returns the country's full list).

    0.5s sleep between every HTTP call (fetch_search_page sleeps
    internally). Single-threaded — at ~17k expected total rows this
    completes in well under an hour.

    Used by migrate_nccaom.main(). Note that the live endpoint is
    Cloudflare-gated — production runs may need a Playwright wrapper
    around fetch_search_page; the parser itself is fully decoupled."""
    seen: set[str] = set()
    out: list[NormalizedPractitionerRow] = []

    # US states + DC + territories.
    for st in US_STATES:
        page = 1
        while True:
            html = fetch_search_page(country="USA", state=st, page=page)
            rows = parse_search_html(html)
            if not rows:
                break
            new_rows = [r for r in rows if r.source_url and r.source_url not in seen]
            for r in new_rows:
                seen.add(r.source_url)
                out.append(r)
            total = parse_total_pages(html)
            if total > 0 and page >= total:
                break
            if total == 0:
                # Defensive: no pagination info -> empty-page break only.
                break
            page += 1

    # Non-US countries.
    for c in NON_US_COUNTRIES:
        page = 1
        while True:
            html = fetch_search_page(country=c, page=page)
            rows = parse_search_html(html)
            if not rows:
                break
            new_rows = [r for r in rows if r.source_url and r.source_url not in seen]
            for r in new_rows:
                seen.add(r.source_url)
                out.append(r)
            total = parse_total_pages(html)
            if total > 0 and page >= total:
                break
            if total == 0:
                break
            page += 1

    return out
