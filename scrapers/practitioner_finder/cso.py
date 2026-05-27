"""College of Syntonic Optometry (CSO) scraper.

CSO publishes its practitioner directory at
https://csovision.org/find-a-practitioner/ as a single static page that
embeds the entire member roster as a JSON blob inside a WP Google Map
Plugin (10Web / WebDorado ``gmwd``) initialization script.

Discovery 2026-05-27: the page returns full HTML directly (no JS render
required to surface the data). Inside the page source, the marker payload
appears as:

    gmwdmapData["mapMarkers" + '5'] = JSON.parse('{"<id>":{...}, ...}');

The single ``JSON.parse('...')`` argument is the entire practitioner
roster — 375 markers as of 2026-05-27 — keyed by stable numeric WP map
marker id. Each marker is a flat dict:

    id           -> stable map-marker id (string of int; dedup key)
    map_id       -> always '5' (the CSO Find-a-Practitioner map)
    lat / lng    -> already-geocoded coords (informational only; the
                    shared Mapbox geocoder owns lat/lng to keep the
                    geocode_quality field consistent across adapters)
    category     -> CSO's internal region grouping ('1'..'12' or ''); NOT
                    a reliable country mapping (cat 3 carries a PR entry,
                    cat 6 mixes UK + Ireland + Ohio, etc.) — country is
                    inferred from the address suffix instead.
    title        -> "First Last, OD, FCSO" or similar — credential
                    suffix is comma-separated after the name. This is the
                    ONLY source of the FCSO fellowship marker.
    address      -> free-form single-line address; format varies by
                    country (US is typically "Street City, ST ZIP"; intl
                    ends in country name).
    description  -> list[str] of one HTML-escaped <p> blob carrying
                    practice name (in <strong><a href="...">), phone,
                    fax, and any mailto: emails.
    link_url     -> practice website (occasionally polluted with "Tel:"
                    text — defensively trimmed)
    pic_url      -> headshot URL (ignored; portal-managed photo_url)
    published    -> '1' for visible rows (all 375 are published)

Fellowship rule: CSO's elite credential is F.C.S.O. / FCSO (Fellow
College of Syntonic Optometry). The directory always writes it without
dots as the bare token "FCSO" inside the credential suffix of the title
(e.g. "Cathy Stern, O.D., FCOVD, FNORA, FCSO"). 33 of 375 records carry
it. We accept the dot variant defensively in case the data ever shifts.

Output rows have tier='org_member', source_org='CSO',
specialties=['syntonic', 'eye_care']. The per-practitioner source_url is
the canonical Find-a-Practitioner page anchored to the marker id (so
each row remains a stable, unique upsert key across re-runs):

    https://csovision.org/find-a-practitioner/#marker-<id>
"""
import html as html_module
import json
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://csovision.org"
FIND_URL = f"{BASE}/find-a-practitioner/"

LOCKED_SPECIALTIES = ["syntonic", "eye_care"]

# The CSO directory mixes everything from US dentists' offices to
# practitioners in Andorra; address tail-detection mirrors the OEPF /
# IABDM convention.
_COUNTRY_NAME_TO_ISO2 = {
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "canada": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "northern ireland": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "ireland": "IE",
    "australia": "AU",
    "new zealand": "NZ",
    "germany": "DE",
    "deutschland": "DE",
    "netherlands": "NL",
    "the netherlands": "NL",
    "belgium": "BE",
    "switzerland": "CH",
    "austria": "AT",
    "spain": "ES",
    "andorra": "AD",
    "portugal": "PT",
    "italy": "IT",
    "france": "FR",
    "poland": "PL",
    "mexico": "MX",
    "brazil": "BR",
    "argentina": "AR",
    "chile": "CL",
    "colombia": "CO",
    "philippines": "PH",
    "india": "IN",
    "japan": "JP",
    "south korea": "KR",
    "korea": "KR",
    "china": "CN",
    "singapore": "SG",
    "malaysia": "MY",
    "indonesia": "ID",
    "south africa": "ZA",
    "kenya": "KE",
    "israel": "IL",
    "turkey": "TR",
    "united arab emirates": "AE",
    "uae": "AE",
}

_US_STATE_ABBR_SET = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

# US full-state-name -> abbr (CSO addresses use both "ST ZIP" and
# "<State Name> ZIP"; we canonicalize to the 2-letter code).
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

# "..., <City>, <ST> <ZIP>[, USA]" or "..., <City>, <ST>[, USA]"
# Two CSO US address shapes:
#   "Street, City, ST ZIP"      (two commas; city in its own segment)
#   "Street City, ST ZIP"       (one comma; only City precedes ", ST ZIP")
# Both are covered by anchoring on the LAST ", ST <ZIP>" tail and walking
# left from there.
_US_TAIL_ABBR_RE = re.compile(
    r",\s*([A-Z]{2})"
    r"(?:\s*,?\s*USA?)?\s*(\d{5}(?:-\d{4})?)?"
    r"(?:\s*,?\s*USA?)?\s*\.?\s*$"
)
# Full-state-name tail: ", <Full State Name>[ <ZIP>][, USA]" anchored
# right. The preceding city is whatever lies between the previous comma
# (or start of string) and this match.
_US_TAIL_NAME_RE = re.compile(
    r",\s*"
    r"(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|"
    r"Delaware|District of Columbia|Florida|Georgia|Hawaii|Idaho|Illinois|"
    r"Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|"
    r"Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|"
    r"New Hampshire|New Jersey|New Mexico|New York|North Carolina|"
    r"North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|"
    r"South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|"
    r"Washington|West Virginia|Wisconsin|Wyoming)\s*"
    r"(\d{5}(?:-\d{4})?)?\s*(?:,?\s*USA?)?\s*\.?\s*$",
    re.IGNORECASE,
)

# Fellowship marker: the bare token FCSO (or the dotted F.C.S.O. variant
# defensively) appearing as a credential token, surrounded by word
# boundaries so we don't false-positive on, e.g., "FCSOX".
_FCSO_RE = re.compile(r"\bF\.?C\.?S\.?O\.?\b")

# Locate the entire mapMarkers JSON.parse(...) call in the page source.
# Captures the single-quoted JSON string argument.
_MARKER_BLOB_RE = re.compile(
    r"""mapMarkers["\s+'\d+\]]+\s*=\s*JSON\.parse\(\s*'(.*?)'\s*\)\s*;""",
    re.DOTALL,
)

# Extract a mailto: email from the description HTML.
_MAILTO_RE = re.compile(
    r"""mailto:([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})""",
    re.IGNORECASE,
)

# Extract the practice name (first <strong>...</strong> after unescape).
_PRACTICE_STRONG_RE = re.compile(r"<strong>(.*?)</strong>", re.IGNORECASE | re.DOTALL)
# Extract phone: "Tel: <number>" up to the next <br> or end-of-tag.
# Number can start with '(', '+', or a digit; allow digits, whitespace,
# parens, plus, dash, dot, slash inside. Min length 6 chars to avoid
# matching stray "Tel: 1" snippets.
_TEL_RE = re.compile(r"Tel\s*:?\s*([+(\d][\d\s().+\-/]{5,})", re.IGNORECASE)


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
# Stage 1: fetch the Find-a-Practitioner page and pull out the marker blob
# ---------------------------------------------------------------------------

def fetch_directory_page() -> str:
    """Download the full Find-a-Practitioner page HTML.

    Single GET — no pagination; the entire roster ships in one response.
    Static UA + 20s timeout + 0.5s sleep (rate-friendly, per
    feedback_e4l_portal_no_concurrency.md)."""
    s = _session()
    r = s.get(FIND_URL, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


def extract_marker_payload(html: str) -> dict:
    """Pull the mapMarkers JSON.parse('...') blob out of the page HTML and
    return the parsed dict {id: marker_record}.

    Returns an empty dict if the blob is missing (defensive — would
    indicate the CSO site swapped its directory plugin).
    """
    m = _MARKER_BLOB_RE.search(html)
    if not m:
        return {}
    raw = m.group(1)
    # JS single-quoted string -> JSON. The only escape WP emits in the
    # captured payload is \' for embedded apostrophes; backslash-escapes
    # of double-quote, backslash, slash, and unicode (\\, \", \/, \uXXXX)
    # are all valid in JSON as-is.
    raw = raw.replace("\\'", "'")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def fetch_all_directory_records() -> list[dict]:
    """Fetch + extract — return the list of marker records (one dict per
    practitioner). Drops the outer id->record keying; the id stays inside
    each record."""
    html = fetch_directory_page()
    data = extract_marker_payload(html)
    return list(data.values())


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Return a stripped string or None for missing/empty values. The CSO
    payload is plain strings throughout (not the GeoDirectory wrapped-dict
    shape), but we accept dicts defensively for symmetry with sibling
    adapters."""
    if val is None:
        return None
    if isinstance(val, dict):
        return _coerce_str(val.get("raw"))
    if isinstance(val, str):
        s = val.strip()
        return s or None
    s = str(val).strip()
    return s or None


def _split_title(title: str) -> tuple[str, Optional[str]]:
    """Split 'First Last, OD, FCSO' into (name, credentials).

    Credentials are everything after the FIRST comma. Honorifics
    ('Dr.', 'Dr', 'Dra.') stay attached to the name. Trailing dots on
    credential tokens are preserved ("O.D." stays "O.D."); only a stray
    trailing comma or whitespace is trimmed. CSO titles are relatively
    clean — no parenthesized credential variants in the live data — so
    a single comma split is sufficient.
    """
    if not title:
        return "", None
    s = title.strip()
    # Period-separated suffix is unusual but defensive: "Brenda Montecalvo. OD"
    # treat the first ". " before a credential-shaped token as a comma.
    s = re.sub(r"\.\s+(?=[A-Z][A-Z.]{0,8}\b)", ", ", s)
    parts = s.split(",", 1)
    if len(parts) == 1:
        return parts[0].strip(), None
    name = parts[0].strip()
    creds = parts[1].strip().rstrip(",").rstrip()
    return name, (creds or None)


def _has_fcso(title: Optional[str]) -> bool:
    """True when the title carries the FCSO credential. Case-sensitive on
    purpose: lowercase 'fcso' would be unusual and a likely false positive
    (no instances in the real data)."""
    if not title:
        return False
    return bool(_FCSO_RE.search(title))


def _country_iso2(name: Optional[str]) -> Optional[str]:
    """Map a free-text country name (or trailing-word from an address) to
    ISO2; None if unrecognized."""
    s = _coerce_str(name)
    if not s:
        return None
    return _COUNTRY_NAME_TO_ISO2.get(s.lower())


def _detect_country_from_address(addr: str) -> Optional[str]:
    """Scan the address for a known country name. Returns ISO2 or None
    (None means 'undetermined' — caller defaults to 'US' since the
    overwhelming majority of CSO records are US).

    Handles three positions:
      - End-of-string ("...Stocksfield NE43 7PY Northumberland United Kingdom")
      - End with trailing postal/punctuation ("...São Paulo, SP. Brazil. 05401-450")
      - Mid-string punctuation ("...Madison, WI USA 53718")

    Longest-name matched first so "United Kingdom" wins over "UK" when
    both apply, and "South Africa" doesn't shadow with a substring.
    """
    if not addr:
        return None
    lower = addr.lower().strip().rstrip(".")
    for country_name in sorted(_COUNTRY_NAME_TO_ISO2.keys(), key=len, reverse=True):
        # Anchored suffix.
        if lower.endswith(country_name):
            return _COUNTRY_NAME_TO_ISO2[country_name]
        # Punctuation/postal trailing the country name. The country must
        # be a separate token (preceded by whitespace, comma, or string
        # start) to avoid matching, e.g., "indiana" -> "india".
        idx = lower.rfind(country_name)
        while idx >= 0:
            pre = lower[idx - 1] if idx > 0 else " "
            post_start = idx + len(country_name)
            post = lower[post_start:] if post_start < len(lower) else ""
            # Boundary before: whitespace / comma / start.
            # Boundary after: ONLY whitespace / punctuation / digits
            # (postal codes / phone). Disallow letters following so we
            # don't false-match "india" inside "indiana".
            after_ok = post == "" or not post[0].isalpha()
            before_ok = pre in " ,\t\n" or idx == 0
            if before_ok and after_ok:
                return _COUNTRY_NAME_TO_ISO2[country_name]
            idx = lower.rfind(country_name, 0, idx)
    return None


def _split_us_address(raw: str) -> Optional[tuple[Optional[str], str, str, Optional[str]]]:
    """Best-effort US address split. Returns (address1, city, state_abbr,
    postal) or None when no confident US tail is detected.

    Anchors on the rightmost ", <STATE>[ <ZIP>][, USA]" tail. Whatever
    sits to the LEFT of that comma is split into (address1, city) by
    walking left to the previous comma (the typical 'Street, City, ST
    ZIP' form). If no previous comma exists, the entire left segment is
    treated as both street and city (the geocoder will resolve)."""
    if not raw:
        return None
    s = raw.strip().rstrip(".")

    # Try abbreviated state form first.
    m = _US_TAIL_ABBR_RE.search(s)
    if m:
        state = m.group(1).strip()
        postal = m.group(2).strip() if m.group(2) else None
        if state in _US_STATE_ABBR_SET:
            head = s[: m.start()].strip()
            # Split head on the LAST remaining comma -> (address1, city).
            if "," in head:
                addr1, city = head.rsplit(",", 1)
                addr1 = addr1.strip() or None
                city = city.strip()
            else:
                # Single-comma form "Street City, ST ZIP": the whole head
                # is "Street City"; geocoder will sort out city/street.
                addr1 = head or None
                city = None
            return addr1, city, state, postal

    # Try full-state-name form.
    m = _US_TAIL_NAME_RE.search(s)
    if m:
        state_name = m.group(1).strip().lower()
        postal = m.group(2).strip() if m.group(2) else None
        state = _US_STATE_NAME_TO_ABBR.get(state_name)
        if state:
            head = s[: m.start()].strip()
            if "," in head:
                addr1, city = head.rsplit(",", 1)
                addr1 = addr1.strip() or None
                city = city.strip()
            else:
                addr1 = head or None
                city = None
            return addr1, city, state, postal

    return None


def _parse_address(raw: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
    """Returns (address1, city, state, postal, country).

    Strategy:
      1) If US-tail regex matches, return the structured split with
         country='US'.
      2) Otherwise detect country from address suffix; the entire raw
         string stays in address1 (city/state/postal stay None — the
         geocoder will sort them out).
      3) Country defaults to 'US' when undetermined. CSO is ~60% US;
         leaving country=None breaks downstream tier/specialty filters
         that expect a country to exist.
    """
    s = _coerce_str(raw)
    if not s:
        return None, None, None, None, "US"

    us = _split_us_address(s)
    if us is not None:
        head, city, state, postal = us
        return head, city, state, postal, "US"

    country = _detect_country_from_address(s) or "US"
    return s, None, None, None, country


def _strip_html(text: str) -> str:
    """Cheap tag-stripper for the description blob (after entity unescape)."""
    return re.sub(r"<[^>]+>", " ", text)


def _parse_description(description) -> dict:
    """Pull practice_name / phone / email out of the description list.

    CSO stores the description as a list of one HTML-escaped string;
    pages occasionally have multiple <p> blocks in one entry but the
    typical structure is:

        <p><strong><a href="...">Practice Name</a></strong><br />
           Tel: 555-555-1234<br />
           Fax: 555-555-5678<br />
           <a href="mailto:foo@bar.com">email: Dr. Foo</a></p>

    Returns a dict with optional 'practice_name', 'phone', 'email'.
    Missing fields are omitted (not set to None) so the caller can
    distinguish 'not in description' from 'extracted as empty'.
    """
    if not description:
        return {}
    if isinstance(description, list):
        joined = " ".join(str(d) for d in description if d)
    else:
        joined = str(description)
    if not joined:
        return {}

    # Two passes of unescape — WP often double-escapes &amp;quot; etc.
    text = html_module.unescape(html_module.unescape(joined))

    out: dict = {}

    m = _PRACTICE_STRONG_RE.search(text)
    if m:
        inner = _strip_html(m.group(1)).strip()
        # Sometimes the <strong> wraps a stand-alone "Tel:" or single
        # heading — only keep when it looks like a real practice name
        # (not starting with 'Tel:' or just a phone number).
        if inner and not inner.lower().startswith("tel"):
            out["practice_name"] = inner

    m = _MAILTO_RE.search(text)
    if m:
        out["email"] = m.group(1).strip().lower()

    m = _TEL_RE.search(text)
    if m:
        phone = m.group(1).strip()
        # Trim anything after a trailing slash or extra whitespace.
        phone = re.split(r"[/\n]", phone, maxsplit=1)[0].strip()
        # Drop the trailing punctuation often left from "555-555-1234<br/>"
        phone = phone.rstrip(".,-")
        if phone:
            out["phone"] = phone

    return out


def _normalize_website(raw) -> Optional[str]:
    """Clean up the link_url field: enforce a scheme, drop CSO's
    occasional 'Tel: ...' pollution that ended up in the URL field."""
    s = _coerce_str(raw)
    if not s:
        return None
    # CSO sometimes concatenates the URL with phone text. Truncate at
    # the first whitespace or known-pollution token.
    s = re.split(r"\s+(?:Tel|Fax|Mobil|email)\b", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if not s or s.lower() in {"none", "n/a", "null"}:
        return None
    if not (s.startswith("http://") or s.startswith("https://")):
        s = f"https://{s}"
    return s


def _build_source_url(rec: dict) -> str:
    """Stable per-practitioner URL. The CSO map page has no per-marker
    detail URL (the directory is a single-page map with info-window
    popups), so we anchor on the marker id within the page URL. This is
    stable across re-runs and unique per record.
    """
    rid = _coerce_str(rec.get("id")) or "unknown"
    return f"{FIND_URL}#marker-{rid}"


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: CSO marker dict -> NormalizedPractitionerRow.

    Returns None when no usable name can be extracted from the title.
    """
    title = _coerce_str(rec.get("title"))
    name, credentials = _split_title(title or "")
    if not name:
        return None

    address1, city, state, postal, country = _parse_address(rec.get("address"))

    desc_fields = _parse_description(rec.get("description"))

    practice = desc_fields.get("practice_name")
    # Suppress practice name when it duplicates the practitioner name.
    if practice and practice.lower() == name.lower():
        practice = None

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="CSO",
        source_url=_build_source_url(rec),
        fellowship_level=_has_fcso(title),
        practice_name=practice,
        credentials=credentials,
        phone=desc_fields.get("phone"),
        email=desc_fields.get("email"),
        website=_normalize_website(rec.get("link_url")),
        address1=address1,
        city=city,
        state=state,
        postal=postal,
        country=country,
    )


def parse_directory_json(payload) -> list[NormalizedPractitionerRow]:
    """Pure parser. Accepts:

      - dict of {id: marker_record}  (the natural shape from
        extract_marker_payload)
      - list of marker_records       (the shape from
        fetch_all_directory_records)
      - JSON string of either

    Returns one NormalizedPractitionerRow per usable record; records
    with no extractable name are silently dropped.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        payload = json.loads(payload)

    if isinstance(payload, dict):
        records = list(payload.values())
    elif isinstance(payload, list):
        records = payload
    else:
        return []

    rows: list[NormalizedPractitionerRow] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        row = _record_to_row(rec)
        if row is not None:
            rows.append(row)
    return rows


def parse_directory_html(html: str) -> list[NormalizedPractitionerRow]:
    """End-to-end pure parser: page HTML -> rows. Used by tests that
    capture full-page HTML fixtures."""
    data = extract_marker_payload(html)
    return parse_directory_json(data)
