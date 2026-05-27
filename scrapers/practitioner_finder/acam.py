"""American College for Advancement in Medicine (ACAM) scraper.

Discovery 2026-05-27
====================

ACAM's primary site (``acam.org`` / ``www.acam.org``) is fronted by a
Cloudflare managed-challenge that returns HTTP 403 to every plain-curl
request (browser-realistic UA, full accept-headers — still 403).
The WordPress / YourMembership REST endpoints behind the wall are
therefore unreachable without solving the JS challenge.

However, the **public-facing directory page** at
``https://www.acam.org/page/MembersState`` embeds the ACAM members map
in an ``<iframe>`` pointing at ZeeMaps:

    https://www.zeemaps.com/pub?group=3473180&legend=1&locate=1&search=1&list=1&shuttered=1

ZeeMaps is NOT behind Cloudflare. Its public ``emarkers`` JSON endpoint
returns the entire roster in a single static GET:

    GET https://www.zeemaps.com/emarkers?g=3473180

121 markers as of 2026-05-27. The payload is a flat list of dicts:

    id        -> stable numeric ZeeMaps marker id (dedup key)
    gid       -> always 3473180 (the ACAM group id)
    nm        -> name string. Usually a bare practitioner name
                 ("Kenneth Bock"); occasionally carries a credential
                 suffix after a comma ("Elena Klimenko, MD IFMCP -
                 Functional Medicine").
    ov        -> "owner" / display name (same as nm in this group)
    a         -> full formatted address single-line, e.g.
                 "50 Old Farm Rd, Red Hook 12571" — useful as a
                 fallback when the parsed city/state pieces are blank.
    s         -> street address (line 1 only)
    city      -> city
    state     -> US state abbr ("NY", "CA", ...) for US records;
                 sometimes a full state name ("Texas", "Sicilia") or a
                 numeric region code for intl records; sometimes blank.
    zip       -> postal / zip code (varies by country)
    cty       -> country code, but in **FIPS 10-4 style** (NOT ISO 3166)
                 — e.g. JA=Japan (ISO would be JP), SZ=Switzerland (ISO
                 CH), SP=Spain (ISO ES), SI=Slovenia (also ISO), CH=China
                 (ISO CN — yes, the FIPS CH/ISO CH collision means we
                 cannot trust a bare 'CH' as Switzerland; we have to
                 inspect the address). Some records use the full country
                 name as a string instead ("United States", "Italy",
                 "Slovak Republic"). The map preserves both.
    lat / lng -> already-geocoded coordinates (ignored here; the shared
                 Mapbox geocoder owns lat/lng for adapter-consistency)
    clr       -> color/legend bucket id (1 or 3 in the live data, 75/46
                 split). ZeeMaps does not expose a legend-label API to
                 the public; the ACAM map is published as "ACAM Members
                 - Embed" without an inline color->label legend
                 mapping reachable via static HTTP. We therefore cannot
                 derive Diplomate/ABCMT membership tier from ``clr`` in
                 isolation.
    c, b, lyr -> ZeeMaps internal flags (all 'F'/0/-1 in this group;
                 not informative)

Fellowship rule
===============

ACAM's elite tier is "Diplomate" (board-certified via the American Board
of Clinical Metal Toxicology — DABCMT/ABCMT — which ACAM created), per
spec. The ZeeMaps marker payload does NOT include a Diplomate flag,
and the public ZeeMaps APIs do not expose the color->label legend in a
form we can fetch without a browser. The only Diplomate signal we have
is when the practitioner happens to type the credential into their
``nm`` field (e.g. "Jane Doe, MD, DABCMT" — none in current data).

Decision: ``fellowship_level=True`` only when the name suffix carries
one of:

  * DABCMT  (Diplomate American Board of Clinical Metal Toxicology)
  * ABCMT   (the board itself; sometimes used as a credential)
  * Diplomate (literal word in the credential string)
  * "Board Certified" (literal phrase; rare in name field)

Default ``fellowship_level=False`` for every other record. This is
strictly conservative: under-counting is preferable to mis-flagging.
If ACAM ever exposes the Diplomate legend label publicly (e.g. by
adopting a per-marker description field on ZeeMaps, or by surfacing
the same data on a non-Cloudflare endpoint), this rule can be relaxed
in a follow-up.

Output rows
===========

Rows have tier='org_member', source_org='ACAM',
specialties=['functional_medicine', 'holistic_health']. The
per-practitioner ``source_url`` is the canonical ACAM MembersState
page anchored to the ZeeMaps marker id (so every row gets a unique,
stable upsert key across re-runs):

    https://www.acam.org/page/MembersState#zm-<id>
"""
import json
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
ACAM_BASE = "https://www.acam.org"
MEMBERS_STATE_URL = f"{ACAM_BASE}/page/MembersState"
ZEEMAPS_GROUP_ID = "3473180"
ZEEMAPS_MARKERS_URL = "https://www.zeemaps.com/emarkers"

LOCKED_SPECIALTIES = ["functional_medicine", "holistic_health"]


# Map ZeeMaps "cty" codes (mix of FIPS 10-4 + full country names + a few
# ISO 3166 alpha-2 codes) to ISO 3166 alpha-2.
#
# Important collisions to be aware of:
#
#   * 'JA' in ZeeMaps means Japan (FIPS) -> JP (ISO)
#   * 'SZ' in ZeeMaps means Switzerland (FIPS) -> CH (ISO)
#   * 'SP' in ZeeMaps means Spain (FIPS) -> ES (ISO)
#   * 'SI' is Slovenia in BOTH FIPS and ISO -> SI
#   * 'CH' in ZeeMaps means *China* (FIPS) -> CN (ISO).
#     (FIPS CH != ISO CH. The ISO code CH is Switzerland.)
#     We override by inspecting the address when 'CH' shows up so a
#     real Chinese address ("Shanghai", "Beijing", "...China") still
#     maps to CN.
#   * 'CA' is Canada in both FIPS and ISO.
#   * 'MX' is Mexico in both.
#   * 'AE' is UAE in both.
#
# Strings (full country names) are normalised case-insensitively. Any
# code we don't recognise returns None and the caller defaults to 'US'.
_CTY_TO_ISO2 = {
    # Full country names ----------------------------------------------------
    "united states": "US",
    "united states of america": "US",
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
    "deutschland": "DE",
    "france": "FR",
    "spain": "ES",
    "portugal": "PT",
    "italy": "IT",
    "italia": "IT",
    "netherlands": "NL",
    "the netherlands": "NL",
    "belgium": "BE",
    "switzerland": "CH",
    "austria": "AT",
    "poland": "PL",
    "czech republic": "CZ",
    "czechia": "CZ",
    "slovak republic": "SK",
    "slovakia": "SK",
    "slovenia": "SI",
    "hungary": "HU",
    "romania": "RO",
    "greece": "GR",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "iceland": "IS",
    "russia": "RU",
    "ukraine": "UA",
    "turkey": "TR",
    "israel": "IL",
    "saudi arabia": "SA",
    "united arab emirates": "AE",
    "uae": "AE",
    "qatar": "QA",
    "kuwait": "KW",
    "bahrain": "BH",
    "oman": "OM",
    "egypt": "EG",
    "south africa": "ZA",
    "kenya": "KE",
    "nigeria": "NG",
    "morocco": "MA",
    "mexico": "MX",
    "brazil": "BR",
    "argentina": "AR",
    "chile": "CL",
    "colombia": "CO",
    "peru": "PE",
    "ecuador": "EC",
    "venezuela": "VE",
    "costa rica": "CR",
    "panama": "PA",
    "guatemala": "GT",
    "dominican republic": "DO",
    "japan": "JP",
    "south korea": "KR",
    "korea": "KR",
    "china": "CN",
    "hong kong": "HK",
    "taiwan": "TW",
    "singapore": "SG",
    "malaysia": "MY",
    "thailand": "TH",
    "philippines": "PH",
    "vietnam": "VN",
    "indonesia": "ID",
    "india": "IN",
    "pakistan": "PK",
    "bangladesh": "BD",
    "sri lanka": "LK",
    # 2-letter codes (FIPS 10-4 -> ISO 3166-1 alpha-2 mapping). Where the
    # FIPS code matches the ISO code we list it explicitly to avoid the
    # lookup falling back to None. The CH collision is handled in the
    # _country_iso2 function below — we leave CH unresolved here so the
    # address-based override fires.
    "ca": "CA", "us": "US", "mx": "MX", "br": "BR",
    "gb": "GB", "de": "DE", "fr": "FR", "es": "ES", "it": "IT",
    "nl": "NL", "be": "BE", "at": "AT", "pl": "PL", "pt": "PT",
    "se": "SE", "no": "NO", "dk": "DK", "fi": "FI", "ie": "IE",
    "ro": "RO", "hu": "HU", "gr": "GR", "cz": "CZ",
    "ae": "AE", "il": "IL", "sa": "SA", "tr": "TR",
    "au": "AU", "nz": "NZ", "in": "IN", "pk": "PK", "id": "ID",
    "kr": "KR", "jp": "JP", "cn": "CN", "tw": "TW", "hk": "HK",
    "sg": "SG", "my": "MY", "th": "TH", "ph": "PH", "vn": "VN",
    "za": "ZA", "ke": "KE", "ng": "NG", "ma": "MA", "eg": "EG",
    "ar": "AR", "cl": "CL", "co": "CO", "pe": "PE", "ec": "EC",
    "ve": "VE", "cr": "CR", "pa": "PA", "gt": "GT", "do": "DO",
    "si": "SI",
    # FIPS-only (non-ISO) codes:
    "ja": "JP",   # FIPS Japan
    "sz": "CH",   # FIPS Switzerland
    "sp": "ES",   # FIPS Spain
    "gm": "DE",   # FIPS Germany (not in live data but defensive)
    "uk": "GB",
    "ic": "IS",   # FIPS Iceland
    "ks": "KR",   # FIPS South Korea
    "tu": "TR",   # FIPS Turkey
    "po": "PT",   # FIPS Portugal
    "gr": "GR",
    # NOTE: 'ch' deliberately omitted — see _country_iso2 below.
}


# Address-suffix keywords that imply a country when the raw cty code is
# ambiguous (e.g. FIPS 'CH' = China but ISO CH = Switzerland). Longest
# substring first to win over substrings (e.g. "China" beats nothing,
# "Shanghai"/"Beijing" cover specific cities).
_CHINA_HINTS = (
    "china", "shanghai", "beijing", "guangzhou", "shenzhen",
    "chengdu", "hangzhou", "wuhan", "tianjin", "nanjing",
)
_SWITZERLAND_HINTS = (
    "switzerland", "zurich", "geneva", "basel", "bern", "lausanne",
    "winterthur",
)


def _fips_ch_disambiguate(address: Optional[str]) -> str:
    """ZeeMaps' 'CH' code is FIPS China, which collides with ISO
    Switzerland. We pick based on the address text. Default to ISO
    convention (Switzerland) when there's no obvious China signal — most
    callers seeing 'CH' from a publisher who mixed conventions will mean
    Switzerland."""
    addr = (address or "").lower()
    for hint in _CHINA_HINTS:
        if hint in addr:
            return "CN"
    for hint in _SWITZERLAND_HINTS:
        if hint in addr:
            return "CH"
    return "CH"  # ISO default; conservative.


def _coerce_str(val) -> Optional[str]:
    """Return a stripped string or None for missing/empty values.

    ZeeMaps payloads are plain primitives (strings / numbers / booleans),
    not wrapped dicts. We accept dicts defensively for symmetry with the
    GeoDirectory-based adapters."""
    if val is None:
        return None
    if isinstance(val, dict):
        return _coerce_str(val.get("raw"))
    if isinstance(val, bool):
        return None  # booleans are not name/text content
    if isinstance(val, str):
        s = val.strip()
        return s or None
    s = str(val).strip()
    return s or None


def _country_iso2(cty_raw, address: Optional[str] = None) -> Optional[str]:
    """Map a ZeeMaps 'cty' value (FIPS-ish code, full country name, or
    occasionally an ISO 3166 alpha-2 code) to ISO 3166 alpha-2.

    Returns None for unrecognised input; caller defaults to 'US'.

    Special handling: bare 'CH' is ambiguous between FIPS China and ISO
    Switzerland; we disambiguate using the address (see
    _fips_ch_disambiguate)."""
    s = _coerce_str(cty_raw)
    if not s:
        return None
    key = s.lower()
    if key == "ch":
        return _fips_ch_disambiguate(address)
    return _CTY_TO_ISO2.get(key)


# Fellowship credential markers in the name suffix. Boundary-anchored
# so 'ABCMT' inside a longer junk token like 'ABCMTX' doesn't false-fire.
_FELLOWSHIP_RE = re.compile(
    r"\b(DABCMT|ABCMT|Diplomate|Board\s+Certified)\b",
    re.IGNORECASE,
)


def _is_fellowship_name(name: Optional[str]) -> bool:
    """True when the (already-extracted) name string carries a Diplomate
    marker. We check the raw nm field (including credential suffix) so
    'Jane Doe, MD, DABCMT' counts even after the comma split would
    have stripped the credential."""
    if not name:
        return False
    return bool(_FELLOWSHIP_RE.search(name))


def _split_name_credentials(nm_raw: Optional[str]) -> tuple[str, Optional[str]]:
    """Split a ZeeMaps name string into (name, credentials).

    Pattern: name is everything before the first comma; credentials are
    everything after. Honorifics ('Dr.', 'Dr', etc.) stay attached to
    the name. When there's no comma the entire value is the name (no
    credentials).

    Some entries pollute the credential field with descriptive copy
    ('MD IFMCP - Functional Medicine'); we keep it verbatim — downstream
    consumers display it as-is and the credential field is informational.

    Defensive: an em-dash / dash separator (' - ') outside of the comma
    form is NOT treated as a credentials boundary. Only the comma is
    structural.
    """
    s = _coerce_str(nm_raw)
    if not s:
        return "", None
    parts = s.split(",", 1)
    if len(parts) == 1:
        return parts[0].strip(), None
    name = parts[0].strip()
    creds = parts[1].strip().rstrip(",").rstrip()
    return name, (creds or None)


def _normalize_state(state_raw, country: str) -> Optional[str]:
    """Pass through US state abbrs / full names; for non-US records, drop
    placeholders that are clearly internal region codes (numeric or 2-3
    char numeric-prefixed). Real subdivision names (e.g. 'Sicilia',
    'British Columbia') are preserved."""
    s = _coerce_str(state_raw)
    if not s:
        return None
    if country == "US":
        return s
    # Non-US: ZeeMaps often stuffs an internal region id ('02', '15',
    # '23', '40', '01', '03') into state for international records.
    # These are not real subdivision names; drop them.
    if re.fullmatch(r"[0-9]{1,3}", s):
        return None
    return s


def _normalize_postal(zip_raw) -> Optional[str]:
    """Light cleanup — trim whitespace, drop obvious placeholders, but
    preserve country-specific formats (Canadian postal codes with a
    space, UAE/Singapore numeric, etc.)."""
    s = _coerce_str(zip_raw)
    if not s:
        return None
    if s.lower() in {"none", "n/a", "null", "0"}:
        return None
    return s


def _build_source_url(rec: dict) -> str:
    """Stable per-practitioner URL. The ZeeMaps map does not expose a
    per-marker detail URL that a member of the public can dereference,
    and the underlying ACAM YourMembership profile pages are gated
    behind Cloudflare. We therefore anchor each row at the ACAM
    MembersState page using the ZeeMaps marker id as the fragment.

    The result is stable across re-runs (ZeeMaps marker ids are
    permanent unless an admin re-creates the marker) and unique per
    practitioner.
    """
    rid = _coerce_str(rec.get("id")) or "unknown"
    return f"{MEMBERS_STATE_URL}#zm-{rid}"


# ---------------------------------------------------------------------------
# Stage 1: ZeeMaps marker fetch
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )
    return s


def fetch_all_directory_records() -> list[dict]:
    """Single GET against the ZeeMaps emarkers endpoint. Returns the
    list of marker records as-is (each is a flat dict — see the module
    docstring for the field layout).

    Single-threaded, static UA, 20s timeout, 0.5s sleep (rate-friendly
    per feedback_e4l_portal_no_concurrency.md). The entire ACAM roster
    ships in one response (~121 markers, ~33KB), so no pagination is
    required.
    """
    s = _session()
    r = s.get(
        ZEEMAPS_MARKERS_URL,
        params={"g": ZEEMAPS_GROUP_ID},
        timeout=20,
    )
    r.raise_for_status()
    time.sleep(0.5)
    data = r.json()
    if not isinstance(data, list):
        return []
    return data


# ---------------------------------------------------------------------------
# Pure parser
# ---------------------------------------------------------------------------

def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """ZeeMaps marker dict -> NormalizedPractitionerRow. Returns None
    when no usable name can be extracted."""
    raw_nm = _coerce_str(rec.get("nm")) or _coerce_str(rec.get("ov"))
    name, credentials = _split_name_credentials(raw_nm)
    if not name:
        return None

    # Address: prefer the structured pieces (s + city + state + zip);
    # fall back to the formatted 'a' for address1 when 's' is blank.
    address1 = _coerce_str(rec.get("s")) or _coerce_str(rec.get("a"))
    city = _coerce_str(rec.get("city"))

    # Country: cty may be FIPS-ish code, full name, or ISO2.
    # Pass the full formatted address as a disambiguation hint for the
    # known FIPS/ISO 'CH' collision.
    addr_for_hint = _coerce_str(rec.get("a"))
    country = _country_iso2(rec.get("cty"), addr_for_hint) or "US"

    state = _normalize_state(rec.get("state"), country)
    postal = _normalize_postal(rec.get("zip"))

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="ACAM",
        source_url=_build_source_url(rec),
        fellowship_level=_is_fellowship_name(raw_nm),
        # Practice name is not exposed in the ZeeMaps payload. Phone,
        # email, website all live on the gated YourMembership profile.
        practice_name=None,
        credentials=credentials,
        phone=None,
        email=None,
        website=None,
        address1=address1,
        city=city,
        state=state,
        postal=postal,
        country=country,
    )


def parse_directory_json(payload) -> list[NormalizedPractitionerRow]:
    """Pure parser. Accepts:

      - list of marker dicts                (natural shape from
        fetch_all_directory_records and the live emarkers endpoint)
      - dict of {id: marker}                (defensive — the same
        shape that the CSO adapter expects)
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
