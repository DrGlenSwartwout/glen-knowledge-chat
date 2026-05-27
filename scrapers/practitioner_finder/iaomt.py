"""International Academy of Oral Medicine and Toxicology (IAOMT) scraper.

IAOMT publishes its directory through a Zoho-backed AJAX endpoint exposed
by the ``publicdirectory`` WordPress plugin:

    POST https://iaomt.org/wp-admin/admin-ajax.php
         action=custom_ajax
         for=getAccountListData
         per_page=100&page=N
         isListInital=false
         initial_country_name=United States
         initial_country_code=us

Quirks discovered 2026-05-27:
  - ``page`` is 1-based even though the JS sends ``currentPage + 1``.
    Calling page=0 produces ``OFFSET -per_page`` (empty result set).
  - ``isListInital=true`` short-circuits the SQL filter and returns
    ``total_records_count: 0``. Sending an explicit country pair flips the
    flag the same way the on-site JS does after init.
  - Active total at discovery time: 1,459 dental + hygiene members.

Each record is a flat dict (Zoho CRM "Account" row) with the relevant
fields:

  Account_Name            -> practitioner name
  Parent_Account          -> practice name
  Address_Line_1/2, City_District, State_Province, Postal_Code, Country
                          -> location
  Office_Phone            -> phone (Main_Office_Fax / Mobile_Phone fallback)
  Main_Office_Email       -> email (Personal_Email fallback)
  Website / Main_Office_Website -> website
  Degrees                 -> JSON-encoded list, e.g. '["BS","DDS","NMD","MS"]'
  Other_Degrees           -> comma-separated extras (often includes "MIAOMT" /
                            "FIAOMT" / "AIAOMT" which encode the IAOMT level)
  Master / Fellow / Accredited
  Hygiene_Master / Hygiene_Fellow / Biological_Dental_Hygiene_Accredited
  Smart / General         -> 1 / "" flags for IAOMT credential tier
  module_id               -> Zoho UUID; stable per practitioner (dedup key)
  Address_lat / Address_lng -> already-geocoded coordinates (ignored here;
                            the shared geocoder owns lat/lng to keep the
                            quality field consistent across adapters)

Output rows have tier='org_member', source_org='IAOMT',
specialties=['biological', 'dental']. Practitioners holding any of
{Master, Fellow, Accredited, Hygiene_Master, Hygiene_Fellow,
 Biological_Dental_Hygiene_Accredited} get fellowship_level=True (per
spec: "Accredited Member or Master or Fellow"). General / SMART-only
members are not fellowship-tier.

The per-practitioner ``source_url`` mirrors the JS-built link:
    https://iaomt.org/for-patients/members/<name-slug>[-<degrees-slug>]/?ppage=dashboard
It is stable across re-runs because name + degrees do not change.
"""
import html as html_module
import json
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://iaomt.org"
AJAX_URL = f"{BASE}/wp-admin/admin-ajax.php"
SEARCH_PAGE_URL = f"{BASE}/for-patients/search/"

LOCKED_SPECIALTIES = ["biological", "dental"]

# Country-name -> ISO2 for the most common entries in the directory.
# Anything not in this map falls back to None so the geocoder sees the raw
# country name in the row's `address1`/`country` (the geocoder is tolerant of
# either ISO2 or full names).
_COUNTRY_NAME_TO_ISO2 = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "canada": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
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
    "poland": "PL",
    "czech republic": "CZ",
    "greece": "GR",
    "hungary": "HU",
    "romania": "RO",
}

# IAOMT fellowship-tier credential flags. Spec: any of Accredited Member /
# Master / Fellow (incl. the hygiene-track equivalents) counts.
_FELLOWSHIP_FLAGS = (
    "Master",
    "Fellow",
    "Accredited",
    "Hygiene_Master",
    "Hygiene_Fellow",
    "Biological_Dental_Hygiene_Accredited",
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    return s


# ---------------------------------------------------------------------------
# Stage 1: paginated AJAX index fetch
# ---------------------------------------------------------------------------

def fetch_directory_json(page: int = 1, per_page: int = 100) -> dict:
    """Hit the IAOMT publicdirectory AJAX endpoint and return the parsed
    JSON dict for a single page.

    Static UA + 0.5s sleep + 20s timeout (mirror reference convention).
    Caller is responsible for paging: when ``has_more`` is True, request
    the next page.
    """
    s = _session()
    payload = {
        "for": "getAccountListData",
        "action": "custom_ajax",
        "per_page": str(per_page),
        "page": str(page),
        "cppageUrl": SEARCH_PAGE_URL,
        "isListInital": "false",
        # Sending a country pair is what flips the JS off the no-op
        # ``isListInital`` path; the value itself is not used as a filter.
        "initial_country_name": "United States",
        "initial_country_code": "us",
    }
    r = s.post(AJAX_URL, data=payload, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.json()


def fetch_all_directory_records() -> list[dict]:
    """Page through the AJAX endpoint until ``has_more`` is False and
    return the concatenated practitioner records."""
    out: list[dict] = []
    page = 1
    per_page = 100
    while True:
        data = fetch_directory_json(page=page, per_page=per_page)
        batch = data.get("data") or []
        out.extend(batch)
        if not data.get("has_more"):
            break
        if not batch:
            # Defensive: has_more=true but empty page would loop forever.
            break
        page += 1
    return out


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Return a stripped string or None for missing/empty/null values."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s or None
    s = str(val).strip()
    return s or None


def _parse_degrees(raw) -> list[str]:
    """Degrees comes as a JSON-encoded list string like '["DDS","NMD"]'.
    Returns the inner list (empty on parse failure or missing)."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip() for x in parsed if str(x).strip()]


def _format_credentials(degrees: list[str], other_degrees: Optional[str]) -> Optional[str]:
    """Combine the JSON-list ``Degrees`` and free-text ``Other_Degrees`` into
    one comma-separated credential string. Other_Degrees often carries the
    actual IAOMT designation (e.g. ``MIAOMT``, ``FIAOMT``, ``AIAOMT``)."""
    parts: list[str] = list(degrees)
    if other_degrees:
        for chunk in other_degrees.split(","):
            chunk = chunk.strip()
            if chunk and chunk not in parts:
                parts.append(chunk)
    if not parts:
        return None
    return ", ".join(parts)


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Add an https:// scheme if the directory entry lists a bare domain."""
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return f"https://{s}"


def _country_iso2(raw: Optional[str]) -> Optional[str]:
    """Map a free-text country name to ISO2; None if unrecognized."""
    if not raw:
        return None
    key = raw.strip().lower()
    return _COUNTRY_NAME_TO_ISO2.get(key)


def _join_address_lines(line1: Optional[str], line2: Optional[str]) -> Optional[str]:
    parts = [p for p in (line1, line2) if p]
    if not parts:
        return None
    return ", ".join(parts)


def _slugify(s: str) -> str:
    s = html_module.unescape(s).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _build_source_url(name: str, degrees: list[str], module_id: Optional[str]) -> str:
    """Reproduce the slug the JS builds for each card's detail link.

    JS:
        accountCustomLink = '/for-patients/members/' + name-words-hyphen-lower
        if degrees: accountCustomLink += '-' + degrees-no-comma-no-paren-lower
        accountCustomLink += '/?ppage=dashboard'

    ``module_id`` is appended as a URL fragment when present so two
    practitioners who share a slug (rare but possible) still produce
    distinct source_urls — the fragment is ignored by the IAOMT server
    but preserved by the (source_url) unique constraint in upsert.
    """
    name_slug = _slugify(name) if name else "unknown"
    # Mirror the JS slug rule:
    #   "(BS, DDS, NMD, MS)" -> strip parens -> strip whitespace ->
    #   commas-to-dashes -> lowercase  ==> "bs-dds-nmd-ms"
    # We skip the paren/whitespace stage and just hyphenate the joined list.
    deg_slug = "-".join(d.strip() for d in degrees if d and d.strip()).lower()
    deg_slug = re.sub(r"[^a-z0-9-]+", "-", deg_slug).strip("-")
    path = f"/for-patients/members/{name_slug}"
    if deg_slug:
        path = f"{path}-{deg_slug}"
    path = f"{path}/?ppage=dashboard"
    url = f"{BASE}{path}"
    if module_id:
        url = f"{url}#{module_id}"
    return url


def _is_fellowship(record: dict) -> bool:
    """True if the record carries any tier-Master/Fellow/Accredited flag."""
    for flag in _FELLOWSHIP_FLAGS:
        v = record.get(flag)
        if v in (1, "1", True):
            return True
    return False


def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: Zoho-shape dict -> NormalizedPractitionerRow.

    Returns None if the record lacks a usable practitioner name (defensive
    against partial / hidden entries)."""
    name = _coerce_str(rec.get("Account_Name"))
    if not name:
        return None

    degrees = _parse_degrees(rec.get("Degrees"))
    other = _coerce_str(rec.get("Other_Degrees"))
    credentials = _format_credentials(degrees, other)

    practice = _coerce_str(rec.get("Parent_Account")) or _coerce_str(rec.get("Practice_Name1"))
    # If the practice name matches the practitioner name (solo practice
    # where they registered themselves as the practice), suppress it to
    # avoid duplicating the name in the UI.
    if practice and practice.lower() == name.lower():
        practice = None

    phone = (
        _coerce_str(rec.get("Office_Phone"))
        or _coerce_str(rec.get("Mobile_Phone"))
        or _coerce_str(rec.get("Main_Office_Fax"))
    )
    email = (
        _coerce_str(rec.get("Main_Office_Email"))
        or _coerce_str(rec.get("Personal_Email"))
    )
    website = _normalize_website(
        _coerce_str(rec.get("Website"))
        or _coerce_str(rec.get("Main_Office_Website"))
    )

    address1 = _join_address_lines(
        _coerce_str(rec.get("Address_Line_1")),
        _coerce_str(rec.get("Address_Line_2")),
    )
    city = _coerce_str(rec.get("City_District"))
    state = _coerce_str(rec.get("State_Province"))
    postal = _coerce_str(rec.get("Postal_Code"))
    country_raw = _coerce_str(rec.get("Country"))
    country = _country_iso2(country_raw) or country_raw or "US"

    module_id = _coerce_str(rec.get("module_id")) or _coerce_str(rec.get("id"))
    source_url = _build_source_url(name, degrees, module_id)

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="IAOMT",
        source_url=source_url,
        fellowship_level=_is_fellowship(rec),
        practice_name=practice,
        credentials=credentials,
        phone=phone,
        email=email,
        website=website,
        address1=address1,
        city=city,
        state=state,
        postal=postal,
        country=country,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_directory_json(payload) -> list[NormalizedPractitionerRow]:
    """Pure parser: takes an AJAX response (dict or raw JSON string)
    and returns one NormalizedPractitionerRow per usable record.

    Accepts either:
      - the full response dict ({'data': [...], 'total_records_count': ...})
      - just the list of records
      - a JSON string of either form
    No I/O."""
    if isinstance(payload, (str, bytes, bytearray)):
        payload = json.loads(payload)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("data") or []
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
