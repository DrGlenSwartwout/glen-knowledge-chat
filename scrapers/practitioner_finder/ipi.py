"""Integrative Psychiatry Institute (IPI) directory scraper.

IPI publishes its provider directory as a GeoDirectory WordPress site at
``directory.psychiatryinstitute.com``. The on-page cards are server-rendered,
but the plugin also exposes a clean public GeoDirectory v2 REST endpoint that
returns every listing as flat JSON — no login, no Cloudflare, no JS execution
required (verified 2026-05-29 over a plain ``requests`` GET):

    GET https://directory.psychiatryinstitute.com/wp-json/geodir/v2/listings
        ?per_page=100&page=N

Pagination: the response headers carry ``X-WP-Total`` / ``X-WP-TotalPages``.
At discovery time the directory held 66 ``gd_place`` listings, so a single
per_page=100 request returns the whole directory. The fetcher still pages
defensively in case the directory grows past 100.

Each record is a flat dict with the relevant fields:

  title.raw / title.rendered  -> practitioner (or practice) name
  link                        -> canonical permalink, unique per listing
                                 (used directly as the stable source_url /
                                 dedup key — never changes for a given listing)
  job_title                   -> e.g. "Integrative Psychiatrist"
  degrees   {raw, rendered[]} -> e.g. ["MD"], ["MSW"], ["MD","MPH"]
  licenses  {raw, rendered[]} -> e.g. ["LCSW"] (often empty)
  ipi_certifications {raw, rendered[]}
                              -> IPI credential designations. The two values
                                 in use are:
                                   "Certified Integrative Psychiatric Provider"
                                     (CIPP — the IPI Fellowship designation)
                                   "Certified Psychedelic Assisted Therapy Provider"
                                 Many listings carry neither (rendered == []).
  phone_number / email / website
  street / city / region / zip / country
  latitude / longitude        -> already geocoded (IGNORED here; the shared
                                 geocoder owns lat/lng so geocode_quality stays
                                 consistent across adapters)

Output rows have tier='org_member', source_org='IPI',
specialties=['integrative_psychiatry', 'holistic_health'].

FELLOWSHIP RULE: ``fellowship_level=True`` when the listing carries the
"Certified Integrative Psychiatric Provider" (CIPP) designation in
``ipi_certifications`` (case-insensitive substring "integrative psychiatric").
This is the IPI Fellowship credential. The directory is NOT fellowship-only —
at discovery the split was 14 CIPP holders, 40 carrying only the Psychedelic
provider cert, and 15 with no IPI certification listed; so only the 14 CIPP
holders get fellowship_level=True.
"""
import html as html_module
import json
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://directory.psychiatryinstitute.com"
LISTINGS_URL = f"{BASE}/wp-json/geodir/v2/listings"

LOCKED_SPECIALTIES = ["integrative_psychiatry", "holistic_health"]

# The IPI Fellowship designation. CIPP = "Certified Integrative Psychiatric
# Provider". Matched case-insensitively as a substring so minor punctuation /
# spacing variants still resolve.
_FELLOWSHIP_MARKER = "integrative psychiatric"

# Country-name -> ISO2. The directory is US-only at discovery, but the
# `country` field is the free-text "United States"; normalize the common
# names and fall back to the raw value otherwise (geocoder tolerates either).
_COUNTRY_NAME_TO_ISO2 = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "canada": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "australia": "AU",
    "new zealand": "NZ",
    "ireland": "IE",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }
    )
    return s


# ---------------------------------------------------------------------------
# Stage 1: paginated REST fetch
# ---------------------------------------------------------------------------

def fetch_directory_json(page: int = 1, per_page: int = 100) -> list[dict]:
    """Hit the GeoDirectory v2 listings endpoint and return the parsed list
    of listing dicts for a single page.

    Static UA + 20s timeout (mirror reference convention). Caller pages until
    fewer than ``per_page`` records come back.
    """
    s = _session()
    params = {"per_page": str(per_page), "page": str(page)}
    r = s.get(LISTINGS_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def fetch_all_directory_records(per_page: int = 100) -> list[dict]:
    """Page through the listings endpoint until a short/empty page and return
    the concatenated listing records."""
    out: list[dict] = []
    page = 1
    while True:
        batch = fetch_directory_json(page=page, per_page=per_page)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return out


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Return a stripped, HTML-unescaped string or None for missing/empty."""
    if val is None:
        return None
    if not isinstance(val, str):
        val = str(val)
    s = html_module.unescape(val).strip()
    return s or None


def _field_list(rec: dict, key: str) -> list[str]:
    """Read a GeoDirectory ``{raw, rendered[]}`` text-field into a clean list.

    Prefers ``rendered`` (already split); falls back to comma-splitting
    ``raw``. Returns [] for missing / None / blank values.
    """
    field = rec.get(key)
    if not isinstance(field, dict):
        return []
    rendered = field.get("rendered")
    items: list[str] = []
    if isinstance(rendered, list):
        items = [x for x in rendered if isinstance(x, str)]
    elif isinstance(field.get("raw"), str):
        items = field["raw"].split(",")
    return [s for s in (html_module.unescape(x).strip() for x in items) if s]


def _title(rec: dict) -> Optional[str]:
    title = rec.get("title")
    if isinstance(title, dict):
        return _coerce_str(title.get("raw") or title.get("rendered"))
    return _coerce_str(title)


def _format_credentials(degrees: list[str], licenses: list[str]) -> Optional[str]:
    """Combine degrees and licenses into one comma-separated string,
    de-duplicated, preserving order (degrees first)."""
    parts: list[str] = []
    for item in list(degrees) + list(licenses):
        if item and item not in parts:
            parts.append(item)
    if not parts:
        return None
    return ", ".join(parts)


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Add an https:// scheme if the listing gives a bare domain."""
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
    return _COUNTRY_NAME_TO_ISO2.get(raw.strip().lower())


def _is_fellowship(certifications: list[str]) -> bool:
    """True if the listing carries the CIPP / IPI Fellowship designation
    ("Certified Integrative Psychiatric Provider")."""
    return any(_FELLOWSHIP_MARKER in c.lower() for c in certifications)


def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: GeoDirectory listing dict -> NormalizedPractitionerRow.

    Returns None if the record lacks a usable name (defensive)."""
    name = _title(rec)
    if not name:
        return None

    degrees = _field_list(rec, "degrees")
    licenses = _field_list(rec, "licenses")
    certifications = _field_list(rec, "ipi_certifications")
    credentials = _format_credentials(degrees, licenses)

    practice = _coerce_str(rec.get("job_title"))

    phone = _coerce_str(rec.get("phone_number"))
    email = _coerce_str(rec.get("email"))
    website = _normalize_website(_coerce_str(rec.get("website")))

    address1 = _coerce_str(rec.get("street"))
    city = _coerce_str(rec.get("city"))
    state = _coerce_str(rec.get("region"))
    postal = _coerce_str(rec.get("zip"))
    country_raw = _coerce_str(rec.get("country"))
    country = _country_iso2(country_raw) or country_raw or "US"

    # The permalink is unique per listing and stable across re-runs; use it
    # directly as the dedup key. Fall back to an id-based URL if absent.
    source_url = _coerce_str(rec.get("link"))
    if not source_url:
        rid = _coerce_str(rec.get("id"))
        source_url = f"{BASE}/?p={rid}" if rid else None
    if not source_url:
        return None

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="IPI",
        source_url=source_url,
        fellowship_level=_is_fellowship(certifications),
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
    """Pure parser: takes a GeoDirectory listings response and returns one
    NormalizedPractitionerRow per usable record.

    Accepts either:
      - the list of listing dicts (the endpoint's native shape)
      - a dict wrapper containing a ``data`` list
      - a JSON string of either form
    No I/O.
    """
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
