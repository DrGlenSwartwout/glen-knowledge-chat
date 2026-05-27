"""International Academy of Biological Dentistry and Medicine (IABDM) scraper.

IABDM publishes its directory through the GeoDirectory WordPress plugin's
REST API. Discovery 2026-05-26: the public WP/REST namespace exposes
``/wp-json/geodir/v2/places`` (custom post type ``gd_place``) which returns
fully-populated practitioner records with no per-listing fetch needed.

Endpoint:

    GET https://iabdm.org/wp-json/geodir/v2/places?per_page=100&page=N

Pagination is standard WP REST: ``X-WP-Total`` / ``X-WP-TotalPages`` headers
give the totals (471 active places / 5 pages as of 2026-05-26). Walking
``page=1..total_pages`` covers every record.

Each record is a flat dict with the relevant fields:

  id                          -> stable numeric WP post id (dedup key)
  slug                        -> URL slug, e.g. ``teresa-scott-dds``
  link                        -> canonical detail URL
                                 ``https://iabdm.org/places/<cat>/<slug>/``
  title.rendered              -> "First M. Last, DDS, MIABDM, ..." (fallback
                                 source for name + credentials when the
                                 separate member_* fields are blank, which
                                 is common for older / international entries)
  member_first_name           -> "Teresa"      (may be blank)
  member_last_name            -> "Scott"       (may be blank)
  member_position             -> "DDS"|"DMD"|"RDH"|"RDA" (may be blank)
  office_name                 -> practice name (may be blank or null)
  street                      -> address line(s) raw
  city / region / zip / country
  phone / email / website
  default_category            -> '1605' Dentist | '1606' Hygienist |
                                 '1866' Office Manager | '1607' Other
  certified_master            -> {raw: 'yes'|None}  Master IABDM
  is_fellow_member            -> {raw: 'yes'|None}  Fellow IABDM
  precertified                -> {raw: 'yes'|'no'|None}
  certifiedmember             -> {raw: 'yes'|'no'} basic Certified Member
  active_member / is_sponsor / claimed / medical_office (informational)
  latitude / longitude        -> already-geocoded coordinates (ignored here;
                                 the shared geocoder owns lat/lng to keep
                                 the quality field consistent across adapters)

Output rows have tier='org_member', source_org='IABDM',
specialties=['biological', 'dental']. The IABDM membership ladder is
Precertified -> Certified Member -> Fellow (FIABDM) -> Master (MIABDM);
per the fellowship-level convention used in the IAOMT adapter
("Accredited / Master / Fellow qualifies"), the IABDM analogue of
IAOMT's Accredited Member tier is IABDM's Certified Member tier — both
are the entry-level vetted-credentials tier. So Certified Member OR
Fellow OR Master gets fellowship_level=True. Precertified-only and
hygienist-only entries (no vetted clinical tier) do not.

The per-practitioner ``source_url`` is the GeoDirectory canonical link
(or, when missing, synthesized from slug + numeric id) and is stable
across re-runs.
"""
import html as html_module
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://iabdm.org"
API_PLACES_URL = f"{BASE}/wp-json/geodir/v2/places"

LOCKED_SPECIALTIES = ["biological", "dental"]

# Country-name -> ISO2 (same convention as iaomt.py). IABDM is dental so the
# spread is similar but includes a few extras seen in the wild.
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


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )
    return s


# ---------------------------------------------------------------------------
# Stage 1: paginated GeoDirectory REST fetch
# ---------------------------------------------------------------------------

def fetch_directory_page(page: int = 1, per_page: int = 100) -> tuple[list[dict], int]:
    """Fetch a single page from the GeoDirectory ``places`` REST endpoint.

    Returns ``(records, total_pages)`` — ``total_pages`` comes from the
    ``X-WP-TotalPages`` response header (0 if not present). Static UA +
    20s timeout + 0.5s sleep (rate-friendly).
    """
    s = _session()
    r = s.get(
        API_PLACES_URL,
        params={"per_page": per_page, "page": page},
        timeout=20,
    )
    r.raise_for_status()
    batch = r.json()
    if not isinstance(batch, list):
        batch = []
    try:
        total_pages = int(r.headers.get("X-WP-TotalPages", "0"))
    except (TypeError, ValueError):
        total_pages = 0
    time.sleep(0.5)
    return batch, total_pages


def fetch_all_directory_records() -> list[dict]:
    """Page through ``places`` until exhausted; return concatenated records.

    Uses the ``X-WP-TotalPages`` header from page 1 to bound the walk, with
    a defensive empty-page break so a missing header (0) cannot infinite
    loop.
    """
    out: list[dict] = []
    page = 1
    per_page = 100
    batch, total_pages = fetch_directory_page(page=page, per_page=per_page)
    out.extend(batch)
    if not batch:
        return out
    page += 1
    while page <= total_pages or total_pages == 0:
        batch, _ = fetch_directory_page(page=page, per_page=per_page)
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Return a stripped string or None for missing/empty/null values.

    Accepts already-string values as well as the wrapped ``{raw, rendered}``
    dicts the GeoDirectory API returns for many fields — we always prefer
    ``raw`` because it carries the canonical value (e.g. 'yes' / None)
    instead of the localized display ('Yes' / 'No')."""
    if val is None:
        return None
    if isinstance(val, dict):
        return _coerce_str(val.get("raw"))
    if isinstance(val, str):
        s = val.strip()
        return s or None
    s = str(val).strip()
    return s or None


def _is_yes(val) -> bool:
    """True when a GeoDirectory yes/no field unwraps to literal 'yes'."""
    s = _coerce_str(val)
    return s is not None and s.lower() == "yes"


def _strip_credentials(name: str) -> tuple[str, Optional[str]]:
    """Split a 'Dr. Jane Doe, DDS, FIABDM, ...' title into (clean_name, credentials).

    Mirrors the OEPF helper's intent: credentials are the trailing
    comma-separated short uppercase abbreviations (with optional dots),
    optionally wrapped in parens like ``(DMD)``. Honorifics (``Dr.``,
    ``Dra.``) are preserved on the name.
    """
    if not name:
        return "", None
    s = name.strip()
    # "Anca Condur (DDS)" -> "Anca Condur, DDS"
    paren = re.match(r"^(.*?)\s*\(([A-Za-z][A-Za-z.,\s/-]*)\)\s*$", s)
    if paren:
        s = f"{paren.group(1).strip()}, {paren.group(2).strip()}"

    # First credential token after a comma. Allow internal dots/slashes.
    cred_pat = re.compile(r",\s*([A-Za-z][A-Za-z./]{1,}[A-Za-z])")
    m = cred_pat.search(s)
    if not m:
        return s.rstrip(", "), None
    # Heuristic: a "credential" token is mostly uppercase letters (the
    # IABDM directory uses DDS, DMD, MIABDM, FIABDM, AIAOMT, FICOI,
    # FAGD, MAGD, ...). If the first token after the comma is title-case
    # (e.g. "Certified Biological Dentist"), treat THAT as still part of
    # the trailing-title block too — chop everything from that comma on.
    clean = s[: m.start()].strip().rstrip(",")
    creds = s[m.start():].lstrip(", ").strip().rstrip(",").rstrip()
    return clean, creds or None


def _name_from_title(title_html: str) -> str:
    """Pull a name out of the rendered title (used when the structured
    member_first_name/member_last_name fields are blank — international
    entries are mostly in this shape)."""
    if not title_html:
        return ""
    s = html_module.unescape(title_html).strip()
    name, _creds = _strip_credentials(s)
    return name


def _credentials_from_title(title_html: str) -> Optional[str]:
    """Pull just the trailing credential string out of the title."""
    if not title_html:
        return None
    s = html_module.unescape(title_html).strip()
    _name, creds = _strip_credentials(s)
    return creds


# Standard set of dentist/doctor degree credentials we expect to see in the
# title block when the listed practitioner is the dentist themselves (as
# opposed to a hygienist registering on behalf of the office). When ANY of
# these appears in the title-extracted credentials we trust the title and
# DROP member_position — because member_position is the account holder's
# role (often RDH) which can be different from the listed practitioner
# (often DDS / DMD).
_DEGREE_CREDENTIALS = {
    "DDS", "DMD", "DO", "MD", "ND", "MS", "MSC", "MBCHB", "BDS",
    "BDENT", "PHD", "DC", "DPM", "DSC", "MBBS", "DABO", "DABT",
}


def _title_has_degree(title_creds: Optional[str]) -> bool:
    """True when the title-extracted credential list includes a dentist /
    doctor degree (DDS, DMD, MD, etc.). Used to suppress the bogus
    'RDH, DDS, ...' concatenation when member_position is the account
    holder's role and the title is the actual practitioner's credentials."""
    if not title_creds:
        return False
    for chunk in title_creds.split(","):
        token = chunk.strip().rstrip(".").upper()
        # Strip any trailing punctuation/parens defensively.
        token = re.sub(r"[^A-Z]", "", token)
        if token in _DEGREE_CREDENTIALS:
            return True
    return False


def _build_full_name(first: Optional[str], last: Optional[str]) -> str:
    parts = [p.strip() for p in (first, last) if p and p.strip()]
    return " ".join(parts)


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Add an https:// scheme if the directory entry lists a bare domain."""
    s = _coerce_str(raw)
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return f"https://{s}"


def _country_iso2(raw: Optional[str]) -> Optional[str]:
    """Map a free-text country name to ISO2; None if unrecognized."""
    s = _coerce_str(raw)
    if not s:
        return None
    return _COUNTRY_NAME_TO_ISO2.get(s.lower())


def _is_fellowship(record: dict) -> bool:
    """True when the record is at IABDM's Certified Member tier or above.

    IABDM's membership ladder is Precertified -> Certified Member ->
    Fellow (FIABDM) -> Master (MIABDM). Per the cross-adapter spec
    ("Accredited Member or Master or Fellow qualifies"), IABDM's
    Certified Member is the structural analogue of IAOMT's Accredited
    Member — the entry-level vetted-credentials tier — so it counts.

    Qualifies: Certified Member OR Fellow OR Master.
    Does NOT qualify: Precertified-only, hygienist-only, or no tier at
    all. The Certified Member field name in the GeoDirectory payload is
    ``certifiedmember`` (no underscore); we also accept ``certified_member``
    in case the API normalization ever changes."""
    return (
        _is_yes(record.get("certified_master"))
        or _is_yes(record.get("is_fellow_member"))
        or _is_yes(record.get("certifiedmember"))
        or _is_yes(record.get("certified_member"))
    )


def _build_source_url(rec: dict) -> str:
    """Stable per-practitioner URL.

    Prefer the API's canonical ``link`` (it's the on-site detail page);
    fall back to ``/places/<slug>/`` and finally to the numeric id when
    the slug is missing too. Always append ``#<id>`` so two entries that
    accidentally share a slug (rare but possible) still produce distinct
    upsert keys."""
    link = _coerce_str(rec.get("link"))
    slug = _coerce_str(rec.get("slug"))
    rid = _coerce_str(rec.get("id"))
    if link:
        base = link.rstrip("/")
    elif slug:
        base = f"{BASE}/places/{slug}"
    else:
        base = f"{BASE}/places/place-{rid or 'unknown'}"
    if rid:
        return f"{base}/#{rid}"
    return f"{base}/"


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def _record_to_row(rec: dict) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: GeoDirectory ``gd_place`` dict -> NormalizedPractitionerRow.

    Returns None when no usable name can be recovered from either the
    structured member fields OR the post title.
    """
    first = _coerce_str(rec.get("member_first_name"))
    last = _coerce_str(rec.get("member_last_name"))
    structured_name = _build_full_name(first, last)

    title_raw = rec.get("title")
    title_html = None
    if isinstance(title_raw, dict):
        title_html = _coerce_str(title_raw.get("rendered")) or _coerce_str(
            title_raw.get("raw")
        )
    else:
        title_html = _coerce_str(title_raw)

    title_creds = _credentials_from_title(title_html or "")

    if structured_name:
        name = structured_name
        # member_position is the *account holder's* role on the registration
        # form (often "RDH" — a hygienist registering on behalf of the
        # dentist). When the post title already carries dentist/doctor
        # degree credentials (DDS, DMD, MD, ...) we trust the title — it
        # describes the actual listed practitioner — and DROP
        # member_position to avoid emitting anatomically-impossible
        # "RDH, DDS, ..." credential lists. Only when the title has NO
        # degree credentials do we fall back to member_position (covers
        # the legitimate hygienist-only listing case).
        position = _coerce_str(rec.get("member_position"))
        parts: list[str] = []
        if position and not _title_has_degree(title_creds):
            parts.append(position)
        if title_creds:
            for chunk in title_creds.split(","):
                chunk = chunk.strip()
                if chunk and chunk.lower() not in {p.lower() for p in parts}:
                    parts.append(chunk)
        credentials = ", ".join(parts) if parts else None
    else:
        name = _name_from_title(title_html or "")
        credentials = title_creds

    if not name:
        return None

    practice = _coerce_str(rec.get("office_name"))
    # If practice matches practitioner name (solo entry where the office
    # field was filled with the practitioner's name), suppress the dup.
    if practice and practice.lower() == name.lower():
        practice = None

    phone = _coerce_str(rec.get("phone")) or _coerce_str(rec.get("mobile"))
    email = _coerce_str(rec.get("email"))
    website = _normalize_website(rec.get("website"))

    address1 = _coerce_str(rec.get("street"))
    city = _coerce_str(rec.get("city"))
    state = _coerce_str(rec.get("region"))
    postal = _coerce_str(rec.get("zip"))
    country_raw = _coerce_str(rec.get("country"))
    country = _country_iso2(country_raw) or country_raw or "US"

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="IABDM",
        source_url=_build_source_url(rec),
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


def parse_directory_json(payload) -> list[NormalizedPractitionerRow]:
    """Pure parser: takes a GeoDirectory ``places`` response (list of dicts,
    or a JSON string of one) and returns one NormalizedPractitionerRow per
    usable record. No I/O."""
    if isinstance(payload, (str, bytes, bytearray)):
        import json
        payload = json.loads(payload)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        # Defensive: a wrapped {'data': [...]} shape would come from some
        # other GeoDirectory plugin variant; accept it.
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
