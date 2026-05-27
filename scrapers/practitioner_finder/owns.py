"""Ocular Wellness & Nutrition Society (OWNS) scraper.

OWNS publishes its "Find a Practitioner" directory as a WordPress
``portfolio`` custom post type. Discovery 2026-05-26: the public WP REST
namespace exposes ``/wp-json/wp/v2/portfolio`` which returns the full set
of practitioner records (37 active as of discovery) with each record
carrying a fully-populated HTML ``content.rendered`` block that follows
the same labelled-line layout the site renders on the public page:

    <strong>Clinic Name:</strong> Eyecare of Lehi</p>
    <p><strong>Address:</strong> 75 W State St<br />
    <strong>City:</strong> Lehi<br />
    <strong>State:</strong> Utah<br />
    <strong>Zip Code:</strong> 84043</p>
    <p><strong>Phone: </strong>(801) 768-4100<br />
    <strong>Email:</strong> <a href="mailto:...">...</a></p>
    <p><strong>Website: </strong><a href="..." ...>...</a></p>

The labelled-line layout varies by country bucket:
  - US records use ``<strong>State:</strong>`` and ``<strong>Zip Code:</strong>``
  - Canadian records use ``<strong>Province:</strong>`` and ``<strong>Postal Code:</strong>``
    (some still use ``Zip Code:`` — both are accepted)
  - UK records use ``<strong>County:</strong>`` with the literal value
    "United Kingdom" (i.e. the County field is repurposed as country)
    and ``<strong>Zip Code:</strong>`` carrying the UK postcode.

Country detection therefore CANNOT rely on the label keyword alone.
Instead we use the ``portfolio_category`` taxonomy term IDs returned
on each record — the same IDs the site's state-dropdown navigates to.
Categories 7..56 are US states (slug 'alabama' .. 'wyoming'), 57 is
'canada', 58 is 'united-kingdom'. We resolve a record's country via
its primary category by fetching the portfolio_category taxonomy once
(``fetch_categories``) and building a slug-keyed map; the adapter
caller passes that map into the parser so parsing remains pure.

Endpoint:

    GET https://ocularnutritionsociety.org/wp-json/wp/v2/portfolio?per_page=100&page=N

Pagination is standard WP REST: ``X-WP-Total`` / ``X-WP-TotalPages``
headers give the totals. At per_page=100 the entire directory fits on
one page (37 < 100); the adapter pages defensively anyway in case the
directory grows.

Fellowship tier: OWNS confers "FOWNS" (Fellow of the Ocular Wellness
& Nutrition Society) as its top tier — three of 37 practitioners carry
it as of discovery (Kaleb Abbott, Daniel Walker, Mila Ioussifova). The
adapter sets fellowship_level=True when "FOWNS" appears in the title's
credential block.

OWNS straddles two parent categories in the Finder UI — Eye Care
(it's an optometrist society) AND Holistic Health (the focus is
nutrition). Every row therefore carries FOUR specialty tags:
``['nutritional_eye_care', 'eye_care', 'nutrition', 'holistic_health']``
so the Finder's filter chips surface OWNS practitioners under both
parent categories.

Output rows have tier='org_member', source_org='OWNS'. The
per-practitioner ``source_url`` is the WP canonical link
(``https://ocularnutritionsociety.org/practitioners/<slug>/``) with
``#<id>`` appended as a stable tiebreak fragment.
"""
import html as html_module
import re
import time
from typing import Optional

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://ocularnutritionsociety.org"
API_PORTFOLIO_URL = f"{BASE}/wp-json/wp/v2/portfolio"
API_PORTFOLIO_CATEGORY_URL = f"{BASE}/wp-json/wp/v2/portfolio_category"

# Cross-parent tagging: OWNS sits under BOTH Eye Care AND Holistic Health
# in the Finder UI, so every row gets all four tags. The Finder's filter
# chip logic relies on this exact set.
LOCKED_SPECIALTIES = ["nutritional_eye_care", "eye_care", "nutrition", "holistic_health"]

# Fellowship credential marker. FOWNS = Fellow, Ocular Wellness & Nutrition
# Society. Case-insensitive word-boundary match guards against a stray
# "FOWNS" prefix in unrelated tokens (none currently exist in the data,
# but the regex is defensive).
_FOWNS_RE = re.compile(r"\bFOWNS\b", re.IGNORECASE)

# US state-name -> ISO2 not needed; we keep the full state name in the
# `state` field to match sibling adapters (IABDM stores "Texas" etc.).
# Country resolution is by portfolio_category slug -> ISO2.
_CATEGORY_SLUG_TO_COUNTRY = {
    # Canada / UK get their own categories
    "canada": "CA",
    "united-kingdom": "GB",
    # All 50 US state slugs (matched against the dropdown state list on
    # the public Find a Practitioner page) -> 'US'. Listed explicitly
    # rather than using a default so a brand-new "mexico" category in
    # the future doesn't silently inherit 'US'.
    "alabama": "US", "alaska": "US", "arizona": "US", "arkansas": "US",
    "california": "US", "colorado": "US", "connecticut": "US",
    "delaware": "US", "florida": "US", "georgia": "US", "hawaii": "US",
    "idaho": "US", "illinois": "US", "indiana": "US", "iowa": "US",
    "kansas": "US", "kentucky": "US", "louisiana": "US", "maine": "US",
    "maryland": "US", "massachusetts": "US", "michigan": "US",
    "minnesota": "US", "mississippi": "US", "missouri": "US",
    "montana": "US", "nebraska": "US", "nevada": "US",
    "new-hampshire": "US", "new-jersey": "US", "new-mexico": "US",
    "new-york": "US", "north-carolina": "US", "north-dakota": "US",
    "ohio": "US", "oklahoma": "US", "oregon": "US", "pennsylvania": "US",
    "rhode-island": "US", "south-carolina": "US", "south-dakota": "US",
    "tennessee": "US", "texas": "US", "utah": "US", "vermont": "US",
    "virginia": "US", "washington": "US", "west-virginia": "US",
    "wisconsin": "US", "wyoming": "US",
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
# Stage 1: paginated WP REST fetch
# ---------------------------------------------------------------------------

def fetch_directory_page(page: int = 1, per_page: int = 100) -> tuple[list[dict], int]:
    """Fetch a single page from the WP ``portfolio`` REST endpoint.

    Returns ``(records, total_pages)`` — ``total_pages`` comes from the
    ``X-WP-TotalPages`` response header (0 if not present). Static UA +
    20s timeout + 0.5s sleep (rate-friendly).
    """
    s = _session()
    r = s.get(
        API_PORTFOLIO_URL,
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
    """Page through ``portfolio`` until exhausted; return concatenated records.

    Uses the ``X-WP-TotalPages`` header from page 1 to bound the walk, with
    a defensive empty-page break so a missing header (0) cannot infinite
    loop. As of discovery the entire 37-row directory fits on page 1 at
    per_page=100, but we walk defensively in case the directory grows.
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


def fetch_categories() -> list[dict]:
    """Fetch the ``portfolio_category`` taxonomy term list.

    Returns the full list of terms (52 as of discovery: 50 US states +
    Canada + United Kingdom). Used to build a category-id -> ISO2 country
    map for parsing. Static UA + 20s timeout + 0.5s sleep.
    """
    s = _session()
    r = s.get(
        API_PORTFOLIO_CATEGORY_URL,
        params={"per_page": 100},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list):
        payload = []
    time.sleep(0.5)
    return payload


def build_category_country_map(categories: list[dict]) -> dict[int, str]:
    """Build {term_id: ISO2-country} from a portfolio_category term list.

    Unknown slugs are omitted (rather than defaulting to 'US') so the
    parser can fall back to its own default when a record's primary
    category is missing or new. Defensive: tolerates terms missing
    'id' or 'slug'.
    """
    out: dict[int, str] = {}
    for term in categories or []:
        if not isinstance(term, dict):
            continue
        tid = term.get("id")
        slug = term.get("slug")
        if not isinstance(tid, int) or not isinstance(slug, str):
            continue
        country = _CATEGORY_SLUG_TO_COUNTRY.get(slug.lower())
        if country:
            out[tid] = country
    return out


# ---------------------------------------------------------------------------
# Parsing helpers (pure)
# ---------------------------------------------------------------------------

def _coerce_str(val) -> Optional[str]:
    """Return a stripped string or None for missing/empty/null values.

    Accepts already-string values as well as the wrapped ``{rendered, ...}``
    dicts the WP REST API returns for title / content fields. Always
    strips the zero-width-space (U+200B) the OWNS titles trail with —
    every record's title ends with "…O.D.​" or similar."""
    if val is None:
        return None
    if isinstance(val, dict):
        # Prefer 'rendered' for WP REST title/content dicts; fall back to
        # 'raw' which IABDM-style records sometimes use.
        return _coerce_str(val.get("rendered")) or _coerce_str(val.get("raw"))
    if isinstance(val, str):
        s = val.replace("​", "").strip()
        return s or None
    s = str(val).strip()
    return s or None


def _strip_credentials(name: str) -> tuple[str, Optional[str]]:
    """Split a 'Walker Shaffer, O.D.' title into (clean_name, credentials).

    Mirrors the IABDM helper. The OWNS title carries the name and a
    comma-separated credential string in formats like:
      "Walker Shaffer, O.D."
      "Kaleb Abbott, OD, MS, FAAO, FOWNS"
      "Dorothy Hitchmoth, OD, FAAO, ABCMO, ABO Dipl."
      "Dr. Ingryd Lorenzana, FAAO, FOVDRA, CBHP - Neuro-Optometrist"
      "Kenneth Daniels, OD FAAO Diplomate American Board of Optometry"
        (space-separated extra titles after the first comma chunk —
         we keep the whole credential tail in this case)
    Honorifics ("Dr.") are preserved on the name. Returns the name with
    trailing punctuation stripped plus the credential tail (or None when
    no comma is present).
    """
    if not name:
        return "", None
    s = name.strip()
    # First credential token after a comma. Allow internal dots/slashes.
    # The OWNS data uses 2-char credentials like "OD" / "MS" frequently
    # (every optometrist has an OD), so the inner-letter requirement is
    # relaxed vs IABDM's regex (which only had to handle 3+ char DDS / DMD
    # style abbreviations). The trailing class accepts a dot so "O.D."
    # parses as a single credential token, and the inner class accepts
    # dots/slashes for compounds like "M.D./PhD".
    cred_pat = re.compile(r",\s*([A-Za-z][A-Za-z./]*[A-Za-z.])")
    m = cred_pat.search(s)
    if not m:
        return s.rstrip(", "), None
    clean = s[: m.start()].strip().rstrip(",")
    creds = s[m.start():].lstrip(", ").strip().rstrip(",").rstrip()
    return clean, creds or None


def _name_from_title(title_raw: str) -> str:
    """Pull the practitioner name out of the title."""
    if not title_raw:
        return ""
    s = html_module.unescape(title_raw).replace("​", "").strip()
    name, _creds = _strip_credentials(s)
    return name


def _credentials_from_title(title_raw: str) -> Optional[str]:
    """Pull just the trailing credential string out of the title."""
    if not title_raw:
        return None
    s = html_module.unescape(title_raw).replace("​", "").strip()
    _name, creds = _strip_credentials(s)
    return creds


def _is_fellowship_title(credentials: Optional[str]) -> bool:
    """True when the credential string includes "FOWNS" (Fellow OWNS).

    FOWNS is OWNS's top tier — a peer-reviewed credential. Per the
    cross-adapter convention ("Fellow / Diplomate / Master qualifies"),
    holding FOWNS marks the record as fellowship-tier."""
    if not credentials:
        return False
    return bool(_FOWNS_RE.search(credentials))


def _normalize_website(raw: Optional[str]) -> Optional[str]:
    """Add an https:// scheme if a website is listed as a bare domain."""
    s = _coerce_str(raw)
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return f"https://{s}"


# Labelled-line extractor. Each record's content.rendered body packs the
# contact fields as <strong>Label:</strong> Value blocks separated by
# <br /> or </p><p>. The regex matches any of the known label variants
# (US "State", Canada "Province", UK "County" all serve as the
# region-line label) and captures the value up to the next <br>, </p>,
# or <strong> sibling.
_FIELD_RE = re.compile(
    r"<strong>\s*(?P<label>[A-Za-z ]+?)\s*:?\s*</strong>\s*"
    r"(?P<value>.*?)"
    r"(?=<br\s*/?>|</p\s*>|<strong)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_fields(content_html: str) -> dict[str, str]:
    """Pull labelled fields out of a portfolio content body.

    Returns a flat lower-cased label -> stripped-text dict. Values are
    HTML-unescaped but inner tags (anchors etc.) are kept so callers can
    pick href out of email/website fields separately. Multiple labels
    with the same name collapse to the first occurrence (the OWNS layout
    never repeats a label inside a single record)."""
    out: dict[str, str] = {}
    if not content_html:
        return out
    for m in _FIELD_RE.finditer(content_html):
        label = m.group("label").strip().lower()
        if label in out:
            continue
        value_html = m.group("value").strip()
        # Strip HTML tags for plain-text view but keep anchors readable
        # by grabbing the visible text. For email/website we re-extract
        # the href below; here we just want the display text.
        text = re.sub(r"<[^>]+>", "", value_html)
        text = html_module.unescape(text).replace("​", "").strip()
        text = text.rstrip(",").strip()
        if text:
            out[label] = text
        # Preserve the raw HTML alongside the plain text so href fields
        # (email, website) can be parsed for the real URL by callers.
        out[f"{label}__html"] = value_html
    return out


def _href_from_field(field_html: Optional[str], scheme: Optional[str] = None) -> Optional[str]:
    """Pull the first ``href`` out of a label-value html chunk.

    Used for email (scheme='mailto') and website (scheme=None) values
    where the OWNS layout wraps the canonical URL in an <a>. Returns
    None when no anchor is present or the href doesn't start with the
    requested scheme."""
    if not field_html:
        return None
    m = re.search(r'href=["\']([^"\']+)["\']', field_html)
    if not m:
        return None
    href = html_module.unescape(m.group(1)).strip()
    if scheme:
        if not href.lower().startswith(scheme):
            return None
        return href[len(scheme):].strip() if scheme.endswith(":") else href
    return href


def _resolve_country(rec: dict, category_country: dict[int, str]) -> str:
    """Decide ISO2 country for a record using its portfolio_category list.

    Falls back to 'US' when the record has no recognized category — the
    overwhelming majority of OWNS practitioners are US-based, and the
    site's state dropdown puts US states at the top, so 'US' is the
    safest default for a brand-new uncategorized record. Pass an empty
    map to make the parser unconditionally default to 'US'."""
    cats = rec.get("portfolio_category") or []
    if isinstance(cats, list):
        for cid in cats:
            if isinstance(cid, int) and cid in category_country:
                return category_country[cid]
    return "US"


def _build_source_url(rec: dict) -> str:
    """Stable per-practitioner URL.

    Prefer the WP REST canonical ``link`` (it's the on-site detail page);
    fall back to ``/practitioners/<slug>/`` and finally the numeric id
    when the slug is missing too. Always append ``#<id>`` so two entries
    that accidentally share a slug (rare but possible) still produce
    distinct upsert keys."""
    link = _coerce_str(rec.get("link"))
    slug = _coerce_str(rec.get("slug"))
    rid = _coerce_str(rec.get("id"))
    if link:
        base = link.rstrip("/")
    elif slug:
        base = f"{BASE}/practitioners/{slug}"
    else:
        base = f"{BASE}/practitioners/practitioner-{rid or 'unknown'}"
    if rid:
        return f"{base}/#{rid}"
    return f"{base}/"


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def _record_to_row(
    rec: dict, category_country: dict[int, str]
) -> Optional[NormalizedPractitionerRow]:
    """Pure transformation: portfolio dict -> NormalizedPractitionerRow.

    Returns None when no usable name can be recovered from the title.
    """
    title_raw = _coerce_str(rec.get("title")) or ""
    name = _name_from_title(title_raw)
    if not name:
        return None
    credentials = _credentials_from_title(title_raw)

    content_raw = rec.get("content")
    if isinstance(content_raw, dict):
        content_html = content_raw.get("rendered") or content_raw.get("raw") or ""
    elif isinstance(content_raw, str):
        content_html = content_raw
    else:
        content_html = ""

    fields = _extract_fields(content_html)

    practice = fields.get("clinic name")
    # Solo-listing duplicate: when the clinic name matches the practitioner
    # name (sometimes happens for self-employed listings), suppress the dup.
    if practice and practice.lower() == name.lower():
        practice = None

    address1 = fields.get("address")
    city = fields.get("city")
    # Region line varies by country: US='State', Canada='Province', UK='County'
    state = fields.get("state") or fields.get("province") or fields.get("county")
    # UK records use 'County' to hold the literal "United Kingdom" string —
    # that's a country, not a region. Drop it from the state slot when it
    # equals a known country name so the geocoder doesn't get confused.
    if state and state.lower() in {"united kingdom", "uk", "canada"}:
        state = None
    postal = fields.get("zip code") or fields.get("postal code")
    phone = fields.get("phone")
    email = (
        _href_from_field(fields.get("email__html"), scheme="mailto:")
        or fields.get("email")
    )
    if email and ":" in email and email.lower().startswith("mailto:"):
        email = email.split(":", 1)[1].strip()
    website = _normalize_website(
        _href_from_field(fields.get("website__html"))
        or fields.get("website")
    )

    country = _resolve_country(rec, category_country)

    return NormalizedPractitionerRow(
        tier="org_member",
        name=name,
        specialties=list(LOCKED_SPECIALTIES),
        source_org="OWNS",
        source_url=_build_source_url(rec),
        fellowship_level=_is_fellowship_title(credentials),
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


def parse_directory_json(
    payload, category_country: Optional[dict[int, str]] = None
) -> list[NormalizedPractitionerRow]:
    """Pure parser: takes a WP ``portfolio`` response (list of dicts, or
    JSON string of one) and returns one NormalizedPractitionerRow per
    usable record.

    ``category_country`` is an optional {term_id: ISO2} map built from
    ``build_category_country_map(fetch_categories())``. When omitted,
    every record defaults to country='US' (the overwhelming majority of
    OWNS practitioners are US-based, but Canadian and UK records will
    misreport country until the caller passes the real map).

    No I/O.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        import json
        payload = json.loads(payload)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("data") or []
    else:
        return []

    cat_map = category_country or {}
    rows: list[NormalizedPractitionerRow] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        row = _record_to_row(rec, cat_map)
        if row is not None:
            rows.append(row)
    return rows
