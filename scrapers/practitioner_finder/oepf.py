"""Optometric Extension Program Foundation (OEPF) directory scraper.

OEPF is a WordPress ListingPro site. Two-stage discovery:

1. WP REST API (cheap, paginated, returns title/slug/link/content + total count):
     https://www.oepf.org/wp-json/wp/v2/listing?per_page=100&page=N
   X-WP-Total header gives total record count (183 as of 2026-05-26).

2. Per-listing HTML page (rich: address, phone, website, doctor name,
   credentials, email, fellowship level, 2nd-doctor block):
     https://www.oepf.org/listing/<slug>/

HTML structure (discovered 2026-05-26):
  - <div class="listing-detail-infos"> holds <li class="lp-details-address">,
    <li class="lp-listing-phone">, <li class="lp-user-web">.
  - <div class="features-listing extra-fields"> holds <li> rows where each
    row has <strong>Label:</strong><span>Value</span>. Labels include:
        Doctor's Name, Doctor's email, About the Doctor,
        2nd Doctor/Therapist's Name, 2nd Doctor/Therapist's Email, ...
    NOTE the curly apostrophe (U+2019) in the label text.

Output rows have tier='org_member', source_org='OEPF',
specialties=['functional', 'eye_care']. Fellowship-level practitioners
(F.C.O.V.D. / FCOVD in credentials) get fellowship_level=True.

Multi-doctor listings produce one row per doctor, with source_url
suffixed '#doctor-1', '#doctor-2', etc. to keep upsert ON CONFLICT
(source_url) idempotent."""
import html as html_module
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605"
BASE = "https://www.oepf.org"
API_LISTING_URL = f"{BASE}/wp-json/wp/v2/listing"

LOCKED_SPECIALTIES = ["functional", "eye_care"]

# Address parsing — practitioners are global. Address strings vary widely:
#   "8950 Villa La Jolla Drive, Ste B128, La Jolla, CA, USA 92037"
#   "56-1400 Cowichan Bay Road, Cobble Hill, BC, Canada"
#   "335 Park Avenue"
# We attempt to pull out city / state / postal / country tail when present
# but leave fields None when ambiguous.

_US_STATE_ABBR_SET = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

# Look for "..., <CITY>, <ST>, USA <ZIP>" or "..., <CITY>, <ST> <ZIP>"
_US_ADDRESS_TAIL_RE = re.compile(
    r",\s*([A-Za-z][\w\s.'-]*?)\s*,\s*([A-Z]{2})(?:\s*,\s*USA?)?\s*(\d{5}(?:-\d{4})?)?\s*$"
)

# Fellowship marker. F.C.O.V.D. (Fellow, College of Optometrists in Vision
# Development) is the canonical fellowship designation. Match with or
# without dots, case insensitive.
_FELLOWSHIP_RE = re.compile(r"\bF\.?C\.?O\.?V\.?D\.?\b", re.IGNORECASE)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/json"}
    )
    return s


# ---------------------------------------------------------------------------
# Stage 1: WP REST API discovery
# ---------------------------------------------------------------------------

def fetch_listing_index() -> list[dict]:
    """Page through the WP REST API and return a list of
    {id, slug, link, title} dicts — one per OEPF listing.

    Static UA + 0.5s sleep between API page fetches (rate-friendly).
    Total record count comes from X-WP-Total response header; we walk
    page=1..N at per_page=100."""
    s = _session()
    out: list[dict] = []
    page = 1
    per_page = 100
    while True:
        r = s.get(
            API_LISTING_URL,
            params={"per_page": per_page, "page": page},
            timeout=20,
        )
        r.raise_for_status()
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        for rec in batch:
            out.append(
                {
                    "id": rec.get("id"),
                    "slug": rec.get("slug"),
                    "link": rec.get("link"),
                    "title": html_module.unescape(
                        (rec.get("title") or {}).get("rendered", "") or ""
                    ),
                }
            )
        if len(batch) < per_page:
            break
        page += 1
        time.sleep(0.5)
    return out


# ---------------------------------------------------------------------------
# Stage 2: per-listing HTML fetch
# ---------------------------------------------------------------------------

def fetch_directory_listing_html(url: str) -> str:
    """Hit a single OEPF listing detail page; return raw HTML.
    Static UA + 0.5s sleep."""
    s = _session()
    r = s.get(url, timeout=20)
    r.raise_for_status()
    time.sleep(0.5)
    return r.text


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _normalize_label(s: str) -> str:
    """Normalize 'Doctor’s email:' -> "doctor's email".
    Strips trailing colon, lowercases, and replaces curly apostrophes with ASCII."""
    if not s:
        return ""
    return s.replace("’", "'").strip().rstrip(":").lower()


def _parse_extra_fields(soup: BeautifulSoup) -> dict[str, str]:
    """Pull the <div class='features-listing extra-fields'> key/value rows
    into a flat dict keyed by normalized label."""
    out: dict[str, str] = {}
    for div in soup.find_all("div", class_="features-listing"):
        classes = div.get("class") or []
        if "extra-fields" not in classes:
            continue
        for li in div.find_all("li"):
            strong = li.find("strong")
            span = li.find("span")
            if not strong:
                continue
            key = _normalize_label(strong.get_text(strip=True))
            val = span.get_text(strip=True) if span else ""
            if key:
                out[key] = val
    return out


def _parse_detail_infos(soup: BeautifulSoup) -> dict[str, str]:
    """Pull the <div class='listing-detail-infos'> address / phone / website
    block into a dict with keys 'address', 'phone', 'website' (any may
    be missing)."""
    out: dict[str, str] = {}
    det = soup.find("div", class_=lambda c: bool(c) and "listing-detail-infos" in c)
    if not det:
        return out
    for li in det.find_all("li"):
        cls_list = li.get("class") or []
        cls = cls_list[0] if cls_list else ""
        if cls == "lp-details-address":
            # Structure: <a><span class="cat-icon">[icon]</span><span>[address]</span></a>
            # Skip the cat-icon span; take the address-bearing span (the second one,
            # or fall back to the full <a> text).
            a = li.find("a")
            target = a if a else li
            spans = target.find_all("span")
            addr_text = None
            for sp in spans:
                if "cat-icon" in (sp.get("class") or []):
                    continue
                txt = sp.get_text(strip=True)
                if txt:
                    addr_text = txt
                    break
            if not addr_text:
                addr_text = target.get_text(separator=" ", strip=True)
            if addr_text:
                out["address"] = addr_text
        elif cls == "lp-listing-phone":
            a = li.find("a")
            if a:
                href = a.get("href") or ""
                if href.startswith("tel:"):
                    out["phone"] = href[len("tel:"):].strip()
                else:
                    span = a.find("span")
                    if span:
                        out["phone"] = span.get_text(strip=True)
        elif cls == "lp-user-web":
            a = li.find("a")
            if a:
                href = a.get("href") or ""
                if href:
                    out["website"] = href.strip()
                else:
                    span = a.find("span")
                    if span:
                        out["website"] = span.get_text(strip=True)
    return out


def _split_address(raw: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Best-effort split: (address1, city, state, postal, country).

    For US addresses ending '..., City, ST [, USA] [zip]' we extract
    city/state/postal and derive country='US'. For international /
    partial addresses we keep the full string in address1 and only set
    country if recognizable (Canada, India, ...).

    The raw stays in address1 when we can't confidently split — geocoder
    will then handle it.
    """
    if not raw:
        return None, None, None, None, None
    raw = raw.strip()

    m = _US_ADDRESS_TAIL_RE.search(raw)
    if m:
        city = m.group(1).strip()
        state = m.group(2).strip()
        postal = m.group(3).strip() if m.group(3) else None
        if state in _US_STATE_ABBR_SET:
            head = raw[: m.start()].rstrip(", ").strip() or None
            return head, city, state, postal, "US"

    # International tail detection
    lower = raw.lower()
    if lower.endswith("canada") or ", canada" in lower or ",canada" in lower:
        return raw, None, None, None, "CA"
    if lower.endswith("india") or ", india" in lower or ",india" in lower:
        return raw, None, None, None, "IN"
    if lower.endswith("uk") or lower.endswith("united kingdom"):
        return raw, None, None, None, "GB"
    if lower.endswith("australia"):
        return raw, None, None, None, "AU"

    # Couldn't confidently split — keep whole raw as address1, country None
    return raw, None, None, None, None


def _extract_credentials(doctor_name: str) -> tuple[str, Optional[str]]:
    """Given a doctor field like 'Claude Valenti, O.D., F.C.O.V.D.' or
    'Dr. Angela Dobson' or 'Brian Thamel', return (clean_name, credentials).

    Credentials are anything matching the post-name suffix of uppercase
    abbreviations separated by commas (O.D., MD, FCOVD, MS, etc.).
    The name retains 'Dr.' / 'Dr' titles if present."""
    if not doctor_name:
        return "", None
    name = doctor_name.strip()
    # Some OEPF entries use a period after the name instead of a comma:
    # "Claude Valenti. O.D., F.C.O.V.D." -> treat the first ". " separator
    # the same as ", " for credential split.
    norm = re.sub(r"\.\s+(?=[A-Z][A-Z.]{0,6})", ", ", name)

    # Credentials pattern: comma then short uppercase abbrev (with optional dots)
    cred_pat = re.compile(r",\s*([A-Z][A-Z.]{1,8})")
    m = cred_pat.search(norm)
    if not m:
        return name.rstrip(". "), None
    clean_name = norm[: m.start()].strip().rstrip(". ")
    creds_part = norm[m.start():].lstrip(", ").strip()
    # Only strip trailing whitespace from creds — keep the terminal period
    # because designations like F.C.O.V.D. end in one and we want to preserve
    # the canonical form.
    creds_part = creds_part.rstrip()
    return clean_name, creds_part or None


def _is_fellowship(credentials: Optional[str], full_name: Optional[str] = None) -> bool:
    """True if F.C.O.V.D. (in any spacing/case) appears in credentials or
    the original name string."""
    for s in (credentials, full_name):
        if s and _FELLOWSHIP_RE.search(s):
            return True
    return False


def _extract_practice_name(soup: BeautifulSoup) -> Optional[str]:
    """The H1 of a listing page is the practice / business name."""
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True) or None
    return None


def _extract_source_url(soup: BeautifulSoup, fallback_url: Optional[str]) -> Optional[str]:
    """Prefer the canonical OG URL meta tag; fall back to the URL the
    caller passed in (or the page's canonical link)."""
    for prop, attr in (("og:url", "content"), ("canonical", "href")):
        if prop == "canonical":
            tag = soup.find("link", rel="canonical")
            if tag and tag.get("href"):
                return tag["href"].strip()
        else:
            tag = soup.find("meta", attrs={"property": prop})
            if tag and tag.get(attr):
                return tag[attr].strip()
    return fallback_url


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def parse_directory_listing_html(
    html: str,
    source_url: Optional[str] = None,
) -> list[NormalizedPractitionerRow]:
    """Parse a single OEPF listing detail page into 1+ NormalizedPractitionerRow.

    Returns one row per doctor (primary + optional '2nd Doctor/Therapist').
    Pure: no I/O. The ``source_url`` arg is the URL the page was fetched
    from; we use it as the base for per-doctor source_url, suffixed with
    '#doctor-1' (and '#doctor-2' if present). If omitted, we try the page's
    canonical/og:url meta tag.
    """
    soup = BeautifulSoup(html, "html.parser")
    extras = _parse_extra_fields(soup)
    contact = _parse_detail_infos(soup)
    practice_name = _extract_practice_name(soup)
    base_url = _extract_source_url(soup, source_url)
    if not base_url:
        # synthesize per the spec for stable dedup even when no canonical
        slug = re.sub(r"[^a-z0-9]+", "-", (practice_name or "unknown").lower()).strip("-")
        base_url = f"{BASE}/find-a-doctor#{slug}"

    address_raw = contact.get("address")
    address1, city, state, postal, country = _split_address(address_raw or "")
    phone = contact.get("phone") or None
    website = contact.get("website") or None

    rows: list[NormalizedPractitionerRow] = []

    # Primary doctor
    primary_doc_field = extras.get("doctor's name") or ""
    primary_email = extras.get("doctor's email") or None
    primary_name, primary_creds = _extract_credentials(primary_doc_field)
    if not primary_name:
        # Fall back to practice name when no Doctor's Name is registered
        primary_name = practice_name or ""

    if primary_name:
        row = NormalizedPractitionerRow(
            tier="org_member",
            name=primary_name,
            specialties=list(LOCKED_SPECIALTIES),
            source_org="OEPF",
            source_url=f"{base_url.rstrip('/')}/#doctor-1",
            fellowship_level=_is_fellowship(primary_creds, primary_doc_field),
            practice_name=practice_name if practice_name != primary_name else None,
            credentials=primary_creds,
            phone=phone,
            email=primary_email,
            website=website,
            address1=address1,
            city=city,
            state=state,
            postal=postal,
            country=country or "US",
        )
        rows.append(row)

    # Secondary doctor (if present)
    second_doc_field = extras.get("2nd doctor/therapist's name") or ""
    second_email = extras.get("2nd doctor/therapist's email") or None
    if second_doc_field:
        second_name, second_creds = _extract_credentials(second_doc_field)
        if second_name:
            rows.append(
                NormalizedPractitionerRow(
                    tier="org_member",
                    name=second_name,
                    specialties=list(LOCKED_SPECIALTIES),
                    source_org="OEPF",
                    source_url=f"{base_url.rstrip('/')}/#doctor-2",
                    fellowship_level=_is_fellowship(second_creds, second_doc_field),
                    practice_name=practice_name,
                    credentials=second_creds,
                    phone=phone,
                    email=second_email,
                    website=website,
                    address1=address1,
                    city=city,
                    state=state,
                    postal=postal,
                    country=country or "US",
                )
            )

    return rows
