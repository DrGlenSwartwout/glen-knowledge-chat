"""Food for Humans (findfoodforhumans.com) adapter.

A robots-compliant crawl of the open regenerative-farm directory:
  robots.txt allows /, disallows only /admin/ and /api/.
  We use the permitted paths: /sitemap.xml -> /listing/<slug>/ pages.
  (The /api/explore/ JSON is robots-disallowed AND its pagination is broken
  from outside — it returns the same first 40 rows for every page — so it is
  deliberately NOT used.)

Each listing page carries everything we need in the rendered HTML:
  - a schema.org LocalBusiness <script type="application/ld+json"> block with
    name, description, email, telephone, full PostalAddress, GeoCoordinates,
    and makesOffer[] (products). lat/lng are already present -> no geocoding.
  - the regenerative *practices* and *ordering options* are rendered as styled
    badge <span>s grouped under their section headings.

Public API:
  fetch_listing_urls()          -> list[str]          (all 1822 listing URLs)
  parse_listing(html, url)      -> NormalizedFarmRow   (pure; unit-tested)
  scrape(limit=None, sleep=0.5) -> list[NormalizedFarmRow]
"""
from __future__ import annotations

import html as _html
import json
import re
import time
from typing import Optional

import requests

from scrapers.farm_finder.models import NormalizedFarmRow

BASE = "https://findfoodforhumans.com"
SITEMAP_URL = f"{BASE}/sitemap.xml"
SOURCE_ORG = "Food for Humans"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# Hosts that are outbound links on a listing page but are NOT the farm's own
# website (social profiles, infra, the directory itself).
_NON_WEBSITE_HOSTS = (
    "findfoodforhumans.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "tiktok.com",
    "googletagmanager.com",
    "google.com",
    "api.mapbox.com",
    "supabase.co",
    "schema.org",
)

# Full state/province name -> 2-letter code. Source stores addressRegion as a
# full name ("Tennessee"); we normalize so the finder can filter by state code
# like the practitioner finder does. Unknown names pass through unchanged.
_REGION_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
    # Canadian provinces / territories
    "alberta": "AB", "british columbia": "BC", "manitoba": "MB",
    "new brunswick": "NB", "newfoundland and labrador": "NL",
    "nova scotia": "NS", "ontario": "ON", "prince edward island": "PE",
    "quebec": "QC", "saskatchewan": "SK", "northwest territories": "NT",
    "nunavut": "NU", "yukon": "YT",
}

_LDJSON_RE = re.compile(
    r'<script type="application/ld\+json">(.*?)</script>', re.S
)
_BADGE_RE = re.compile(
    r'<span class="text-\[13px\] font-medium leading-tight '
    r'text-\[color:var\(--foreground\)\]">([^<]+)</span>'
)
_OUTBOUND_RE = re.compile(
    r'<a[^>]+href="(https?://[^"]+)"[^>]*target="_blank"', re.I
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_listing_urls(sess: Optional[requests.Session] = None) -> list[str]:
    """Return every /listing/<slug>/ URL from the sitemap."""
    sess = sess or _session()
    resp = sess.get(SITEMAP_URL, timeout=30)
    resp.raise_for_status()
    return re.findall(r"<loc>(https?://[^<]+/listing/[^<]+)</loc>", resp.text)


def _region_code(region: Optional[str]) -> Optional[str]:
    if not region:
        return None
    r = region.strip()
    return _REGION_TO_CODE.get(r.lower(), r)


def _pick_localbusiness(html_text: str) -> Optional[dict]:
    for block in _LDJSON_RE.findall(html_text):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "LocalBusiness":
            return data
    return None


def _section_badges(html_text: str) -> dict[str, list[str]]:
    """Group badge spans by the section heading they fall under.

    Headings observed: 'Products', 'Farming practices', 'Ordering options',
    'Listing details'. Each badge is assigned to the nearest heading that
    precedes it. Returns {heading_lower: [badge, ...]}."""
    heading_positions = []
    for m in re.finditer(r"<h[23][^>]*>([^<]{2,40})</h[23]>", html_text):
        heading_positions.append((m.start(), m.group(1).strip().lower()))
    heading_positions.sort()

    out: dict[str, list[str]] = {}
    for m in _BADGE_RE.finditer(html_text):
        pos = m.start()
        section = None
        for hpos, htext in heading_positions:
            if hpos < pos:
                section = htext
            else:
                break
        if section is None:
            continue
        out.setdefault(section, []).append(_html.unescape(m.group(1).strip()))
    return out


def _pick_website(html_text: str) -> Optional[str]:
    for url in _OUTBOUND_RE.findall(html_text):
        host = re.sub(r"https?://([^/]+).*", r"\1", url).lower()
        host = host[4:] if host.startswith("www.") else host
        if not any(host == h or host.endswith("." + h) for h in _NON_WEBSITE_HOSTS):
            return url
    return None


def parse_listing(html_text: str, url: str) -> Optional[NormalizedFarmRow]:
    """Parse one listing page's HTML into a NormalizedFarmRow. Pure function.

    Returns None if the page has no LocalBusiness block (e.g. a non-farm page
    that slipped into the sitemap)."""
    lb = _pick_localbusiness(html_text)
    if not lb:
        return None

    addr = lb.get("address") or {}
    geo = lb.get("geo") or {}

    products = [
        offer.get("itemOffered", {}).get("name")
        for offer in lb.get("makesOffer", [])
        if offer.get("itemOffered", {}).get("name")
    ]

    badges = _section_badges(html_text)
    practices = badges.get("farming practices", [])
    order_options = badges.get("ordering options", [])

    images = lb.get("image") or []
    image_url = images[0] if isinstance(images, list) and images else None
    # Source occasionally emits a doubled-up URL (base + absolute supabase url).
    if image_url and image_url.startswith(BASE + "/https://"):
        image_url = image_url[len(BASE) + 1:]

    lat = geo.get("latitude")
    lng = geo.get("longitude")

    # streetAddress arrives as the full one-line address ("1400 Buttermilk Road,
    # Lenoir City, Tennessee 37771, United States"); keep only the street line so
    # it doesn't duplicate the city/state/postal columns on the card.
    street = addr.get("streetAddress")
    if street:
        street = street.split(",")[0].strip() or None

    return NormalizedFarmRow(
        name=lb.get("name", "").strip(),
        source_org=SOURCE_ORG,
        source_url=url,
        practices=practices,
        products=products,
        order_options=order_options,
        description=(lb.get("description") or "").strip() or None,
        phone=lb.get("telephone"),
        email=lb.get("email"),
        website=_pick_website(html_text),
        image_url=image_url,
        address1=street,
        city=addr.get("addressLocality"),
        state=_region_code(addr.get("addressRegion")),
        postal=addr.get("postalCode"),
        country=addr.get("addressCountry") or "US",
        lat=float(lat) if lat is not None else None,
        lng=float(lng) if lng is not None else None,
        geocode_quality="source" if lat is not None else None,
    )


def scrape(
    limit: Optional[int] = None,
    sleep: float = 0.5,
    sess: Optional[requests.Session] = None,
) -> list[NormalizedFarmRow]:
    """Crawl listing pages and return parsed farm rows.

    limit caps the number of listings fetched (for the pilot). sleep throttles
    between requests to be a polite crawler."""
    sess = sess or _session()
    urls = fetch_listing_urls(sess)
    if limit is not None:
        urls = urls[:limit]

    rows: list[NormalizedFarmRow] = []
    for url in urls:
        try:
            resp = sess.get(url, timeout=30)
            resp.raise_for_status()
            row = parse_listing(resp.text, url)
            if row:
                rows.append(row)
        except requests.RequestException:
            # Isolate per-listing failures; the orchestrator logs the count.
            pass
        if sleep:
            time.sleep(sleep)
    return rows
