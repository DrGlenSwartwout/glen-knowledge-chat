"""A Campaign for Real Milk (realmilk.com) adapter.

The Weston A. Price Foundation's raw-milk farm directory, a WordPress Business
Directory (wpbdp) site. robots.txt allows all with Crawl-delay: 10, and exposes
five wpbdp_listing sitemaps -> ~4,500 /farm-directory/<slug>/ pages.

Each listing page carries a schema.org LocalBusiness ld+json (reliable name) and
a set of wpbdp field-display blocks, each `<div class="value">…</div>` under a
`wpbdp-field-<slug>` container: city, state, zip_code, country, email,
description, type_of_location (category). There is NO street address, phone,
website, or coordinates — so realmilk rows carry city/state/zip and are geocoded
by the run_all global sweep (like any address-only practitioner row).

Because of the mandated 10s crawl-delay, a full crawl is ~12 hours: realmilk is
the Farm Finder's SLOW lane (its own monthly job), not part of the weekly run
(see scrapers.farm_finder.sources.WEEKLY_SOURCES).

Public API:
  fetch_listing_urls()          -> list[str]
  parse_listing(html, url)      -> NormalizedFarmRow | None   (pure; unit-tested)
  scrape(limit=None, sleep=10.0) -> list[NormalizedFarmRow]
"""
from __future__ import annotations

import html as _html
import json
import re
import time
from typing import Optional

import requests

from scrapers.farm_finder.foodforhumans import _region_code  # shared name->code
from scrapers.farm_finder.models import NormalizedFarmRow

BASE = "https://www.realmilk.com"
SOURCE_ORG = "A Campaign for Real Milk"
SITEMAP_INDEX = f"{BASE}/sitemap_index.xml"
# Every listing is raw-dairy-related; this is the filterable base practice.
BASE_PRACTICE = "raw_milk"
CRAWL_DELAY = 10.0  # robots.txt Crawl-delay
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# type_of_location category -> ordering label shown on the card.
_LOCATION_ORDER = {
    "home delivery": "Home Delivery",
    "delivery": "Home Delivery",
    "farm": "Farm Pickup",
    "on-farm": "Farm Pickup",
    "store": "Retail Store",
    "retail": "Retail Store",
    "buying club": "Buying Club",
    "co-op": "Co-op",
    "farmers market": "Farmers Market",
    "drop point": "Drop Point",
    "restaurant": "Restaurant",
}

_LDJSON_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
_LISTING_SITEMAP_RE = re.compile(r"<loc>([^<]+wpbdp_listing-sitemap[^<]*)</loc>")
_LISTING_URL_RE = re.compile(r"<loc>([^<]+/farm-directory/[^<]+)</loc>")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_listing_urls(sess: Optional[requests.Session] = None) -> list[str]:
    """Return every /farm-directory/<slug>/ URL across the wpbdp sitemaps."""
    sess = sess or _session()
    idx = sess.get(SITEMAP_INDEX, timeout=30)
    idx.raise_for_status()
    urls: list[str] = []
    for sm_url in _LISTING_SITEMAP_RE.findall(idx.text):
        resp = sess.get(sm_url, timeout=30)
        resp.raise_for_status()
        urls.extend(_LISTING_URL_RE.findall(resp.text))
    # de-dup, preserve order
    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def _pick_localbusiness(html_text: str) -> Optional[dict]:
    for block in _LDJSON_RE.findall(html_text):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "LocalBusiness":
            return data
    return None


def _field_value(html_text: str, slug: str) -> Optional[str]:
    """Pull the text of the `<div class="value">` inside a wpbdp-field-<slug>."""
    m = re.search(
        r'wpbdp-field-' + re.escape(slug) + r'\b.*?<div class="value">(.*?)</div>',
        html_text, re.S,
    )
    if not m:
        return None
    val = re.sub(r"<[^>]+>", " ", m.group(1))
    val = _html.unescape(re.sub(r"\s+", " ", val)).strip()
    return val or None


def _field_links(html_text: str, slug: str) -> list[str]:
    """Pull anchor texts (e.g. categories) from a wpbdp-field-<slug> block."""
    m = re.search(
        r'wpbdp-field-' + re.escape(slug) + r'\b.*?</div>\s*</div>',
        html_text, re.S,
    )
    if not m:
        return []
    return [_html.unescape(x).strip()
            for x in re.findall(r">([^<>]+)</a>", m.group(0)) if x.strip()]


def parse_listing(html_text: str, url: str) -> Optional[NormalizedFarmRow]:
    """Parse one realmilk listing page into a NormalizedFarmRow. Pure function.

    Returns None if the page has no LocalBusiness block (a non-listing page)."""
    lb = _pick_localbusiness(html_text)
    if not lb:
        return None

    name = _html.unescape((lb.get("name") or "").strip())
    if not name:
        return None

    # Prefer the wpbdp field values (the ld+json addressRegion is unreliable).
    city = _field_value(html_text, "city")
    state = _region_code(_field_value(html_text, "state") or "") or None
    postal = _field_value(html_text, "zip_code")
    country = _field_value(html_text, "country") or "US"
    email = _field_value(html_text, "email")
    description = _field_value(html_text, "description")

    categories = _field_links(html_text, "type_of_location")
    order_options: list[str] = []
    for cat in categories:
        label = _LOCATION_ORDER.get(cat.strip().lower(), cat.strip())
        if label and label not in order_options:
            order_options.append(label)

    if country.upper() in ("USA", "UNITED STATES", "US"):
        country = "US"

    return NormalizedFarmRow(
        name=name,
        source_org=SOURCE_ORG,
        source_url=url,
        practices=[BASE_PRACTICE],
        products=[],
        order_options=order_options,
        description=description,
        email=email,
        website=None,          # wpbdp listing has no website field
        city=city,
        state=state,
        postal=postal,
        country=country,
        lat=None,              # no coords on page -> global geocode sweep
        lng=None,
        geocode_quality=None,
    )


def scrape(
    limit: Optional[int] = None,
    sleep: float = CRAWL_DELAY,
    sess: Optional[requests.Session] = None,
) -> list[NormalizedFarmRow]:
    """Crawl listing pages and return parsed farm rows.

    Honors the robots Crawl-delay (10s default) — a full crawl is ~12h, so this
    runs on its own slow lane, not the weekly farm run. limit caps listings."""
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
            pass
        if sleep:
            time.sleep(sleep)
    return rows
