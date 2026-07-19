"""USDA Local Food Directories adapter.

Source: usdalocalfoodportal.com (a USDA AMS / Michigan State cooperative). Five
directories — Farmers Market, CSA, Food Hub, On-Farm Market, Agritourism — each
downloadable in full as JSON from the KEYLESS bulk endpoint used by the site's
own "Download directories data" button:

    GET /api/download_by_directory/?directory=<name>
      headers: X-Requested-With: XMLHttpRequest  (else 301 -> empty)

The bulk export is public-domain and pre-geocoded (location_x / location_y on
~99% of rows) but SANITIZED: it carries no website / phone / email (markets are
locations, not businesses). So USDA rows land with address + coords + production
and sales-channel metadata and NO website link — the finder renders that fine
(website block is conditional). The keyed geo-radius API (…?apikey=…&state=mi)
exists for spatial queries but is NOT needed for full national coverage, so this
adapter deliberately uses the keyless bulk path.

Public API:
  DIRECTORIES                          -> default directory names (agritourism opt-in)
  fetch_directory(name)                -> list[dict]  (raw records)
  parse_row(rec, directory)            -> NormalizedFarmRow | None   (pure; unit-tested)
  scrape(limit=None, sleep=..., directories=None) -> list[NormalizedFarmRow]
"""
from __future__ import annotations

import re
import time
from typing import Optional

import requests

from scrapers.farm_finder.foodforhumans import (  # shared name->code map
    _REGION_TO_CODE,
    _region_code,
)
from scrapers.farm_finder.models import NormalizedFarmRow

_STATE_CODES = set(_REGION_TO_CODE.values())

BASE = "https://www.usdalocalfoodportal.com"
SOURCE_ORG = "USDA Local Food Directories"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# directory name -> the regenerative-practice sub-chip slug it maps to.
# Agritourism is excluded from DIRECTORIES by default (wineries/petting zoos are
# a weaker fit) but parse_row handles it if a caller opts in.
_DIRECTORY_PRACTICE = {
    "farmersmarket": "farmers_market",
    "csa": "csa",
    "foodhub": "food_hub",
    "onfarmmarket": "on_farm_market",
    "agritourism": "agritourism",
}
DIRECTORIES = ["farmersmarket", "csa", "foodhub", "onfarmmarket"]

# Trailing zip. 4 OR 5 digits: New England zips (CT/MA/NH/VT/ME/RI/NJ) begin
# with a 0 that the USDA export drops, so "06484" arrives as "6484".
_ZIP_RE = re.compile(r"(\d{4,5})(?:-\d{4})?\s*$")


def _resolve_state(text: Optional[str]) -> Optional[str]:
    """Resolve a 2-letter state code from a 'state' fragment that may be a full
    name ('Colorado'), a code ('CO'), or a name+code tail ('Tennessee TN')."""
    if not text:
        return None
    t = text.strip().strip(",").strip()
    if not t:
        return None
    # Whole fragment is a known full name?
    code = _region_code(t)
    if code != t and code in _STATE_CODES:
        return code
    toks = t.split()
    # Trailing 2-letter code ('... TN', 'Beaumont Texas' -> handled below).
    if toks and len(toks[-1]) == 2 and toks[-1].upper() in _STATE_CODES:
        return toks[-1].upper()
    # Trailing full state name (1-2 trailing words: 'New York', 'Texas').
    for n in (2, 1):
        if len(toks) >= n:
            cand = " ".join(toks[-n:])
            c = _region_code(cand)
            if c != cand and c in _STATE_CODES:
                return c
    return None


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE}/fe/datasharing/",
    })
    return s


def fetch_directory(
    directory: str, sess: Optional[requests.Session] = None
) -> list[dict]:
    """Download one directory's full record list from the keyless bulk endpoint."""
    sess = sess or _session()
    url = f"{BASE}/api/download_by_directory/?directory={directory}"
    resp = sess.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def _parse_address(raw: Optional[str]) -> tuple[Optional[str], Optional[str],
                                                Optional[str], Optional[str]]:
    """Split a combined 'street, city, State zip' into (street, city, state, zip).

    Tolerant: missing pieces come back None; coords (not this) drive the map, so
    an imperfect split only degrades the state filter, never the pin."""
    if not raw:
        return None, None, None, None
    # Drop a trailing country token ("..., United States") before splitting.
    cleaned = re.sub(r",?\s*(United States|USA|US)\s*$", "", raw.strip(),
                     flags=re.I)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return None, None, None, None

    postal = None
    state = None
    tail = parts[-1]
    m = _ZIP_RE.search(tail)
    if m:
        z = m.group(1)
        postal = z.zfill(5)          # 4-digit New England zip -> pad leading 0
        tail = tail[: m.start()].strip().rstrip(",").strip()

    # After stripping the zip, whatever remains on the last part is the state.
    if tail:
        state = _resolve_state(tail)
        rest = parts[:-1]
    else:
        # Zip stood alone on the last part; state is the prior part.
        rest = parts[:-1]
        if rest:
            state = _resolve_state(rest[-1])
            rest = rest[:-1]

    city = rest[-1] if rest else None
    street = ", ".join(rest[:-1]) if len(rest) > 1 else None
    return street or None, city or None, state or None, postal


def _order_options(rec: dict) -> list[str]:
    """Derive human ordering labels from the sparse saleschannel_* / FNAP flags."""
    opts: list[str] = []
    if rec.get("saleschannel_onlineorder"):
        opts.append("Online Order")
    if rec.get("saleschannel_phoneorder"):
        opts.append("Phone Order")
    if rec.get("saleschannel_csaorder") or rec.get("saleschannel_csa_vendor"):
        opts.append("CSA")
    if rec.get("saleschannel_deliverymethod"):
        opts.append("Delivery")
    fnap = (rec.get("FNAP") or "") + (rec.get("SNAP_option") or "")
    if "SNAP" in fnap or "EBT" in fnap:
        opts.append("SNAP/EBT")
    return opts


def parse_row(rec: dict, directory: str) -> Optional[NormalizedFarmRow]:
    """Parse one USDA record into a NormalizedFarmRow. Pure function.

    Returns None if the record has no usable name."""
    name = (rec.get("listing_name") or "").strip()
    if not name:
        return None

    street, city, state, postal = _parse_address(rec.get("location_address"))

    practices = [_DIRECTORY_PRACTICE.get(directory, "regenerative_farm")]
    spm = (rec.get("specialproductionmethods") or "")
    if "organic" in spm.lower():
        practices.append("usda_organic")   # reuse the finder's existing chip slug

    lat = rec.get("location_y")   # y = latitude
    lng = rec.get("location_x")   # x = longitude
    try:
        lat = float(lat) if lat not in (None, "") else None
        lng = float(lng) if lng not in (None, "") else None
    except (TypeError, ValueError):
        lat = lng = None

    listing_id = rec.get("listing_id")
    desc = (rec.get("listing_desc") or rec.get("location_desc") or "").strip()

    return NormalizedFarmRow(
        name=name,
        source_org=SOURCE_ORG,
        # Stable, unique idempotency key (provenance only — not user-rendered).
        source_url=f"{BASE}/fe/{directory}/{listing_id}",
        practices=practices,
        products=[],
        order_options=_order_options(rec),
        description=desc or None,
        website=None,           # bulk export carries no contact fields
        address1=street,
        city=city,
        state=state,
        postal=postal,
        country="US",
        lat=lat,
        lng=lng,
        geocode_quality="source" if lat is not None else None,
    )


def scrape(
    limit: Optional[int] = None,
    sleep: float = 1.0,
    directories: Optional[list[str]] = None,
    sess: Optional[requests.Session] = None,
) -> list[NormalizedFarmRow]:
    """Download the configured directories and return parsed farm rows.

    limit caps rows PER DIRECTORY (for sampling). sleep throttles between the
    (few) bulk downloads."""
    sess = sess or _session()
    dirs = directories or DIRECTORIES
    rows: list[NormalizedFarmRow] = []
    for directory in dirs:
        try:
            records = fetch_directory(directory, sess=sess)
        except requests.RequestException:
            continue
        if limit is not None:
            records = records[:limit]
        for rec in records:
            row = parse_row(rec, directory)
            if row:
                rows.append(row)
        if sleep:
            time.sleep(sleep)
    return rows
