"""Mapbox forward-geocoding wrapper. Takes a NormalizedPractitionerRow,
returns (lat, lng, quality). Caller writes them back to the row + DB."""
import os
import time
from typing import Optional, Tuple

import requests

from scrapers.practitioner_finder.models import NormalizedPractitionerRow
from scrapers.practitioner_finder.normalize import (
    detect_geocode_quality,
    geocode_input_string,
)


class MapboxError(Exception):
    pass


MAPBOX_GEOCODE_URL = (
    "https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"
)
_LAST_CALL_AT = [0.0]
_MIN_INTERVAL = 0.1  # 10 req/sec ceiling — well under Mapbox 600/min limit


def _throttle():
    now = time.monotonic()
    delta = now - _LAST_CALL_AT[0]
    if delta < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - delta)
    _LAST_CALL_AT[0] = time.monotonic()


# Messy US-country strings seen in the source directories that mean "United
# States". Normalised to the Mapbox country filter "us". Everything else that
# is not a 2-letter ISO 3166-1 alpha-2 code is treated as a foreign country
# name we don't have a code for — we omit the filter and let the freeform
# query string (which ends with the country name) drive the match.
_US_COUNTRY_ALIASES = {
    "US", "USA", "U.S.", "U.S.A", "U.S.A.", "UNITED STATES",
    "UNITED STATES OF AMERICA", "UNITED SATES", "PUERTO RICO",
}


def mapbox_country_filter(country: Optional[str]) -> Optional[str]:
    """Map a row's country to a Mapbox `country` filter value (ISO 3166-1
    alpha-2, lowercase) or None to omit the filter entirely.

    - Missing/US-variant  -> "us"  (patient ZIP lookups default country="US")
    - 2-letter ISO code   -> lowercased ("KR" -> "kr")
    - Any other full name -> None  (don't force a country; freeform query drives it)

    Forcing every row to "us" (the old behaviour) silently relocated foreign
    practitioners onto random US coordinates, polluting US ZIP searches."""
    if not country or not country.strip():
        return "us"
    c = country.strip()
    if c.upper() in _US_COUNTRY_ALIASES:
        return "us"
    if len(c) == 2 and c.isalpha():
        return c.lower()
    return None


def geocode_row(
    row: NormalizedPractitionerRow,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Return (lat, lng, quality). lat/lng are None if Mapbox returns no result
    or the row has no geocodable input. quality reflects what input we passed,
    NOT whether geocoding succeeded — so a row with full address that doesn't
    resolve still has quality='full'."""
    quality = detect_geocode_quality(row)
    if quality is None:
        return None, None, None

    token = os.environ.get("MAPBOX_PUBLIC_TOKEN") or os.environ.get("MAPBOX_SECRET_TOKEN")
    if not token:
        raise MapboxError("MAPBOX_PUBLIC_TOKEN (or MAPBOX_SECRET_TOKEN) env var not set")

    query = requests.utils.quote(geocode_input_string(row), safe="")
    params = {"access_token": token, "limit": 1}
    country_filter = mapbox_country_filter(row.country)
    if country_filter:
        params["country"] = country_filter
    _throttle()
    resp = requests.get(
        MAPBOX_GEOCODE_URL.format(query=query),
        params=params,
        timeout=10,
    )
    if resp.status_code != 200:
        raise MapboxError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    features = resp.json().get("features", [])
    if not features:
        return None, None, quality

    lng, lat = features[0]["center"]
    return float(lat), float(lng), quality
