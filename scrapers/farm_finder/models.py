"""Normalized farm row — adapter output, db.py input.

Mirrors scrapers/practitioner_finder/models.py but for the Farm Finder: a
separate directory/map of regenerative-agriculture farms & ranches (Glen's
decision: keep farms and health practitioners in separate surfaces).

Key differences from a practitioner:
  - `practices`  (regenerative markers) replaces `specialties`
  - `products`   (what the farm sells) is farm-specific
  - `order_options` (Farm Pickup / Shipping / CSA / Bulk ...) replaces the
    telehealth/accepting-new-patients booleans
  - lat/lng usually arrive ALREADY geocoded from the source, so the Mapbox
    geocode sweep is a fallback, not the primary path.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class NormalizedFarmRow:
    # Required
    name: str

    # Source / provenance
    source_org: Optional[str] = None       # e.g. "Food for Humans"
    source_url: Optional[str] = None       # the listing page (upsert key)

    # Classification
    practices: list[str] = field(default_factory=list)      # regenerative markers
    products: list[str] = field(default_factory=list)       # what they sell
    order_options: list[str] = field(default_factory=list)  # how to buy

    # Identity / description
    description: Optional[str] = None

    # Contact
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    image_url: Optional[str] = None

    # Location
    address1: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None            # 2-letter where recognized, else raw
    postal: Optional[str] = None
    country: str = "US"

    # Geocoded (often provided by the source; geocode.py is fallback only)
    lat: Optional[float] = None
    lng: Optional[float] = None
    geocode_quality: Optional[str] = None  # 'source' | 'full' | 'city' | 'zip'

    def to_dict(self) -> dict:
        """Return dict with None values stripped — ready for db upsert.
        Empty lists ARE kept (they map to Postgres empty arrays)."""
        raw = asdict(self)
        return {k: v for k, v in raw.items() if v is not None}
