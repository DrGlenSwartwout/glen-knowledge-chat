"""Map a NormalizedFarmRow onto a `practitioners`-table upsert dict.

Glen's decision (2026-07-13): the farm finder is NOT a separate surface — farms
are integrated into the existing practitioner finder as a new top-level
category. The finder maps every parent category chip to `specialties[]` (only
'certification' maps to tier[]). So a farm is stored as a practitioners row
whose specialties[] carries the parent tag `regenerative_farms` plus a slug per
regenerative practice. That makes the existing `specialties && %s` search filter
work for farms with ZERO backend query changes:

  - parent chip "Regenerative Farms" -> specialties[]=['regenerative_farms']
    matches every farm row.
  - a sub-chip (e.g. Pasture-Raised) -> specialties[]=['pasture_raised']
    narrows within farms via the same filter.

Farm rows get tier='farm' (a new tier value; see migrations/practitioners-farms.sql)
so the card renderer can show a farm-style card (products, ordering options,
"visit farm website") instead of a clinician card.
"""
import re

from scrapers.farm_finder.models import NormalizedFarmRow

# The parent-category specialty tag every farm row carries. Matches the
# data-parent value of the new top-level chip in practitioner-finder.html.
PARENT_SPECIALTY = "regenerative_farms"

TIER_FARM = "farm"


def practice_slug(practice: str) -> str:
    """'Pasture-Raised' -> 'pasture_raised', 'Non-GMO' -> 'non_gmo'."""
    return re.sub(r"[^a-z0-9]+", "_", practice.strip().lower()).strip("_")


def to_practitioner_row(farm: NormalizedFarmRow) -> dict:
    """Return a dict ready for practitioners upsert (idempotent on source_url).

    specialties = [PARENT_SPECIALTY] + slug(practice) for each practice, so both
    the parent 'Regenerative Farms' chip and any practice sub-chip resolve via
    the existing specialties filter. products / order_options land in the new
    farm-only array columns; name/contact/geo map straight across."""
    specialties = [PARENT_SPECIALTY] + [practice_slug(p) for p in farm.practices]
    # de-dup, preserve order
    seen: set[str] = set()
    specialties = [s for s in specialties if s and not (s in seen or seen.add(s))]

    row = {
        "tier": TIER_FARM,
        "name": farm.name,
        "specialties": specialties,
        "source_org": farm.source_org,
        "source_url": farm.source_url,
        "practice_name": farm.name,
        "bio": farm.description,
        "phone": farm.phone,
        "email": farm.email,
        "website": farm.website,
        "photo_url": farm.image_url,
        "address1": farm.address1,
        "city": farm.city,
        "state": farm.state,
        "postal": farm.postal,
        "country": farm.country,
        "lat": farm.lat,
        "lng": farm.lng,
        # The source ships exact coordinates -> 'full' precision, which is a
        # valid practitioner_geocode_quality enum value (avoids altering the enum).
        "geocode_quality": "full" if farm.lat is not None else None,
        # farm-only array columns (added by migrations/practitioners-farms.sql)
        "products": farm.products or [],
        "order_options": farm.order_options or [],
    }
    return {k: v for k, v in row.items() if v is not None}
