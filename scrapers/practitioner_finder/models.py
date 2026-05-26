"""Normalized row model — adapter output, db.py input."""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class NormalizedPractitionerRow:
    # Required
    tier: str                       # 'org_member' | 'eyehealing' | 'panel_in_cert' | 'panel_certified'
    name: str
    specialties: list[str]

    # Source / provenance
    source_org: Optional[str] = None
    source_url: Optional[str] = None
    fellowship_level: bool = False

    # Identity
    practice_name: Optional[str] = None
    credentials: Optional[str] = None

    # Contact
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None

    # Location
    address1: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal: Optional[str] = None
    country: str = "US"

    # Geocoded (populated by geocode.py, not by adapters)
    lat: Optional[float] = None
    lng: Optional[float] = None
    geocode_quality: Optional[str] = None  # 'full' | 'city' | 'zip' | 'state_only'

    # Portal-managed (always None from scrapers)
    photo_url: Optional[str] = None
    bio: Optional[str] = None

    def to_dict(self) -> dict:
        """Return dict with None values stripped — ready for db.upsert."""
        raw = asdict(self)
        return {k: v for k, v in raw.items() if v is not None}
