"""Pure helpers: geocode-quality detection and specialty inference."""
from scrapers.practitioner_finder.models import NormalizedPractitionerRow


DEFAULT_EYEHEALING_SPECIALTIES = ["eye_care"]


def detect_geocode_quality(row: NormalizedPractitionerRow) -> str | None:
    """Return the best geocoding granularity available for this row.
    Priority: full address > city+state > zip > state alone > None."""
    if row.address1 and row.city and row.state:
        return "full"
    if row.city and row.state:
        return "city"
    if row.postal:
        return "zip"
    if row.state:
        return "state_only"
    return None


def geocode_input_string(row: NormalizedPractitionerRow) -> str:
    """Compose the freeform location string passed to Mapbox geocoder.
    Uses whatever granularity is available."""
    parts = []
    if row.address1:
        parts.append(row.address1)
    if row.city:
        parts.append(row.city)
    state_zip = " ".join(p for p in [row.state, row.postal] if p)
    if state_zip:
        parts.append(state_zip)
    parts.append(row.country or "US")
    return ", ".join(parts)


def infer_eyehealing_specialties(description: str) -> list[str]:
    """Phase 1 strategy (per spec): tag every eyehealingcenter row with
    parent 'eye_care' only. Sub-tags (functional/syntonic/rehab/nutritional_eye_care)
    are added by a one-time classification sweep AFTER initial migration completes.
    See docs/superpowers/specs/2026-05-26-practitioner-finder-design.md."""
    return list(DEFAULT_EYEHEALING_SPECIALTIES)
