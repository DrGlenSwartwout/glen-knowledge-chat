"""Pure helpers: geocode-quality detection and specialty inference."""
from scrapers.practitioner_finder.models import NormalizedPractitionerRow


DEFAULT_EYEHEALING_SPECIALTIES = ["eye_care"]


# Country strings the source directories use to mean "United States". Mapped to
# the ISO 3166-1 alpha-2 code "US". Mirrors geocode._US_COUNTRY_ALIASES but
# deliberately EXCLUDES "PUERTO RICO" — PR is its own ISO code and we keep it
# distinct so it can be selected on its own in the finder.
_US_COUNTRY_ALIASES = {
    "US", "USA", "U.S.", "U.S.A", "U.S.A.", "UNITED STATES",
    "UNITED STATES OF AMERICA", "UNITED SATES", "UNITED STATE",
}

# Full country names / messy values observed in the live practitioners table
# (audit 2026-06-04) mapped to ISO 3166-1 alpha-2. Keys are upper-cased.
# "Georgia" -> GE (the column is a country column); flagged by the migration
# report in case a row meant the US state.
_COUNTRY_NAME_TO_ISO2 = {
    "PUERTO RICO": "PR",
    "SLOVENIA": "SI",
    "GUATEMALA": "GT",
    "VENEZUELA": "VE",
    "LEBANON": "LB",
    "IRAQ": "IQ",
    "CYPRUS": "CY",
    "LUXEMBOURG": "LU",
    "IRAN (ISLAMIC REPUBLIC OF)": "IR",
    "IRAN": "IR",
    "GUYANA": "GY",
    "TANZANIA": "TZ",
    "CROATIA (HRVATSKA)": "HR",
    "CROATIA": "HR",
    "JAMAICA": "JM",
    "ZIMBABWE": "ZW",
    "LATVIA": "LV",
    "BARBADOS": "BB",
    "PARAGUAY": "PY",
    "MAURITIUS": "MU",
    "DOMINICAN REPUBLIC": "DO",
    "JORDAN": "JO",
    "GEORGIA": "GE",
}


def normalize_country(raw: str | None) -> str | None:
    """Normalize a country value to an ISO 3166-1 alpha-2 code where possible.

    - None / empty            -> "US" (the table default; scrapers never emit None)
    - US aliases / typos       -> "US"
    - Full names we recognise  -> their ISO-2 code (e.g. "Slovenia" -> "SI")
    - An existing 2-letter code -> passed through, upper-cased
    - Anything else            -> returned upper-cased, unchanged (never dropped;
                                   the migration report surfaces it for review)

    Pure function: no I/O. The single source of truth for the stored
    practitioners.country column — used at the db upsert boundary and by
    migrate_normalize_country.py."""
    if raw is None or not raw.strip():
        return "US"
    c = raw.strip().upper()
    if c in _US_COUNTRY_ALIASES:
        return "US"
    if c in _COUNTRY_NAME_TO_ISO2:
        return _COUNTRY_NAME_TO_ISO2[c]
    if len(c) == 2 and c.isalpha():
        return c
    return c


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
