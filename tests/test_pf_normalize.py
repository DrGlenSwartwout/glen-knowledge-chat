import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.practitioner_finder.models import NormalizedPractitionerRow
from scrapers.practitioner_finder.normalize import (
    detect_geocode_quality,
    geocode_input_string,
    infer_eyehealing_specialties,
    DEFAULT_EYEHEALING_SPECIALTIES,
)


def _row(**kw):
    return NormalizedPractitionerRow(tier="eyehealing", name="X", specialties=[], **kw)


def test_geocode_quality_full_address():
    row = _row(address1="123 Main St", city="Honolulu", state="HI", postal="96813")
    assert detect_geocode_quality(row) == "full"


def test_geocode_quality_city_only():
    row = _row(city="Honolulu", state="HI")
    assert detect_geocode_quality(row) == "city"


def test_geocode_quality_zip_only():
    row = _row(postal="96813")
    assert detect_geocode_quality(row) == "zip"


def test_geocode_quality_state_only():
    row = _row(state="HI")
    assert detect_geocode_quality(row) == "state_only"


def test_geocode_quality_nothing():
    row = _row()
    assert detect_geocode_quality(row) is None


def test_geocode_input_full():
    row = _row(address1="123 Main St", city="Honolulu", state="HI", postal="96813")
    assert geocode_input_string(row) == "123 Main St, Honolulu, HI 96813, US"


def test_geocode_input_city_only():
    row = _row(city="Honolulu", state="HI")
    assert geocode_input_string(row) == "Honolulu, HI, US"


def test_geocode_input_state_only():
    row = _row(state="HI")
    assert geocode_input_string(row) == "HI, US"


def test_eyehealing_specialties_default():
    """Phase 1: every eyehealingcenter listing tagged eye_care by default.
    Sub-tags layered in by later classification sweep, not at scrape time."""
    assert infer_eyehealing_specialties("any description") == DEFAULT_EYEHEALING_SPECIALTIES
    assert DEFAULT_EYEHEALING_SPECIALTIES == ["eye_care"]
