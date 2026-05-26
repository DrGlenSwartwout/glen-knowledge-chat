import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.practitioner_finder.models import NormalizedPractitionerRow
from scrapers.practitioner_finder.geocode import geocode_row, MapboxError


def _row(**kw):
    return NormalizedPractitionerRow(tier="eyehealing", name="X", specialties=[], **kw)


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_SECRET_TOKEN": "sk.fake"})
def test_geocode_full_address_success(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"features": [{"center": [-157.8581, 21.3099]}]},
    )
    row = _row(address1="123 Main St", city="Honolulu", state="HI", postal="96813")
    lat, lng, quality = geocode_row(row)
    assert lat == 21.3099
    assert lng == -157.8581
    assert quality == "full"


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_SECRET_TOKEN": "sk.fake"})
def test_geocode_no_features_returns_none(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"features": []})
    row = _row(address1="999 Nowhere Ln", city="Nowheresville", state="XX")
    lat, lng, quality = geocode_row(row)
    assert lat is None
    assert lng is None
    assert quality == "full"  # quality reflects input, not success


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_SECRET_TOKEN": "sk.fake"})
def test_geocode_http_error_raises(mock_get):
    mock_get.return_value = MagicMock(status_code=429, text="rate limited")
    row = _row(address1="123 Main St", city="Honolulu", state="HI")
    try:
        geocode_row(row)
        assert False, "expected MapboxError"
    except MapboxError as e:
        assert "429" in str(e)


def test_geocode_no_location_returns_none_quality():
    """Row with nothing geocodable — skip without API call."""
    row = _row()
    lat, lng, quality = geocode_row(row)
    assert lat is None
    assert lng is None
    assert quality is None
