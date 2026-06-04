import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.practitioner_finder.models import NormalizedPractitionerRow
from scrapers.practitioner_finder.geocode import geocode_row, geocode_place, MapboxError


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
def test_geocode_place_bare_city_with_country_bias(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"features": [{"center": [13.40, 52.52]}]},
    )
    lat, lng = geocode_place("Berlin", "DE")
    assert (lat, lng) == (52.52, 13.40)
    # Country bias passed to Mapbox, lower-cased.
    assert mock_get.call_args.kwargs["params"]["country"] == "de"


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
def test_geocode_place_no_country_omits_bias(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"features": [{"center": [2.35, 48.85]}]},
    )
    geocode_place("Paris", None)
    # International search: no country forced (NOT defaulted to "us").
    assert "country" not in mock_get.call_args.kwargs["params"]


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
def test_geocode_place_no_feature_returns_none(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"features": []})
    assert geocode_place("Nowheresville", "US") == (None, None)


def test_geocode_place_empty_input_returns_none():
    assert geocode_place("", "US") == (None, None)
    assert geocode_place("   ", None) == (None, None)


def _row(**kw):
    return NormalizedPractitionerRow(tier="eyehealing", name="X", specialties=[], **kw)


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
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
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
def test_geocode_us_row_constrains_to_us(mock_get):
    """Default country='US' row biases Mapbox to the US (country=us)."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"features": [{"center": [-157.8581, 21.3099]}]},
    )
    row = _row(address1="123 Main St", city="Honolulu", state="HI", postal="96813")
    geocode_row(row)
    assert mock_get.call_args.kwargs["params"]["country"] == "us"


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
def test_geocode_foreign_iso2_passes_real_country(mock_get):
    """A KR practitioner must geocode to Korea, not be force-matched to the US."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"features": [{"center": [129.16, 35.16]}]},
    )
    row = _row(address1="Centum 5-ro 55", city="Busan", state="Busan",
               postal="48059", country="KR")
    geocode_row(row)
    assert mock_get.call_args.kwargs["params"]["country"] == "kr"


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
def test_geocode_fullname_country_omits_filter(mock_get):
    """A non-ISO2 country name (e.g. 'Zimbabwe') omits the country filter so
    Mapbox resolves it from the freeform query rather than force-matching the US."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"features": [{"center": [31.05, -17.83]}]},
    )
    row = _row(address1="1 Harare St", city="Harare", state="Mashonaland",
               postal="0000", country="Zimbabwe")
    geocode_row(row)
    assert "country" not in mock_get.call_args.kwargs["params"]


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
def test_geocode_us_name_variant_constrains_to_us(mock_get):
    """Messy US strings ('U.S.A', 'United Sates') still bias to the US."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"features": [{"center": [-90.3, 38.6]}]},
    )
    row = _row(address1="1 Main St", city="St. Louis", state="Missouri",
               postal="63131", country="U.S.A")
    geocode_row(row)
    assert mock_get.call_args.kwargs["params"]["country"] == "us"


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
def test_geocode_no_features_returns_none(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"features": []})
    row = _row(address1="999 Nowhere Ln", city="Nowheresville", state="XX")
    lat, lng, quality = geocode_row(row)
    assert lat is None
    assert lng is None
    assert quality == "full"  # quality reflects input, not success


@patch("scrapers.practitioner_finder.geocode.requests.get")
@patch.dict("os.environ", {"MAPBOX_PUBLIC_TOKEN": "pk.fake"})
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
