import os
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("requires PINECONE_API_KEY in env (use doppler run)", allow_module_level=True)

import app  # noqa: E402


@patch("scrapers.practitioner_finder.db.run_search")
@patch("scrapers.practitioner_finder.geocode.geocode_place")
def test_search_returns_results(mock_geocode, mock_run_search):
    mock_geocode.return_value = (21.3099, -157.8581)
    mock_run_search.return_value = [
        {"id": "abc", "name": "Dr. X", "lat": 21.31, "lng": -157.86,
         "specialties": ["eye_care"], "distance_miles": 1.2},
    ]
    client = app.app.test_client()
    resp = client.get(
        "/api/practitioner-finder/search?location=96813&radius_miles=25",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 1
    assert data["practitioners"][0]["name"] == "Dr. X"
    # Default country US flows through as a single-country filter.
    assert mock_run_search.call_args.kwargs["countries"] == ["US"]


def test_search_missing_location_returns_400():
    client = app.app.test_client()
    resp = client.get("/api/practitioner-finder/search")
    assert resp.status_code == 400
    assert "location" in resp.get_json()["error"].lower()


@patch("scrapers.practitioner_finder.db.run_search")
@patch("scrapers.practitioner_finder.geocode.geocode_place")
def test_search_zip_backcompat(mock_geocode, mock_run_search):
    # Legacy callers still pass ?zip= — it's used as the location.
    mock_geocode.return_value = (21.3, -157.8)
    mock_run_search.return_value = []
    client = app.app.test_client()
    resp = client.get("/api/practitioner-finder/search?zip=96813")
    assert resp.status_code == 200
    assert mock_geocode.call_args.args[0] == "96813"


@patch("scrapers.practitioner_finder.db.run_search")
@patch("scrapers.practitioner_finder.geocode.geocode_place")
def test_search_country_flows_to_run_search_and_geocode_bias(mock_geocode, mock_run_search):
    mock_geocode.return_value = (52.52, 13.40)
    mock_run_search.return_value = []
    client = app.app.test_client()
    resp = client.get("/api/practitioner-finder/search?country=DE&location=Berlin")
    assert resp.status_code == 200
    assert mock_geocode.call_args.args == ("Berlin", "DE")
    assert mock_run_search.call_args.kwargs["countries"] == ["DE"]


@patch("scrapers.practitioner_finder.db.run_search")
@patch("scrapers.practitioner_finder.geocode.geocode_place")
def test_search_international_any_has_no_country_filter(mock_geocode, mock_run_search):
    mock_geocode.return_value = (48.85, 2.35)
    mock_run_search.return_value = []
    client = app.app.test_client()
    resp = client.get("/api/practitioner-finder/search?country=ANY&location=Paris")
    assert resp.status_code == 200
    # No country bias on geocode, no country filter on search.
    assert mock_geocode.call_args.args == ("Paris", None)
    assert mock_run_search.call_args.kwargs["countries"] is None


@patch("scrapers.practitioner_finder.db.run_search")
@patch("scrapers.practitioner_finder.geocode.geocode_place")
def test_search_country_wide_radius_maps_to_large_radius(mock_geocode, mock_run_search):
    mock_geocode.return_value = (51.50, -0.12)
    mock_run_search.return_value = []
    client = app.app.test_client()
    resp = client.get(
        "/api/practitioner-finder/search?country=GB&location=London&radius_miles=country-wide"
    )
    assert resp.status_code == 200
    assert mock_run_search.call_args.kwargs["radius_miles"] == app.PF_COUNTRYWIDE_RADIUS_MILES


@patch("scrapers.practitioner_finder.db.run_search")
@patch("scrapers.practitioner_finder.geocode.geocode_place")
def test_search_unlocatable_place_returns_404(mock_geocode, mock_run_search):
    mock_geocode.return_value = (None, None)
    client = app.app.test_client()
    resp = client.get("/api/practitioner-finder/search?location=Nowheresville")
    assert resp.status_code == 404
    mock_run_search.assert_not_called()


@patch("scrapers.practitioner_finder.db.run_search")
@patch("scrapers.practitioner_finder.geocode.geocode_place")
def test_search_with_specialties(mock_geocode, mock_run_search):
    mock_geocode.return_value = (21.3, -157.8)
    mock_run_search.return_value = []
    client = app.app.test_client()
    resp = client.get(
        "/api/practitioner-finder/search?location=96813&specialties[]=eye_care&specialties[]=syntonic"
    )
    assert resp.status_code == 200
    call_kwargs = mock_run_search.call_args.kwargs
    assert call_kwargs["specialties"] == ["eye_care", "syntonic"]
