import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parent.parent))

import app  # noqa: E402


@patch("scrapers.practitioner_finder.db.run_search")
@patch("scrapers.practitioner_finder.geocode.geocode_row")
def test_search_returns_results(mock_geocode, mock_run_search):
    mock_geocode.return_value = (21.3099, -157.8581, "zip")
    mock_run_search.return_value = [
        {"id": "abc", "name": "Dr. X", "lat": 21.31, "lng": -157.86,
         "specialties": ["eye_care"], "distance_miles": 1.2},
    ]
    client = app.app.test_client()
    resp = client.get(
        "/api/practitioner-finder/search?zip=96813&radius_miles=25",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 1
    assert data["practitioners"][0]["name"] == "Dr. X"


def test_search_missing_zip_returns_400():
    client = app.app.test_client()
    resp = client.get("/api/practitioner-finder/search")
    assert resp.status_code == 400
    assert "zip" in resp.get_json()["error"].lower()


@patch("scrapers.practitioner_finder.db.run_search")
@patch("scrapers.practitioner_finder.geocode.geocode_row")
def test_search_with_specialties(mock_geocode, mock_run_search):
    mock_geocode.return_value = (21.3, -157.8, "zip")
    mock_run_search.return_value = []
    client = app.app.test_client()
    resp = client.get(
        "/api/practitioner-finder/search?zip=96813&specialties[]=eye_care&specialties[]=syntonic"
    )
    assert resp.status_code == 200
    call_kwargs = mock_run_search.call_args.kwargs
    assert call_kwargs["specialties"] == ["eye_care", "syntonic"]
