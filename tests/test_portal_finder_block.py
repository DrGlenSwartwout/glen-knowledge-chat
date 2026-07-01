from dashboard.portal_view import _practitioner_finder_block


def test_prefers_zip_over_city():
    b = _practitioner_finder_block({"zip": "80202", "city": "Denver", "country": "US"}, enabled=True)
    assert b == {"enabled": True, "location": "80202", "country": "US"}


def test_falls_back_to_city_when_no_zip():
    b = _practitioner_finder_block({"zip": "", "city": "Berlin", "country": "DE"}, enabled=True)
    assert b == {"enabled": True, "location": "Berlin", "country": "DE"}


def test_empty_location_and_default_country_when_no_address():
    b = _practitioner_finder_block({}, enabled=True)
    assert b == {"enabled": True, "location": "", "country": "US"}


def test_disabled_flag_passthrough():
    b = _practitioner_finder_block({"zip": "80202", "country": "US"}, enabled=False)
    assert b["enabled"] is False
