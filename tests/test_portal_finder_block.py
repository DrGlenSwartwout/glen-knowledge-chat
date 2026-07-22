from dashboard.portal_view import _practitioner_finder_block

def test_finder_block_enabled_prefills_zip_over_city():
    b = _practitioner_finder_block({"zip": "96720", "city": "Hilo", "country": "US"}, True)
    assert b == {"enabled": True, "location": "96720", "country": "US"}

def test_finder_block_disabled_flag_off():
    b = _practitioner_finder_block({"zip": "96720"}, False)
    assert b["enabled"] is False

def test_finder_block_absent_address_empty_location_defaults_us():
    b = _practitioner_finder_block(None, True)
    assert b == {"enabled": True, "location": "", "country": "US"}
