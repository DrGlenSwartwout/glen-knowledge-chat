import pathlib


def test_client_portal_renders_finder_card_gated_on_flag():
    html = pathlib.Path("static/client-portal.html").read_text()
    assert "v.practitioner_finder && v.practitioner_finder.enabled" in html
    assert "Find a Practitioner Near You" in html
    assert "/practitioner-finder" in html
    assert "<iframe" in html


def test_finder_page_prefills_from_url_and_autosearches():
    html = pathlib.Path("static/practitioner-finder.html").read_text()
    # reads the embed params, and the prefill is wired into the country-load path
    assert "URLSearchParams(window.location.search)" in html
    assert "applyUrlPrefill" in html
    # applyUrlPrefill triggers a search when a location is supplied
    assert "runSearch();" in html
