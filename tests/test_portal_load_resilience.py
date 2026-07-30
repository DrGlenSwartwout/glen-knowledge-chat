from pathlib import Path


HTML = (Path(__file__).resolve().parents[1] / "static" /
        "client-portal.html").read_text()


def test_portal_fetches_are_bounded_and_settled_independently():
    assert "const PORTAL_FETCH_TIMEOUT_MS = 10000;" in HTML
    assert "new AbortController()" in HTML
    assert "Promise.allSettled([" in HTML


def test_portal_distinguishes_invalid_link_from_transient_failure():
    assert 'required.status === 404) notFound()' in HTML
    assert "showPortalLoadFailure(required)" in HTML
    assert "Retry missing information" in HTML


def test_shared_portal_hashes_route_to_real_panels_and_cards():
    for route in ("biofield", "recs", "offers", "photo", "cart", "shop"):
        assert f"{route}:" in HTML
    assert 'window.addEventListener("hashchange", applyPortalHash)' in HTML
    assert 'id="photo-section"' in HTML
    assert 'id="offers-card"' in HTML
    assert 'id="biofield-section"' in HTML
