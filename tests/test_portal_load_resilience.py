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
    for route in ("biofield", "recs", "intake", "offers", "photo", "cart", "shop"):
        assert f"{route}:" in HTML
    assert 'window.addEventListener("hashchange", applyPortalHash)' in HTML
    assert 'id="photo-section"' in HTML
    assert 'id="offers-card"' in HTML
    assert 'id="biofield-section"' in HTML
    assert 'data-panel="intake"' in HTML
    assert 'id="portal-intake-card"' in HTML


def test_native_intake_is_resumable_and_reports_save_state():
    assert '"My Intake", "Build or continue your clinical health profile"' in HTML
    assert "initPortalIntakeCard();" in HTML
    assert "Save and finish later" in HTML
    assert "Section ${sectionIndex + 1} of ${sectionEls.length}" in HTML
    assert 'saveState.textContent = "Saving…"' in HTML
    assert 'saveState.textContent = "Saved"' in HTML


def test_life_stress_essence_links_are_readable_in_dark_mode():
    assert ':root[data-theme="dark"] .life-stress-card a' in HTML
    assert ':root[data-theme="dark"] .life-stress-card a:visited' in HTML
    assert "color:#8ecbff" in HTML
