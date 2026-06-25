from dashboard.biofield_report_html import render_author_html


def _html():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [], "schedule": []}
    return render_author_html(rep, [], "")


def test_phase_toggle_present():
    h = _html()
    assert "Capture stresses" in h and "Balance" in h
    assert "setPhase(" in h            # toggle handler
    assert "function captureStresses" in h


def test_capture_posts_to_route_and_reloads_panel():
    h = _html()
    assert "/author/a7/capture-stresses" in h
    assert "loadStress()" in h         # panel refresh after capture
