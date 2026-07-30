from pathlib import Path


PORTAL_HTML = (
    Path(__file__).resolve().parents[1] / "static" / "client-portal.html"
).read_text()


def test_health_profile_uses_select_when_field_has_options():
    assert 'return `<select class="health-edit-input">' in PORTAL_HTML
    assert "Select an option" in PORTAL_HTML


def test_health_profile_displays_selected_option_label():
    assert "const displayValue = selected" in PORTAL_HTML
    assert "selected.label" in PORTAL_HTML
