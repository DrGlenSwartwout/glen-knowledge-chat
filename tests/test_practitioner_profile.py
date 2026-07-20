import pytest
from dashboard import practitioner_profile as pp


def test_sanitize_bio_strips_html_and_collapses_ws():
    assert pp.sanitize_bio("  <b>Dr.</b>  Glen   heals ") == "Dr. Glen heals"


def test_sanitize_bio_keeps_contact_detail():
    """Deliberate divergence from the client sanitizer — a practitioner may put
    their own phone/email/URL in their own bio."""
    s = pp.sanitize_bio("Reach me at dr@x.com or 555-123-4567, https://drglen.com")
    assert "dr@x.com" in s and "555-123-4567" in s and "https://drglen.com" in s


def test_sanitize_bio_rejects_over_600():
    with pytest.raises(ValueError):
        pp.sanitize_bio("x" * 601)


def test_sanitize_bio_600_exactly_ok():
    assert len(pp.sanitize_bio("x" * 600)) == 600


def test_clean_services_strips_caps_and_drops_empties():
    out = pp.clean_services(["<i>Acupuncture</i>", "  ", "Nutrition", ""])
    assert out == ["Acupuncture", "Nutrition"]


def test_clean_services_caps_count_at_12():
    assert len(pp.clean_services([f"svc{i}" for i in range(20)])) == 12


def test_format_location_variants():
    assert pp.format_location("Hilo", "HI") == "Hilo, HI"
    assert pp.format_location("Hilo", "") == "Hilo"
    assert pp.format_location("", "HI") == ""
    assert pp.format_location(None, None) == ""


def test_profile_public_fields_frozen():
    assert pp.PROFILE_PUBLIC_FIELDS == frozenset(
        {"bio", "photo_url", "logo_url", "services", "location", "accepting_clients"})
