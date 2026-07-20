import re

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


def test_clean_services_caps_item_len_at_60():
    """Finding 2: MAX_SERVICE_LEN=60 truncation had no test."""
    out = pp.clean_services(["x" * 100])
    assert out == ["x" * 60]
    assert len(out[0]) == 60


def test_clean_services_60_char_item_preserved_whole():
    """Boundary: exactly 60 chars must survive untruncated."""
    item = "x" * 60
    out = pp.clean_services([item])
    assert out == [item]
    assert len(out[0]) == 60


# --- Finding 1: bare comparison operators in prose must not be eaten as tags ---

def test_sanitize_bio_survives_comparison_operators_intact():
    text = "Reduced A1C from 9.2 to <6.0 and BP <120 >80 today"
    assert pp.sanitize_bio(text) == text


def test_sanitize_bio_survives_spaced_comparison_intact():
    text = "kept IOP < 15 mmHg"
    assert pp.sanitize_bio(text) == text


def test_sanitize_bio_still_strips_bold_tags():
    assert pp.sanitize_bio("<b>Dr.</b> Glen") == "Dr. Glen"


def test_sanitize_bio_still_strips_script_tag_markup():
    out = pp.sanitize_bio("<script>alert(1)</script>bio")
    assert "<script>" not in out
    assert "</script>" not in out


def test_sanitize_bio_still_strips_img_tag_with_attrs():
    assert pp.sanitize_bio("<img src=x onerror=y>hi") == "hi"


def test_sanitize_bio_nested_angle_brackets_no_interpretable_tag_survives():
    out = pp.sanitize_bio("<<b>>text")
    # the inner "<b>" is a real tag and is stripped; the leftover "<" and ">"
    # are not adjacent to a letter, so they can't be interpreted as a tag.
    assert "<b>" not in out
    assert not re.search(r"<\s*/?\s*[a-zA-Z]", out)


def test_format_location_variants():
    assert pp.format_location("Hilo", "HI") == "Hilo, HI"
    assert pp.format_location("Hilo", "") == "Hilo"
    assert pp.format_location("", "HI") == ""
    assert pp.format_location(None, None) == ""


def test_profile_public_fields_frozen():
    assert pp.PROFILE_PUBLIC_FIELDS == frozenset(
        {"bio", "photo_url", "logo_url", "services", "location", "accepting_clients"})
