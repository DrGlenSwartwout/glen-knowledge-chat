"""Static contract test for the course Body-Map concern-marking widget
(static/course-bodymap.js). No browser -- just asserts the JS source exposes
the expected API surface and touches the right endpoint, without any
PHI/photo/token handling.
"""
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "static" / "course-bodymap.js"


def _source():
    return SRC_PATH.read_text(encoding="utf-8")


def test_file_exists():
    assert SRC_PATH.exists()


def test_defines_course_body_map_with_mount_and_serialize():
    src = _source()
    assert "window.CourseBodyMap" in src
    assert "mount" in src
    assert "_serialize" in src
    # exposed as an object with these two keys
    assert "mount: mount" in src
    assert "_serialize: _serialize" in src


def test_references_public_data_endpoint():
    src = _source()
    assert "/body-map/data" in src


def test_uses_600_viewbox_and_scale_group():
    src = _source()
    assert 'viewBox="0 0 600 600"' in src
    assert "scale(600)" in src


def test_posts_to_submit_url_with_location_search():
    src = _source()
    assert "submitUrl" in src
    assert "location.search" in src


def test_does_not_touch_photo_or_token_handling():
    src = _source()
    assert "bodymap-photo" not in src
    assert "token=" not in src
    assert ".blob" not in src


def test_contains_all_ten_system_slugs():
    src = _source()
    for slug in [
        "organs", "skeleton", "muscle", "nervous", "endocrine",
        "respiratory", "digestive", "cardiovascular", "urogenital", "lymph",
    ]:
        assert ('"' + slug + '"') in src, "missing system slug: " + slug
