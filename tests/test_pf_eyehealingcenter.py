import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.eyehealingcenter import (
    parse_by_state_html,
    parse_by_city_html,
)


def test_parse_by_state_returns_normalized_rows():
    html = (FIXTURE_DIR / "eyehealingcenter_by_state.html").read_text()
    rows = parse_by_state_html(html)
    assert len(rows) > 0, "expected at least one practitioner parsed from the fixture"
    first = rows[0]
    assert first.tier == "eyehealing"
    assert first.source_org == "eyehealingcenter"
    assert first.specialties == ["eye_care"]
    assert first.name  # non-empty
    assert first.source_url and first.source_url.startswith("https://eyehealingcenter.com")


def test_parse_by_state_at_minimum_has_state():
    html = (FIXTURE_DIR / "eyehealingcenter_by_state.html").read_text()
    rows = parse_by_state_html(html)
    states_present = {r.state for r in rows if r.state}
    assert len(states_present) > 0


def test_parse_by_city_returns_normalized_rows():
    html = (FIXTURE_DIR / "eyehealingcenter_by_city.html").read_text()
    rows = parse_by_city_html(html)
    assert len(rows) > 0
    assert all(r.tier == "eyehealing" for r in rows)
    assert all(r.specialties == ["eye_care"] for r in rows)


def test_parse_by_state_returns_two_letter_state_codes():
    html = (FIXTURE_DIR / "eyehealingcenter_by_state.html").read_text()
    rows = parse_by_state_html(html)
    states_present = {r.state for r in rows if r.state}
    # All state values must be 2-letter codes (or None)
    for s in states_present:
        assert len(s) == 2, f"non-abbreviated state: {s!r}"
        assert s.isupper(), f"state not uppercase: {s!r}"
