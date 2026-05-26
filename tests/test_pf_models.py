import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.practitioner_finder.models import NormalizedPractitionerRow


def test_row_minimum_required_fields():
    row = NormalizedPractitionerRow(
        tier="eyehealing",
        name="Dr. Jane Doe",
        specialties=["eye_care"],
    )
    assert row.tier == "eyehealing"
    assert row.name == "Dr. Jane Doe"
    assert row.specialties == ["eye_care"]
    assert row.country == "US"
    assert row.fellowship_level is False


def test_row_to_dict_strips_none():
    row = NormalizedPractitionerRow(
        tier="eyehealing",
        name="Dr. Jane Doe",
        specialties=["eye_care"],
        city="Honolulu",
        state="HI",
    )
    d = row.to_dict()
    assert d["city"] == "Honolulu"
    assert d["state"] == "HI"
    assert "phone" not in d  # None values stripped


def test_row_to_dict_keeps_empty_lists_and_false():
    row = NormalizedPractitionerRow(
        tier="eyehealing",
        name="Dr. X",
        specialties=[],
        fellowship_level=False,
    )
    d = row.to_dict()
    assert d["specialties"] == []
    assert d["fellowship_level"] is False
