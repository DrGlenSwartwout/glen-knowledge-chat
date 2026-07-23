"""Test condition recommendation source registration."""
from dashboard import recommendation_sources as rs


def test_condition_source_registered():
    """Condition source must be registered with clinical kind."""
    assert rs.known_source("condition")
    s = rs.RECOMMENDATION_SOURCES["condition"]
    assert s["kind"] == "clinical" and "history" in s["label"].lower()
