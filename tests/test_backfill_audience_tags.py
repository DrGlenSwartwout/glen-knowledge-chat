import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_audience_tags import classify_audience


def test_practitioner_keyword_routes_to_practitioner():
    text = "For clinicians using ASH certification protocols, dose..."
    assert classify_audience(text) == "practitioner"


def test_client_keyword_routes_to_client():
    text = "I have macular degeneration and I feel my vision worsening at home."
    assert classify_audience(text) == "client"


def test_depth_keywords_route_to_practitioner():
    """Two depth keywords should classify as practitioner content."""
    text = "Mechanism: tight-junction proteins regulate paracellular flux."
    assert classify_audience(text) == "practitioner"


def test_default_is_both():
    text = "Drink lots of water for general hydration."
    assert classify_audience(text) == "both"


def test_empty_text_returns_both():
    assert classify_audience("") == "both"
    assert classify_audience(None) == "both"
