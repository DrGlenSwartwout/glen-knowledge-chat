"""Unit tests for the pure dedupe_tags_ci helper (no DB, no Flask, no env)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.people import dedupe_tags_ci


def test_no_dupes_passthrough():
    assert dedupe_tags_ci(["type:client", "pb-intake"]) == ["type:client", "pb-intake"]


def test_collapses_case_twin_prefers_mixed_case():
    # GHL echoes a lowercase twin of an app-authored mixed-case tag.
    assert dedupe_tags_ci(["terrain:Aging/Rejuvenation", "terrain:aging/rejuvenation"]) == \
        ["terrain:Aging/Rejuvenation"]


def test_prefers_mixed_case_regardless_of_order():
    # Even when the lowercase form appears first, the mixed-case wins.
    assert dedupe_tags_ci(["regulation:positive", "regulation:Positive"]) == ["regulation:Positive"]


def test_lowercase_only_kept_as_is():
    assert dedupe_tags_ci(["pb-intake", "found-via:internet"]) == ["pb-intake", "found-via:internet"]


def test_distinct_tags_preserved_in_first_seen_order():
    assert dedupe_tags_ci(["p:1", "q:2", "p:1"]) == ["p:1", "q:2"]


def test_drops_blank_and_non_str():
    assert dedupe_tags_ci(["  ", "", None, 5, "x:y"]) == ["x:y"]


def test_strips_whitespace():
    assert dedupe_tags_ci([" a:B ", "a:b"]) == ["a:B"]


def test_realistic_mixed_bag():
    tags = ["terrain:Degenerative", "terrain:degenerative", "regulation:Positive",
            "regulation:positive", "tissue-layer:Containment", "tissue-layer:containment",
            "pb-intake", "pb:dry-amd"]
    assert dedupe_tags_ci(tags) == [
        "terrain:Degenerative", "regulation:Positive", "tissue-layer:Containment",
        "pb-intake", "pb:dry-amd",
    ]
