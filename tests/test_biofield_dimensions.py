"""Increment 4b: the extensible five-fold-dimensions framework + the
depth-of-penetration reach match-check."""
import sqlite3
from dashboard.biofield_dimensions import (
    init_dimension_tables, seed_dimensions, list_dimensions, dimension_values,
    tag, get_tag, depth_match, depth_label, DEPTH_KEY)


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    init_dimension_tables(cx)
    seed_dimensions(cx)
    return cx


def test_seed_has_the_dimensions_with_depth_ordered_and_matchcheck(tmp_path):
    cx = _cx(tmp_path)
    dims = list_dimensions(cx)
    keys = [d["key"] for d in dims]
    assert DEPTH_KEY in keys
    assert len(keys) >= 14
    depth = next(d for d in dims if d["key"] == DEPTH_KEY)
    assert depth["has_match_check"] == 1 and depth["ordered"] == 1
    vals = dimension_values(cx, DEPTH_KEY)
    assert [v["rank"] for v in vals] == [1, 2, 3, 4, 5]
    assert vals[0]["value"].startswith("Gut") and vals[4]["code"] == "nucleus"


def test_seed_is_idempotent(tmp_path):
    cx = _cx(tmp_path)
    seed_dimensions(cx)
    assert len([d for d in list_dimensions(cx) if d["key"] == DEPTH_KEY]) == 1
    assert len(dimension_values(cx, DEPTH_KEY)) == 5


def test_tag_roundtrip(tmp_path):
    cx = _cx(tmp_path)
    tag(cx, "auth_stress", "5", DEPTH_KEY, 4)
    assert get_tag(cx, "auth_stress", "5", DEPTH_KEY) == 4
    tag(cx, "auth_stress", "5", DEPTH_KEY, 2)        # update in place
    assert get_tag(cx, "auth_stress", "5", DEPTH_KEY) == 2
    assert get_tag(cx, "auth_stress", "9", DEPTH_KEY) is None


def test_depth_match_logic():
    assert depth_match(3, 5) == "ok"        # remedy reaches deeper than the stress
    assert depth_match(5, 5) == "ok"
    assert depth_match(5, 3) == "shallow"   # remedy falls short of the stress depth
    assert depth_match(None, 5) == "unknown"
    assert depth_match(5, None) == "unknown"


def test_depth_label(tmp_path):
    cx = _cx(tmp_path)
    assert depth_label(cx, 5).lower().startswith("nucle")
    assert depth_label(cx, 1).startswith("Gut")
    assert depth_label(cx, None) == ""
