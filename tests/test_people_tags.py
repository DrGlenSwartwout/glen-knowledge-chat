"""Unit tests for the pure set_person_tags helper (no DB, no Flask, no env)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.people import set_person_tags, MAX_TAG_LEN


def test_add_to_empty():
    assert set_person_tags([], add=["tier:pro-influencer"]) == ["tier:pro-influencer"]


def test_add_to_populated_appends():
    assert set_person_tags(["type:client"], add=["OD"]) == ["type:client", "OD"]


def test_remove_existing():
    assert set_person_tags(["type:client", "OD"], remove=["OD"]) == ["type:client"]


def test_remove_absent_is_noop():
    assert set_person_tags(["type:client"], remove=["nope"]) == ["type:client"]


def test_add_and_remove_same_tag_keeps_it():
    # remove runs first, then add re-adds
    assert set_person_tags(["x"], add=["y"], remove=["y", "x"]) == ["y"]


def test_adding_existing_is_noop_dedup():
    assert set_person_tags(["type:client"], add=["type:client"]) == ["type:client"]


def test_duplicate_inputs_collapse():
    assert set_person_tags([], add=["a", "a", "b"]) == ["a", "b"]


def test_whitespace_trimmed_and_empty_dropped():
    assert set_person_tags([], add=["  spaced  ", "   ", ""]) == ["spaced"]


def test_overlong_add_dropped():
    long = "x" * (MAX_TAG_LEN + 1)
    assert set_person_tags([], add=[long, "ok"]) == ["ok"]


def test_case_sensitive_distinct():
    assert set_person_tags(["OD"], add=["od"]) == ["OD", "od"]


def test_existing_order_preserved_minus_removals():
    assert set_person_tags(["a", "b", "c"], remove=["b"], add=["d"]) == ["a", "c", "d"]


def test_non_string_inputs_ignored():
    assert set_person_tags(["a", 5, None], add=[7, "b"]) == ["a", "b"]


from dashboard.people import distinct_tags


def test_distinct_union_and_sorted():
    rows = [["type:client", "OD"], ["type:client", "tier:pro-influencer"]]
    assert distinct_tags(rows) == ["OD", "tier:pro-influencer", "type:client"]


def test_distinct_accepts_json_strings():
    rows = ['["a", "b"]', '["b", "c"]']
    assert distinct_tags(rows) == ["a", "b", "c"]


def test_distinct_mixed_list_and_string():
    rows = [["a"], '["b"]']
    assert distinct_tags(rows) == ["a", "b"]


def test_distinct_skips_malformed_json():
    rows = ['["a"]', "not json", None, 5]
    assert distinct_tags(rows) == ["a"]


def test_distinct_strips_and_drops_empty():
    rows = [["  a  ", "", "   "]]
    assert distinct_tags(rows) == ["a"]


def test_distinct_case_sensitive_ascii_order():
    assert distinct_tags([["od", "OD"]]) == ["OD", "od"]


def test_distinct_empty():
    assert distinct_tags([]) == []
