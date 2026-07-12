"""Formulation-map curation CRUD: append-at-bottom, idempotent, remove, reorder."""
import sqlite3

from dashboard.formulation_map import (
    init_tables, add_mapping, remove_mapping, reorder, mappings_for)


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "e4l.db"))
    init_tables(cx)
    return cx


def test_add_appends_at_bottom_and_creates_formulation(tmp_path):
    cx = _cx(tmp_path)
    add_mapping(cx, "ED5", "Heart Health")
    add_mapping(cx, "ED5", "Circulation Support")
    m = mappings_for(cx, "ED5")
    assert [x["name"] for x in m] == ["Heart Health", "Circulation Support"]
    assert [x["priority"] for x in m] == [1, 2]          # appended in order
    # a brand-new remedy name got a formulations row
    assert cx.execute("SELECT COUNT(*) FROM formulations WHERE name='Heart Health'").fetchone()[0] == 1


def test_add_is_idempotent(tmp_path):
    cx = _cx(tmp_path)
    add_mapping(cx, "ED5", "Heart Health")
    add_mapping(cx, "ED5", "heart health")               # case-insensitive dup -> no change
    m = mappings_for(cx, "ED5")
    assert len(m) == 1 and m[0]["priority"] == 1


def test_remove_individual(tmp_path):
    cx = _cx(tmp_path)
    add_mapping(cx, "ED5", "Heart Health")
    add_mapping(cx, "ED5", "Circulation Support")
    fid = mappings_for(cx, "ED5")[0]["formulation_id"]
    remove_mapping(cx, "ED5", fid)
    assert [x["name"] for x in mappings_for(cx, "ED5")] == ["Circulation Support"]


def test_reorder_sets_priority(tmp_path):
    cx = _cx(tmp_path)
    add_mapping(cx, "ED5", "Heart Health")
    add_mapping(cx, "ED5", "Circulation Support")
    ids = [x["formulation_id"] for x in mappings_for(cx, "ED5")]
    reorder(cx, "ED5", [ids[1], ids[0]])                 # flip the order
    m = mappings_for(cx, "ED5")
    assert [x["name"] for x in m] == ["Circulation Support", "Heart Health"]
    assert [x["priority"] for x in m] == [1, 2]
