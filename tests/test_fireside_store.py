import sqlite3
import pytest
from dashboard import fireside_store as fs


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    return cx


def test_get_or_create_creates_then_reuses(tmp_path):
    cx = _cx(tmp_path)
    a = fs.get_or_create(cx, "sess-1")
    assert a["id"] >= 1
    assert a["amg_session"] == "sess-1"
    assert a["turn_count"] == 0
    assert a["transcript"] == []
    assert a["ash_coverage"] == {}
    assert a["ended_at"] is None
    b = fs.get_or_create(cx, "sess-1")
    assert b["id"] == a["id"]  # reused, not duplicated


def test_get_or_create_new_after_ended(tmp_path):
    cx = _cx(tmp_path)
    a = fs.get_or_create(cx, "sess-2")
    fs.mark_ended(cx, a["id"])
    b = fs.get_or_create(cx, "sess-2")
    assert b["id"] != a["id"]  # ended session is not resumed in v1


def test_append_turn_counts_only_traveler(tmp_path):
    cx = _cx(tmp_path)
    s = fs.get_or_create(cx, "sess-3")
    fs.append_turn(cx, s["id"], "traveler", "I'm so tired lately.")
    fs.append_turn(cx, s["id"], "glendalf", "Tell me where you feel it.")
    fs.append_turn(cx, s["id"], "traveler", "In my chest.")
    got = fs.get(cx, s["id"])
    assert got["turn_count"] == 2  # two traveler turns
    assert [t["speaker"] for t in got["transcript"]] == ["traveler", "glendalf", "traveler"]
    assert got["transcript"][0]["text"] == "I'm so tired lately."
    assert got["transcript"][0]["ts"]  # stamped
    assert got["last_turn_at"]


def test_update_coverage_roundtrips(tmp_path):
    cx = _cx(tmp_path)
    s = fs.get_or_create(cx, "sess-4")
    cov = {"summary": "tired, chest-centered", "dimensions": {"symptoms": {"state": "opened"}}}
    fs.update_coverage(cx, s["id"], cov)
    got = fs.get(cx, s["id"])
    assert got["ash_coverage"] == cov


def test_mark_ended_sets_timestamp(tmp_path):
    cx = _cx(tmp_path)
    s = fs.get_or_create(cx, "sess-5")
    assert fs.get(cx, s["id"])["ended_at"] is None
    fs.mark_ended(cx, s["id"])
    assert fs.get(cx, s["id"])["ended_at"]


def test_get_missing_returns_none(tmp_path):
    cx = _cx(tmp_path)
    fs.init_table(cx)
    assert fs.get(cx, 999) is None
