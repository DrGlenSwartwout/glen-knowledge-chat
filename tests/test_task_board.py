"""Unit tests for dashboard.task_board (the console-side snapshot store)."""
import sqlite3

from dashboard import task_board as tb


def _cx():
    cx = sqlite3.connect(":memory:")
    tb.init_task_board_table(cx)
    return cx


def test_upsert_then_get_roundtrips_cards():
    cx = _cx()
    cards = [{"id": "a1", "title": "x", "auto_lane": "next"},
             {"id": "b2", "title": "y", "auto_lane": "done", "terminal": True}]
    n = tb.upsert_board(cx, cards, "2026-07-12T10:00:00-10:00")
    assert n == 2
    got = tb.get_board(cx)
    assert got["generated_at"] == "2026-07-12T10:00:00-10:00"
    assert len(got["cards"]) == 2
    assert got["cards"][1]["terminal"] is True
    assert got["synced_at"]  # stamped


def test_upsert_replaces_previous_snapshot():
    cx = _cx()
    tb.upsert_board(cx, [{"id": "a1", "title": "old"}], "t1")
    tb.upsert_board(cx, [{"id": "z9", "title": "new"}], "t2")
    got = tb.get_board(cx)
    assert got["generated_at"] == "t2"
    assert [c["id"] for c in got["cards"]] == ["z9"]  # not appended, replaced


def test_get_empty_board_is_safe():
    cx = _cx()
    got = tb.get_board(cx)
    assert got["cards"] == []
    assert got["generated_at"] is None
