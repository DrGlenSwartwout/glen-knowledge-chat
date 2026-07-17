"""Tests for Task 2: skipped-settler -> console todo.

When dashboard.order_settlement.settle_paid_order_effects reports a skipped
per-kind effect (a settler raised and was swallowed best-effort, then the
order was marked settled and will NOT retry), app._raise_settlement_skip_todo
must raise ONE deduped console todo so the stranded effect is visible +
actionable. It must never raise into the caller.
"""
import sqlite3

import app as appmod


def _db(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    return db


def test_skip_todo_inserted_with_expected_columns(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    appmod._raise_settlement_skip_todo("tok1", "subscribe", ["subscription"])

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    try:
        rows = cx.execute("SELECT * FROM todos WHERE dedup_key=?", ("settle-skip:tok1",)).fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["owner"] == "glen"
        assert row["category"] == "Fulfillment"
        assert row["source"] == "settlement-skip"
        assert row["priority"] == "high"
        assert row["dedup_key"] == "settle-skip:tok1"
        assert "tok1" in row["title"]
        assert "subscription" in row["body"]
    finally:
        cx.close()


def test_skip_todo_idempotent_dedup_key(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    appmod._raise_settlement_skip_todo("tok1", "subscribe", ["subscription"])
    appmod._raise_settlement_skip_todo("tok1", "subscribe", ["subscription"])

    cx = sqlite3.connect(db)
    try:
        n = cx.execute(
            "SELECT COUNT(*) FROM todos WHERE dedup_key=?", ("settle-skip:tok1",)
        ).fetchone()[0]
        assert n == 1
    finally:
        cx.close()


def test_skip_todo_swallows_db_error(monkeypatch, tmp_path):
    # Point LOG_DB at a path whose parent directory doesn't exist, so sqlite3.connect
    # raises. The helper must swallow this and never propagate.
    bad_db = str(tmp_path / "no-such-dir" / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", bad_db)
    # Must not raise.
    appmod._raise_settlement_skip_todo("tok2", "biofield", ["biofield"])


def test_skip_todo_noop_when_skipped_empty(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    appmod._init_todos_table()  # pre-create so we can assert no row was added
    appmod._raise_settlement_skip_todo("tok3", "retail", [])

    cx = sqlite3.connect(db)
    try:
        n = cx.execute("SELECT COUNT(*) FROM todos").fetchone()[0]
        assert n == 0
    finally:
        cx.close()
