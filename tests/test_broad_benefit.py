"""Slice 1: dashboard/broad_benefit.py — sqlite store flagging formulations
that are frequently a good match and broadly beneficial. seed_if_empty must
be idempotent."""
import sqlite3

from dashboard import broad_benefit as bb

SLUGS = ["glutathione-syntropy", "vitamin-c-syntropy", "wholomega"]


def _cx(tmp_db):
    cx = sqlite3.connect(tmp_db)
    cx.row_factory = sqlite3.Row
    return cx


def test_init_table_creates_schema(tmp_db):
    cx = _cx(tmp_db)
    bb.init_table(cx)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(broad_benefit)")}
    assert cols == {"slug", "added_at"}


def test_seed_if_empty_populates(tmp_db):
    cx = _cx(tmp_db)
    bb.init_table(cx)
    bb.seed_if_empty(cx, SLUGS)
    assert set(bb.all_slugs(cx)) == set(SLUGS)


def test_seed_if_empty_idempotent(tmp_db):
    cx = _cx(tmp_db)
    bb.init_table(cx)
    bb.seed_if_empty(cx, SLUGS)
    bb.seed_if_empty(cx, SLUGS)
    assert len(bb.all_slugs(cx)) == len(SLUGS)


def test_seed_if_empty_noop_when_table_nonempty(tmp_db):
    cx = _cx(tmp_db)
    bb.init_table(cx)
    bb.add(cx, "custom-slug")
    bb.seed_if_empty(cx, SLUGS)
    assert bb.all_slugs(cx) == ["custom-slug"]


def test_is_broad_true_and_false(tmp_db):
    cx = _cx(tmp_db)
    bb.init_table(cx)
    bb.seed_if_empty(cx, SLUGS)
    assert bb.is_broad(cx, "wholomega") is True
    assert bb.is_broad(cx, "not-a-real-slug") is False


def test_add_then_is_broad(tmp_db):
    cx = _cx(tmp_db)
    bb.init_table(cx)
    bb.add(cx, "new-slug")
    assert bb.is_broad(cx, "new-slug") is True


def test_add_is_idempotent(tmp_db):
    cx = _cx(tmp_db)
    bb.init_table(cx)
    bb.add(cx, "dup-slug")
    bb.add(cx, "dup-slug")
    assert bb.all_slugs(cx).count("dup-slug") == 1


def test_remove(tmp_db):
    cx = _cx(tmp_db)
    bb.init_table(cx)
    bb.seed_if_empty(cx, SLUGS)
    bb.remove(cx, "wholomega")
    assert bb.is_broad(cx, "wholomega") is False
    assert "wholomega" not in bb.all_slugs(cx)


def test_remove_nonexistent_is_safe(tmp_db):
    cx = _cx(tmp_db)
    bb.init_table(cx)
    bb.remove(cx, "never-existed")  # must not raise
    assert bb.all_slugs(cx) == []
