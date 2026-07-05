import sqlite3
from dashboard import community as _c


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _c.init_feed_tables(cx)
    return cx


def test_embedding_roundtrip():
    cx = _cx()
    _c.set_embedding(cx, 5, [0.1, 0.2, 0.3], "ada")
    assert _c.get_embeddings(cx, [5], "ada") == {5: [0.1, 0.2, 0.3]}


def test_get_embeddings_skips_other_model():
    cx = _cx()
    _c.set_embedding(cx, 5, [0.1], "ada")
    assert _c.get_embeddings(cx, [5], "newmodel") == {}   # model mismatch → re-embed
    assert _c.get_embeddings(cx, [5], "ada") == {5: [0.1]}


def test_set_embedding_upserts():
    cx = _cx()
    _c.set_embedding(cx, 5, [0.1], "ada")
    _c.set_embedding(cx, 5, [0.9], "ada")
    assert _c.get_embeddings(cx, [5], "ada") == {5: [0.9]}  # replaced, not duplicated


def test_member_interest_roundtrip_and_model_guard():
    cx = _cx()
    assert _c.get_member_interest(cx, "A@B.com", "ada") is None
    _c.set_member_interest(cx, "A@B.com", [0.4, 0.5], "ada")
    got = _c.get_member_interest(cx, "a@b.com", "ada")
    assert got["vec"] == [0.4, 0.5] and got["built_at"]
    assert _c.get_member_interest(cx, "a@b.com", "othermodel") is None  # stale model
