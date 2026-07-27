import sqlite3
from dashboard import purity_photos as pp


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    pp.init_table(cx); return cx


def test_save_and_get_roundtrips_bytes():
    cx = _cx()
    assert pp.save(cx, "fullscript::x", "a@b.com", b"\xff\xd8\xff-image", "image/jpeg") is True
    row = pp.get(cx, "fullscript::x")
    assert bytes(row["image_blob"]) == b"\xff\xd8\xff-image"
    assert row["content_type"] == "image/jpeg" and row["email"] == "a@b.com"


def test_save_upserts_latest():
    cx = _cx()
    pp.save(cx, "fullscript::x", "a@b.com", b"first", "image/png")
    pp.save(cx, "fullscript::x", "c@d.com", b"second", "image/jpeg")
    row = pp.get(cx, "fullscript::x")
    assert bytes(row["image_blob"]) == b"second" and row["email"] == "c@d.com"


def test_save_rejects_empty():
    cx = _cx()
    assert pp.save(cx, "", "a@b.com", b"x", "image/png") is False
    assert pp.save(cx, "fullscript::x", "a@b.com", b"", "image/png") is False


def test_get_missing_is_none():
    assert pp.get(_cx(), "fullscript::nope") is None


def test_keys_with_photos():
    cx = _cx()
    pp.save(cx, "fullscript::a", "e", b"1", "image/png")
    pp.save(cx, "fullscript::b", "e", b"2", "image/png")
    assert pp.keys_with_photos(cx) == {"fullscript::a", "fullscript::b"}
