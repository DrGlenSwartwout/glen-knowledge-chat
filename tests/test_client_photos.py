import sqlite3
from dashboard import client_photos as cp


def _cx():
    return sqlite3.connect(":memory:")


def test_put_then_get_roundtrip_by_email():
    cx = _cx()
    blob = b"\xff\xd8\xff\xe0JFIF-fake-bytes"
    assert cp.put(cx, "Michael@Example.com ", blob, "image/png", source="fmp") == "michael@example.com"
    got = cp.get(cx, "michael@example.com")
    assert got["blob"] == blob
    assert got["content_type"] == "image/png"
    assert cp.has(cx, "MICHAEL@example.com") is True


def test_get_absent_returns_none():
    cx = _cx()
    assert cp.get(cx, "nobody@example.com") is None
    assert cp.has(cx, "nobody@example.com") is False
    assert cp.get(cx, "") is None


def test_put_upserts_not_duplicates():
    cx = _cx()
    cp.put(cx, "a@b.com", b"first", "image/jpeg")
    cp.put(cx, "a@b.com", b"second", "image/webp")
    got = cp.get(cx, "a@b.com")
    assert got["blob"] == b"second"
    assert got["content_type"] == "image/webp"
    n = cx.execute("SELECT COUNT(*) FROM client_photos WHERE email='a@b.com'").fetchone()[0]
    assert n == 1


def test_put_without_email_or_blob_is_noop():
    cx = _cx()
    cp.init_table(cx)
    assert cp.put(cx, "", b"x", "image/png") is None
    assert cp.put(cx, "a@b.com", b"", "image/png") is None
    assert cx.execute("SELECT COUNT(*) FROM client_photos").fetchone()[0] == 0
