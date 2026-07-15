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


def test_precedence_fmp_does_not_overwrite_portal_self():
    cx = _cx()
    cp.put(cx, "a@b.com", b"client-chosen", "image/png", source="portal-self")
    # bulk fmp write must NOT clobber the client's own photo
    assert cp.put(cx, "a@b.com", b"from-fmp", "image/jpeg", source="fmp", force=False) is None
    got = cp.get(cx, "a@b.com")
    assert got["blob"] == b"client-chosen" and got["content_type"] == "image/png"


def test_precedence_fmp_overwrites_lower_and_equal():
    cx = _cx()
    cp.put(cx, "a@b.com", b"ghl-img", "image/png", source="ghl")
    assert cp.put(cx, "a@b.com", b"fmp-img", "image/jpeg", source="fmp", force=False) == "a@b.com"
    assert cp.get(cx, "a@b.com")["blob"] == b"fmp-img"          # fmp(2) > ghl(1)
    assert cp.put(cx, "a@b.com", b"fmp-2", "image/jpeg", source="fmp", force=False) == "a@b.com"
    assert cp.get(cx, "a@b.com")["blob"] == b"fmp-2"            # fmp == fmp, still writes


def test_precedence_fmp_writes_when_absent():
    cx = _cx()
    assert cp.put(cx, "a@b.com", b"fmp-img", "image/jpeg", source="fmp", force=False) == "a@b.com"


def test_force_true_default_always_writes():
    cx = _cx()
    cp.put(cx, "a@b.com", b"client", "image/png", source="portal-self")
    # a deliberate operator upload (default force=True) overwrites even portal-self
    assert cp.put(cx, "a@b.com", b"operator", "image/png", source="console") == "a@b.com"
    assert cp.get(cx, "a@b.com")["blob"] == b"operator"


def test_would_skip_precedence():
    cx = _cx()
    assert cp.would_skip_precedence(cx, "a@b.com", "fmp") is False        # absent -> no skip
    cp.put(cx, "a@b.com", b"x", "image/png", source="ghl")
    assert cp.would_skip_precedence(cx, "a@b.com", "fmp") is False        # fmp(2) > ghl(1)
    cp.put(cx, "a@b.com", b"y", "image/png", source="portal-self")
    assert cp.would_skip_precedence(cx, "a@b.com", "fmp") is True         # portal-self(4) > fmp(2)
