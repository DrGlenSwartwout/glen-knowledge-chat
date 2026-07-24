import sqlite3
from dashboard import client_documents as cd


def _cx():
    cx = sqlite3.connect(":memory:")
    cd.init_table(cx)
    return cx


def test_bytea_column_round_trips_all_byte_values():
    """Pins the BYTEA choice: BLOB would fail on Postgres, and BYTEA must still
    carry every byte value (including NUL) losslessly on SQLite."""
    cx = _cx()
    blob = bytes(range(256)) * 40
    r = cd.put(cx, "c@x.com", blob, "scan.pdf", "application/pdf", "console")
    got = cd.get(cx, r["id"])
    assert got["blob"] == blob
    assert got["byte_size"] == len(blob)


def test_put_then_get_round_trip_fields():
    cx = _cx()
    r = cd.put(cx, "C@X.com", b"hello", "labs.pdf", "application/pdf", "portal-self")
    assert r["deduped"] is False
    got = cd.get(cx, r["id"])
    assert got["email"] == "c@x.com"          # lowercased
    assert got["filename"] == "labs.pdf"
    assert got["content_type"] == "application/pdf"
    assert got["source"] == "portal-self"
    assert got["extract_status"] == "pending"
    assert got["uploaded_at"]


def test_identical_bytes_for_same_email_dedup_to_one_row():
    cx = _cx()
    a = cd.put(cx, "c@x.com", b"same", "a.pdf", "application/pdf", "console")
    b = cd.put(cx, "c@x.com", b"same", "b.pdf", "application/pdf", "portal-self")
    assert b["deduped"] is True
    assert a["id"] == b["id"]
    assert len(cd.list_for_email(cx, "c@x.com")) == 1


def test_same_bytes_different_email_are_separate_rows():
    cx = _cx()
    a = cd.put(cx, "a@x.com", b"same", "a.pdf", "application/pdf", "console")
    b = cd.put(cx, "b@x.com", b"same", "a.pdf", "application/pdf", "console")
    assert a["id"] != b["id"]


def test_list_for_email_excludes_blob():
    cx = _cx()
    cd.put(cx, "c@x.com", b"bytes", "a.pdf", "application/pdf", "console")
    rows = cd.list_for_email(cx, "c@x.com")
    assert len(rows) == 1
    assert "blob" not in rows[0]


def test_get_for_email_is_the_scoping_primitive():
    cx = _cx()
    r = cd.put(cx, "owner@x.com", b"bytes", "a.pdf", "application/pdf", "console")
    assert cd.get_for_email(cx, r["id"], "owner@x.com") is not None
    assert cd.get_for_email(cx, r["id"], "other@x.com") is None


def test_put_rejects_empty_email_or_blob():
    cx = _cx()
    assert cd.put(cx, "", b"x", "a.pdf", "application/pdf", "console") is None
    assert cd.put(cx, "c@x.com", b"", "a.pdf", "application/pdf", "console") is None


def test_set_extract_status_and_pending():
    cx = _cx()
    r = cd.put(cx, "c@x.com", b"one", "a.pdf", "application/pdf", "console")
    assert [p["id"] for p in cd.pending(cx)] == [r["id"]]
    cd.set_extract_status(cx, r["id"], "drafted")
    assert cd.pending(cx) == []
    assert cd.get(cx, r["id"])["extract_status"] == "drafted"


def test_pending_excludes_unreadable():
    cx = _cx()
    r = cd.put(cx, "c@x.com", b"one", "a.zip", "application/zip", "console")
    cd.set_extract_status(cx, r["id"], "skipped-unreadable")
    assert cd.pending(cx) == []
