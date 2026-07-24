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


# --- CRITICAL 2: client_visible=0 must not silently revert. init_table's
# backfill exists ONLY to give rows that predate the client_visible column a
# sane value -- it must touch ONLY rows left NULL by the ADD COLUMN
# migration, by source, and never an explicit 0/1 written later. ---

def test_client_visible_is_a_proper_bool_never_a_raw_int():
    cx = _cx()
    r = cd.put(cx, "c@x.com", b"one", "a.pdf", "application/pdf", "portal-self")
    assert cd.get(cx, r["id"])["client_visible"] is True
    assert cd.list_for_email(cx, "c@x.com")[0]["client_visible"] is True


def test_legacy_null_client_visible_is_backfilled_by_source_on_first_touch():
    """Simulates a genuinely pre-migration row: the table exists WITHOUT the
    client_visible column at all (as it did before this feature shipped), so
    init_table's ALTER TABLE ADD COLUMN leaves existing rows NULL rather than
    defaulted. The backfill must resolve NULL -> visible for portal-self and
    NULL -> hidden for everything else, on the very first init_table() call
    that sees them (i.e. the first store call touching this pre-migration
    table)."""
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE client_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL, filename TEXT, content_type TEXT,
        byte_size INTEGER, sha256 TEXT, blob BYTEA, source TEXT,
        uploaded_at TEXT, extract_status TEXT)""")
    cx.execute("INSERT INTO client_documents(email, filename, content_type, "
               "byte_size, sha256, blob, source, uploaded_at, extract_status) "
               "VALUES('a@x.com','f.pdf','application/pdf',1,'h1',x'00',"
               "'portal-self','t','pending')")
    cx.execute("INSERT INTO client_documents(email, filename, content_type, "
               "byte_size, sha256, blob, source, uploaded_at, extract_status) "
               "VALUES('b@x.com','f.pdf','application/pdf',1,'h2',x'00',"
               "'console','t','pending')")
    cx.commit()

    cd.init_table(cx)  # runs the ALTER (-> NULL) + the IS-NULL backfill

    rows = {r[0]: r[1] for r in cx.execute(
        "SELECT source, client_visible FROM client_documents").fetchall()}
    assert rows["portal-self"] == 1   # backfilled visible
    assert rows["console"] == 0       # backfilled hidden


def test_explicit_hide_is_never_reverted_by_a_later_init_table_call():
    """The exact CRITICAL 2 repro at the store level: a portal-self document
    starts visible; an explicit hide (set_client_visible(False), what the
    visibility route calls) must stick through any number of subsequent
    store calls, each of which re-runs init_table()'s backfill."""
    cx = _cx()
    r = cd.put(cx, "c@x.com", b"one", "a.pdf", "application/pdf", "portal-self")
    assert cd.get(cx, r["id"])["client_visible"] is True

    cd.set_client_visible(cx, r["id"], False)
    assert cd.get(cx, r["id"])["client_visible"] is False

    # Several more store calls, each re-running init_table()'s backfill --
    # this is exactly the path that previously reverted the hide.
    cd.get(cx, r["id"])
    cd.list_for_email(cx, "c@x.com")
    cd.pending(cx)
    assert cd.get(cx, r["id"])["client_visible"] is False


def test_reshowing_after_a_hide_works_and_also_sticks():
    cx = _cx()
    r = cd.put(cx, "c@x.com", b"one", "a.pdf", "application/pdf", "portal-self")
    cd.set_client_visible(cx, r["id"], False)
    cd.set_client_visible(cx, r["id"], True)
    cd.list_for_email(cx, "c@x.com")  # re-run the backfill again
    assert cd.get(cx, r["id"])["client_visible"] is True
