import os, pytest
from dashboard import db
from dashboard.dbwrite import insert_or_ignore

pg = bool(os.environ.get("PG_DSN"))

def _setup(cx, with_index):
    cx.execute("DROP TABLE IF EXISTS w")
    cx.execute("CREATE TABLE w (k TEXT, v TEXT)")
    if with_index:
        cx.execute("CREATE UNIQUE INDEX ux_w_k ON w (k)")
    cx.commit()

def _idempotent(cx):
    insert_or_ignore(cx, "w", ["k", "v"], ("a", "1"), conflict_cols=["k"])
    insert_or_ignore(cx, "w", ["k", "v"], ("a", "2"), conflict_cols=["k"])  # ignored
    cx.commit()
    assert cx.execute("SELECT COUNT(*) FROM w").fetchone()[0] == 1
    assert cx.execute("SELECT v FROM w WHERE k=?", ("a",)).fetchone()[0] == "1"

def test_sqlite_insert_or_ignore(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_BACKEND", raising=False)
    cx = db.connect(str(tmp_path / "w.db"))
    _setup(cx, with_index=True)
    _idempotent(cx)
    cx.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_postgres_insert_or_ignore(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("/data/dbwrite.db")
    _setup(cx, with_index=True)
    _idempotent(cx)
    cx.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_postgres_missing_index_raises_clear_error(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("/data/dbwrite.db")
    _setup(cx, with_index=False)  # NO unique index -> ON CONFLICT would be 42P10
    with pytest.raises(RuntimeError) as ei:
        insert_or_ignore(cx, "w", ["k", "v"], ("a", "1"), conflict_cols=["k"])
    msg = str(ei.value).lower()
    assert "unique index" in msg and "w" in msg   # clear + names the table
    cx.rollback()  # transaction was aborted by the failed statement
    cx.close()


import pytest
_pg = bool(__import__("os").environ.get("PG_DSN"))


def test_insert_or_replace_sqlite(tmp_path):
    import sqlite3
    from dashboard import dbwrite
    dbp = str(tmp_path / "t.db")
    cx = sqlite3.connect(dbp)
    cx.execute("CREATE TABLE t (k TEXT PRIMARY KEY, v TEXT, w TEXT)")
    dbwrite.insert_or_replace(cx, "t", ("k", "v", "w"), ("a", "1", "x"), conflict_cols=("k",))
    dbwrite.insert_or_replace(cx, "t", ("k", "v", "w"), ("a", "2", "y"), conflict_cols=("k",))
    cx.commit()
    assert cx.execute("SELECT v, w FROM t WHERE k='a'").fetchone() == ("2", "y")
    assert cx.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1


@pytest.mark.skipif(not _pg, reason="PG_DSN not set")
def test_insert_or_replace_pg_on_pk(monkeypatch):
    from dashboard import db, dbwrite
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS ior_t")
    cx.execute("CREATE TABLE ior_t (k TEXT PRIMARY KEY, v TEXT, w TEXT)")
    dbwrite.insert_or_replace(cx, "ior_t", ("k", "v", "w"), ("a", "1", "x"), conflict_cols=("k",))
    dbwrite.insert_or_replace(cx, "ior_t", ("k", "v", "w"), ("a", "2", "y"), conflict_cols=("k",))
    cx.commit()
    assert cx.execute("SELECT v, w FROM ior_t WHERE k=?", ("a",)).fetchone()[:] == ("2", "y")
    assert cx.execute("SELECT COUNT(*) FROM ior_t").fetchone()[0] == 1
    cx.close()


@pytest.mark.skipif(not _pg, reason="PG_DSN not set")
def test_insert_or_replace_pg_on_secondary_unique(monkeypatch):
    # Mirrors inquiry_reply_tokens: PK is a fresh random hash; dedup key is a
    # SEPARATE unique constraint. Two inserts with different PK but same unique
    # key must collapse to one row carrying the second PK.
    from dashboard import db, dbwrite
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS tok_t")
    cx.execute("CREATE TABLE tok_t (token_hash TEXT PRIMARY KEY, inquiry_id TEXT, "
               "practitioner_id TEXT, created_at TEXT, UNIQUE(inquiry_id, practitioner_id))")
    cols = ("token_hash", "inquiry_id", "practitioner_id", "created_at")
    dbwrite.insert_or_replace(cx, "tok_t", cols, ("hash1", "inq1", "p1", "t1"),
                              conflict_cols=("inquiry_id", "practitioner_id"))
    dbwrite.insert_or_replace(cx, "tok_t", cols, ("hash2", "inq1", "p1", "t2"),
                              conflict_cols=("inquiry_id", "practitioner_id"))
    cx.commit()
    rows = cx.execute("SELECT token_hash, created_at FROM tok_t").fetchall()
    assert len(rows) == 1 and rows[0][0] == "hash2" and rows[0][1] == "t2"
    cx.close()
