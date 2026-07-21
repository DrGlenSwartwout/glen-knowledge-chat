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
