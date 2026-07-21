import os, pytest, threading
from dashboard import db

pg = bool(os.environ.get("PG_DSN"))

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_parity_roundtrip(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS parity_t")
    cx.execute("CREATE TABLE parity_t (id BIGINT PRIMARY KEY, v TEXT)")
    cx.execute("INSERT INTO parity_t (id, v) VALUES (?, ?)", (1, "hi"))
    cx.commit()
    row = cx.execute("SELECT id, v FROM parity_t WHERE id=?", (1,)).fetchone()
    assert row[0] == 1 and row["v"] == "hi"
    cx.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_rotate_race_no_lost_update(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS rota")
    cx.execute("CREATE TABLE rota (id BIGINT PRIMARY KEY, n BIGINT)")
    cx.execute("INSERT INTO rota (id, n) VALUES (1, 0)")
    cx.commit()
    WRITERS, ITERS = 6, 20
    errors = []
    def worker():
        try:
            c = db.connect("ignored")
            for _ in range(ITERS):
                row = c.execute("SELECT n FROM rota WHERE id=? FOR UPDATE", (1,)).fetchone()
                c.execute("UPDATE rota SET n=? WHERE id=?", (row[0] + 1, 1))
                c.commit()
            c.close()
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=worker) for _ in range(WRITERS)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    final = cx.execute("SELECT n FROM rota WHERE id=?", (1,)).fetchone()[0]
    assert not errors, errors[:2]
    assert final == WRITERS * ITERS, f"lost updates: {final} != {WRITERS*ITERS}"
    cx.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_context_manager_commits_on_success(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    with db.connect("ignored") as cx:
        cx.execute("DROP TABLE IF EXISTS cm_t")
        cx.execute("CREATE TABLE cm_t (id BIGINT)")
        cx.execute("INSERT INTO cm_t (id) VALUES (?)", (42,))
    cx2 = db.connect("ignored")
    assert cx2.execute("SELECT id FROM cm_t").fetchone()[0] == 42
    cx2.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_context_manager_rolls_back_on_exception(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    with db.connect("ignored") as cx:
        cx.execute("DROP TABLE IF EXISTS cm_r")
        cx.execute("CREATE TABLE cm_r (id BIGINT)")
        cx.commit()
    try:
        with db.connect("ignored") as cx:
            cx.execute("INSERT INTO cm_r (id) VALUES (?)", (1,))
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    cx3 = db.connect("ignored")
    assert cx3.execute("SELECT count(*) FROM cm_r").fetchone()[0] == 0
    cx3.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_schema_isolation(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    a = db.connect("/data/iso_a.db")
    b = db.connect("/data/iso_b.db")
    for cx in (a, b):
        cx.execute("CREATE TABLE IF NOT EXISTS t (id BIGINT)")
        cx.execute("DELETE FROM t")
        cx.commit()
    a.execute("INSERT INTO t (id) VALUES (?)", (1,))
    a.commit()
    # same unqualified table name, different schema -> b's t is still empty
    assert b.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    a.close()
    b.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_pool_reuses_backends(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    seen = set()
    for _ in range(20):
        with db.connect("/data/pooltest.db") as cx:
            seen.add(cx.execute("SELECT pg_backend_pid()").fetchone()[0])
    assert len(seen) < 20  # pooled: far fewer backends than checkouts


def test_backend_of_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_BACKEND", raising=False)
    cx = db.connect(str(tmp_path / "b.db"))
    assert db.backend_of(cx) == "sqlite"
    cx.close()


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_backend_of_postgres(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("/data/backendof.db")
    assert db.backend_of(cx) == "postgres"
    cx.close()


def test_backend_of_untagged_is_sqlite():
    import sqlite3
    raw = sqlite3.connect(":memory:")
    assert db.backend_of(raw) == "sqlite"
    raw.close()


def test_column_exists_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_BACKEND", raising=False)
    cx = db.connect(str(tmp_path / "colexists.db"))
    cx.execute("CREATE TABLE t (known TEXT)")
    cx.commit()
    assert db.column_exists(cx, "t", "known") is True
    assert db.column_exists(cx, "t", "missing") is False
    cx.close()


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_column_exists_postgres(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("/data/colexists_pg.db")
    cx.execute("DROP TABLE IF EXISTS t")
    cx.execute("CREATE TABLE t (known TEXT)")
    cx.commit()
    assert db.column_exists(cx, "t", "known") is True
    assert db.column_exists(cx, "t", "missing") is False
    cx.close()


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_cursor_is_iterable(monkeypatch):
    # Regression: `for row in cx.execute(...)` (121 sites) must work on Postgres.
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("/data/iter_t.db")
    cx.execute("DROP TABLE IF EXISTS iter_t")
    cx.execute("CREATE TABLE iter_t (id BIGINT, v TEXT)")
    cx.execute("INSERT INTO iter_t (id, v) VALUES (?, ?)", (1, "a"))
    cx.execute("INSERT INTO iter_t (id, v) VALUES (?, ?)", (2, "b"))
    cx.commit()
    rows = [r for r in cx.execute("SELECT id, v FROM iter_t ORDER BY id")]
    assert len(rows) == 2
    assert rows[0][0] == 1 and rows[0]["v"] == "a"
    assert [r[0] for r in cx.execute("SELECT id FROM iter_t ORDER BY id")] == [1, 2]
    cx.close()


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_executescript_multi_statement(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS es_a"); cx.execute("DROP TABLE IF EXISTS es_b")
    cx.commit()
    # A ';'-separated DDL script (as sqlite3.executescript takes) with an
    # AUTOINCREMENT idiom and a DEFAULT to prove per-statement translation.
    cx.executescript("""
        CREATE TABLE es_a (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT DEFAULT '');
        CREATE TABLE es_b (id BIGINT PRIMARY KEY);
        CREATE INDEX IF NOT EXISTS idx_es_b ON es_b(id);
    """)
    cx.commit()
    for t in ("es_a", "es_b"):
        assert cx.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema=current_schema() AND table_name=?", (t,)).fetchone() is not None
    cx.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_rowcount_and_executemany(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS rc_t")
    cx.execute("CREATE TABLE rc_t (id BIGINT PRIMARY KEY, v TEXT)")
    cx.executemany("INSERT INTO rc_t (id, v) VALUES (?, ?)", [(1, "a"), (2, "b"), (3, "c")])
    cx.commit()
    cur = cx.execute("UPDATE rc_t SET v='x' WHERE id<=?", (2,))
    assert cur.rowcount == 2
    assert cx.execute("SELECT COUNT(*) FROM rc_t").fetchone()[0] == 3
    cx.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_add_column_idempotent_and_missing_table_noop(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS ac_t")
    cx.execute("CREATE TABLE ac_t (id BIGINT)")
    cx.commit()
    # 1) ADD COLUMN twice -> IF NOT EXISTS makes the 2nd a no-op (no DuplicateColumn)
    cx.execute("ALTER TABLE ac_t ADD COLUMN extra TEXT")
    cx.execute("ALTER TABLE ac_t ADD COLUMN extra TEXT")
    cx.commit()
    # 2) ADD COLUMN on a table that doesn't exist -> IF EXISTS makes it a silent
    #    no-op (mirrors SQLite's swallowed 'no such table' in the migration try-blocks)
    cx.execute("ALTER TABLE ac_missing_table ADD COLUMN whatever TEXT")
    cx.commit()
    assert db.column_exists(cx, "ac_t", "extra") is True
    cx.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_lastrowid_raises_clear_error(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cur = cx.execute("SELECT 1")
    with pytest.raises(AttributeError, match="RETURNING"):
        _ = cur.lastrowid
    cx.close()
