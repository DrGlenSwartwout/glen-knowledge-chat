import sqlite3
from scripts.pgmig import dedup, introspect

def _mk(path):
    cx = sqlite3.connect(path)
    cx.executescript(
        "CREATE TABLE w (k TEXT, v TEXT);"
        "CREATE UNIQUE INDEX ux_w_k ON w (k);"          # unique index present...
        "CREATE TABLE t (id INTEGER PRIMARY KEY, email TEXT);"
        "CREATE UNIQUE INDEX ux_t_email ON t (email);"
    )
    # w: sqlite lets these through only if inserted before the index or via OR IGNORE;
    # simulate a real dirty source by inserting then (the index already exists so we
    # use a table WITHOUT the index for the dup, then assert the scanner finds dups by KEY):
    cx.executescript(
        "CREATE TABLE dirty (k TEXT, v TEXT);"           # no unique index in DDL
        "INSERT INTO dirty (k,v) VALUES ('a','1'),('a','2'),('b','3');"
    )
    cx.execute("INSERT INTO t (email) VALUES ('x@y.com')")
    cx.commit(); cx.close()

def test_scan_collisions_finds_duplicate_keys(tmp_path):
    p = str(tmp_path / "src.db"); _mk(p)
    cx = sqlite3.connect(p)
    res = dedup.scan_collisions(cx, "dirty", ["k"])
    assert res["n_groups"] == 1 and res["n_excess_rows"] == 1   # key 'a' appears twice -> 1 excess
    assert dedup.scan_collisions(cx, "t", ["email"])["n_groups"] == 0
    cx.close()

def test_unique_indexes_introspection(tmp_path):
    p = str(tmp_path / "src.db"); _mk(p)
    cx = sqlite3.connect(p)
    assert ["k"] in introspect.unique_indexes(cx, "w")
    assert ["email"] in introspect.unique_indexes(cx, "t")
    assert "w" in introspect.sqlite_tables(cx) and "sqlite_sequence" not in introspect.sqlite_tables(cx)
    cx.close()
