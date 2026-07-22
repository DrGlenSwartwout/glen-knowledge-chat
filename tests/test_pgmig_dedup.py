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

def test_scan_db_direct_file_path(tmp_path):
    """Test scan_db aggregation via file-path entry point. NULL-key groups
    must NOT be reported as collisions (NULLs are distinct under UNIQUE) --
    this is the exact false-positive shape the real dry-run hit
    (subscriptions.order_ref: 3 NULLs; todos.dedup_key: 6 NULLs). A table
    with only NULL-key rows under a live unique index has ZERO real
    collisions and must produce no finding at all."""
    p = str(tmp_path / "collide.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        # Table with unique index and only NULL keys -- no genuine collision.
        "CREATE TABLE users (id INTEGER PRIMARY KEY, code TEXT);"
        "CREATE UNIQUE INDEX ux_users_code ON users (code);"
        "INSERT INTO users (code) VALUES (NULL), (NULL), (NULL), ('ABC');"
        # Clean table with unique index, no collisions
        "CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT);"
        "CREATE UNIQUE INDEX ux_products_sku ON products (sku);"
        "INSERT INTO products (sku) VALUES ('SKU1'), ('SKU2');"
    )
    cx.close()

    # Call the file-path entry point (the aggregation function)
    findings = dedup.scan_db(p)

    # No genuine collisions anywhere -- the 3 NULLs must NOT be reported.
    assert findings == []


def test_scan_collisions_excludes_null_key_groups(tmp_path):
    """A table where the entire key repeats as NULL must report ZERO
    collision groups -- this is the exact false-positive shape the real
    dry-run hit (subscriptions.order_ref: 3 NULLs; todos.dedup_key: 6 NULLs)."""
    p = str(tmp_path / "nulls.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE subscriptions (id INTEGER PRIMARY KEY, order_ref TEXT);"
        "CREATE UNIQUE INDEX ux_subs_ref ON subscriptions (order_ref);"
        "INSERT INTO subscriptions (order_ref) VALUES (NULL), (NULL), (NULL);"
    )
    cx.commit()
    res = dedup.scan_collisions(cx, "subscriptions", ["order_ref"])
    assert res["n_groups"] == 0
    assert res["n_excess_rows"] == 0
    assert res["examples"] == []
    cx.close()


def test_scan_collisions_still_flags_genuine_non_null_dups(tmp_path):
    """Genuine duplicate NON-NULL keys must still be reported. (No live
    unique index here -- a real UNIQUE index would refuse to admit these
    duplicate non-NULL rows in the first place; this mirrors the 'dirty'
    table pattern used elsewhere in this file for the same reason: the
    scanner's own GROUP BY logic is what's under test, independent of
    whether the source's index actually built.)"""
    p = str(tmp_path / "dups.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT);"
        "INSERT INTO t (k) VALUES ('a'), ('a'), ('b');"
    )
    cx.commit()
    res = dedup.scan_collisions(cx, "t", ["k"])
    assert res["n_groups"] == 1
    assert res["n_excess_rows"] == 1
    assert res["examples"] == [{"key": ["a"], "count": 2}]
    cx.close()


def test_scan_collisions_mixed_nulls_and_genuine_dup(tmp_path):
    """Mixed case: NULL-key rows coexist with a genuine non-NULL dup -- only
    the non-NULL dup is reported."""
    p = str(tmp_path / "mixed.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE todos (id INTEGER PRIMARY KEY, dedup_key TEXT);"
        "INSERT INTO todos (dedup_key) VALUES "
        "(NULL), (NULL), (NULL), (NULL), (NULL), (NULL), ('same'), ('same'), ('unique');"
    )
    cx.commit()
    res = dedup.scan_collisions(cx, "todos", ["dedup_key"])
    assert res["n_groups"] == 1
    assert res["n_excess_rows"] == 1
    assert res["examples"] == [{"key": ["same"], "count": 2}]
    cx.close()
