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
    """Test scan_db aggregation with real collision data via file-path entry point."""
    p = str(tmp_path / "collide.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        # Table with unique index and NULL-based collision
        "CREATE TABLE users (id INTEGER PRIMARY KEY, code TEXT);"
        "CREATE UNIQUE INDEX ux_users_code ON users (code);"
        "INSERT INTO users (code) VALUES (NULL), (NULL), ('ABC');"
        # Clean table with unique index, no collisions
        "CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT);"
        "CREATE UNIQUE INDEX ux_products_sku ON products (sku);"
        "INSERT INTO products (sku) VALUES ('SKU1'), ('SKU2');"
    )
    cx.close()

    # Call the file-path entry point (the aggregation function)
    findings = dedup.scan_db(p)

    # Should find exactly one table with duplicates
    assert len(findings) == 1
    assert findings[0]["table"] == "users"
    assert findings[0]["key_cols"] == ["code"]
    assert findings[0]["n_groups"] == 1  # One duplicate group (the two NULLs)
    assert findings[0]["n_excess_rows"] == 1  # Two NULLs = 1 excess row
    # examples should contain the NULL key
    assert any(ex["key"] == [None] for ex in findings[0]["examples"])
