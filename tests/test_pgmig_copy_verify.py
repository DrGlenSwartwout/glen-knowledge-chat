"""FK-topological SQLite->Postgres copy + row-count parity verify (Task 2, P05).

Skip-guarded on PG_DSN — these tests only run against a real Postgres.
Each test uses an isolated schema keyed off a distinct db_path basename
(schema_for_path derives it) so it never collides with shared app tables.
"""
import os
import sqlite3

import pytest

from scripts.pgmig import copy, verify, introspect
from dashboard import db

pg = bool(os.environ.get("PG_DSN"))


def _mk_sqlite_parent_child(path):
    cx = sqlite3.connect(path)
    cx.executescript(
        "CREATE TABLE parent (id INTEGER PRIMARY KEY, name TEXT);"
        "CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parent(id), tag TEXT);"
    )
    cx.executescript(
        "INSERT INTO parent (id,name) VALUES (1,'a'),(2,'b');"
        "INSERT INTO child (id,parent_id,tag) VALUES (10,1,'x'),(11,2,'y');"
    )
    cx.commit()
    cx.close()


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_copy_all_and_verify(monkeypatch, tmp_path):
    """Contract test: copy_all copies both tables, parity is green, and
    pg_fk_order places parent before child. The source basename
    ('pgmig_test.db') is chosen to match the target schema basename so
    schema_for_path lands both the DDL setup and copy_all in the SAME
    isolated 'pgmig_test' schema -- no monkeypatch/indirection needed."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_test.db")
    _mk_sqlite_parent_child(src)

    with db.connect(src) as cx:
        cx.execute("DROP TABLE IF EXISTS child CASCADE")
        cx.execute("DROP TABLE IF EXISTS parent CASCADE")
        cx.execute("CREATE TABLE parent (id BIGINT PRIMARY KEY, name TEXT)")
        cx.execute("CREATE TABLE child (id BIGINT PRIMARY KEY, parent_id BIGINT REFERENCES parent(id), tag TEXT)")
        cx.commit()

    res = copy.copy_all(src)
    by_table = {r["table"]: r for r in res}
    assert {"parent", "child"} <= set(by_table)
    assert by_table["parent"]["source_rows"] == 2
    assert by_table["parent"]["inserted"] == 2
    assert by_table["parent"]["conflicts"] == 0
    assert by_table["child"]["source_rows"] == 2
    assert by_table["child"]["inserted"] == 2

    ver = verify.parity(src)
    assert verify.all_ok(ver)
    ver_by_table = {r["table"]: r for r in ver}
    assert ver_by_table["parent"]["sqlite"] == ver_by_table["parent"]["postgres"] == 2
    assert ver_by_table["child"]["sqlite"] == ver_by_table["child"]["postgres"] == 2

    order = introspect.pg_fk_order(db.connect(src), "pgmig_test")
    assert order.index("parent") < order.index("child")

    # Re-running copy_all is idempotent: everything is now a conflict, no new rows.
    res2 = copy.copy_all(src)
    by_table2 = {r["table"]: r for r in res2}
    assert by_table2["parent"]["inserted"] == 0
    assert by_table2["parent"]["conflicts"] == 2
    assert verify.all_ok(verify.parity(src))


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_copy_table_reports_conflicts_from_pre_existing_row(monkeypatch, tmp_path):
    """A row that already exists on the PG side (same PK) is skipped via
    ON CONFLICT DO NOTHING and counted in `conflicts`, not `inserted` --
    while parity still passes because both sides end up with matching totals."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_conflict.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE parent (id INTEGER PRIMARY KEY, name TEXT);")
    cx.executescript("INSERT INTO parent (id,name) VALUES (1,'a'),(2,'b');")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS parent CASCADE")
        pcx.execute("CREATE TABLE parent (id BIGINT PRIMARY KEY, name TEXT)")
        pcx.execute("INSERT INTO parent (id, name) VALUES (1, 'pre-existing')")
        pcx.commit()

    res = copy.copy_all(src)
    by_table = {r["table"]: r for r in res}
    assert by_table["parent"]["source_rows"] == 2
    assert by_table["parent"]["inserted"] == 1
    assert by_table["parent"]["conflicts"] == 1

    ver = verify.parity(src)
    assert verify.all_ok(ver)


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_fk_order_no_fk_table_sorts_first_and_cycle_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cycle.db")
    # sqlite side is irrelevant to this test; only pg_fk_order is exercised.
    sqlite3.connect(src).close()

    with db.connect(src) as cx:
        cx.execute("DROP TABLE IF EXISTS b CASCADE")
        cx.execute("DROP TABLE IF EXISTS a CASCADE")
        cx.execute("DROP TABLE IF EXISTS lonely CASCADE")
        cx.execute("CREATE TABLE lonely (id BIGINT PRIMARY KEY)")
        cx.execute("CREATE TABLE b (id BIGINT PRIMARY KEY, a_id BIGINT)")
        cx.execute("CREATE TABLE a (id BIGINT PRIMARY KEY, b_id BIGINT)")
        # one-way FK only (a -> b): no cycle yet.
        cx.execute('ALTER TABLE a ADD CONSTRAINT a_b_fk FOREIGN KEY (b_id) REFERENCES b(id)')
        cx.commit()

    schema = "pgmig_cycle"
    order = introspect.pg_fk_order(db.connect(src), schema)
    assert "lonely" in order  # a table with no FK at all is still included
    assert order.index("b") < order.index("a")  # parent (b) before child (a)

    with db.connect(src) as cx:
        # add the reverse FK (b -> a) to close the cycle a <-> b.
        cx.execute('ALTER TABLE b ADD CONSTRAINT b_a_fk FOREIGN KEY (a_id) REFERENCES a(id)')
        cx.commit()

    with pytest.raises(RuntimeError, match="FK cycle"):
        introspect.pg_fk_order(db.connect(src), schema)
