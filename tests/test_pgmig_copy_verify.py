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


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_per_row_fallback_isolates_bad_row_with_savepoint(monkeypatch, tmp_path):
    """Reproduces the review's exact scenario: a NOT NULL violation partway
    through the per-row fallback must NOT roll back the good rows inserted
    earlier in the SAME pass. Source has good rows BEFORE (1,2) and AFTER (4)
    a bad row (3, NULL into a NOT NULL column). Proves: the good rows
    actually persist in PG (queried directly), `inserted` equals the real
    persisted count, the bad row lands in `errors` (not silently dropped),
    and `verify.parity` reports the true (mismatched) state rather than a
    false green from a miscounted `inserted`.

    Before the fix this reproduced the reviewed bug exactly: `inserted`
    reported 3 but only 1 row (id=4) actually persisted, because the shared,
    savepoint-less transaction's rollback() on row 3's failure discarded
    rows 1 and 2 as well, and they were never added to `errors`.
    """
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_savepoint.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE widget (id INTEGER PRIMARY KEY, name TEXT);")
    cx.executescript(
        "INSERT INTO widget (id,name) VALUES (1,'a'),(2,'b'),(3,NULL),(4,'d');"
    )
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT NOT NULL)")
        pcx.commit()

    res = copy.copy_all(src)
    by_table = {r["table"]: r for r in res}
    widget = by_table["widget"]

    assert widget["source_rows"] == 4
    # Rows 1, 2, 4 persist; row 3 (NULL name) fails and is isolated by its
    # own savepoint -- it must NOT drag rows 1/2 down with it.
    assert widget["inserted"] == 3
    assert "errors" in widget
    assert len(widget["errors"]) == 1
    assert widget["errors"][0]["row"][0] == 3
    assert widget["conflicts"] == 0

    # The proof: query PG directly for what ACTUALLY persisted.
    with db.connect(src) as pcx:
        actual_ids = sorted(r[0] for r in pcx.execute('SELECT id FROM "widget"').fetchall())
        actual_count = pcx.execute('SELECT COUNT(*) FROM "widget"').fetchone()[0]

    assert actual_ids == [1, 2, 4]
    assert actual_count == 3 == widget["inserted"]

    # verify.parity must reflect the TRUE state -- 4 source rows vs 3 landed,
    # a genuine mismatch, not a false green from a miscounted `inserted`.
    ver = verify.parity(src)
    ver_by_table = {r["table"]: r for r in ver}
    assert ver_by_table["widget"]["sqlite"] == 4
    assert ver_by_table["widget"]["postgres"] == 3
    assert ver_by_table["widget"]["ok"] is False
    assert not verify.all_ok(ver)

    assert copy.any_errors(res) is True


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_any_errors_false_on_clean_copy(monkeypatch, tmp_path):
    """`any_errors` is the CLI's loud-failure gate: False when nothing failed."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_clean.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE widget (id INTEGER PRIMARY KEY, name TEXT);")
    cx.executescript("INSERT INTO widget (id,name) VALUES (1,'a'),(2,'b');")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT)")
        pcx.commit()

    res = copy.copy_all(src)
    assert by_table_inserted(res, "widget") == 2
    assert copy.any_errors(res) is False


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_copy_all_resets_identity_sequence_after_explicit_id_inserts(monkeypatch, tmp_path):
    """The Task 3 contract: copying an identity-PK table with explicit,
    non-contiguous ids (1, 2, 5 -- gaps are normal, e.g. deleted rows in the
    source) must (a) land every row under its ORIGINAL id, and (b) leave the
    Postgres identity sequence advanced past the highest id copied, so the
    very next auto-generated insert (no explicit id given) returns max+1 = 6,
    not a colliding 1/2/3."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_identity.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE widget (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT);")
    cx.executescript("INSERT INTO widget (id,v) VALUES (1,'a'),(2,'b'),(5,'e');")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        # The Task 2 translation target: BY DEFAULT accepts explicit ids too.
        pcx.execute(
            "CREATE TABLE widget (id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, v TEXT)"
        )
        pcx.commit()

    res = copy.copy_all(src)
    by_table = {r["table"]: r for r in res}
    widget = by_table["widget"]
    assert widget["source_rows"] == 3
    assert widget["inserted"] == 3
    assert "errors" not in widget

    # Parity must hold BEFORE the proof-insert below adds a 4th row on
    # purpose (that intentional divergence is not a migration defect).
    assert verify.all_ok(verify.parity(src))

    with db.connect(src) as pcx:
        ids = sorted(r[0] for r in pcx.execute('SELECT id FROM "widget"').fetchall())
        assert ids == [1, 2, 5]

        # Proof: a fresh auto-generated insert (no id given) must land at
        # max(id)+1 = 6, not collide with any migrated id.
        new_id = pcx.execute(
            'INSERT INTO "widget" (v) VALUES (\'new\') RETURNING id'
        ).fetchone()[0]
        assert new_id == 6


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_copy_all_empty_identity_table_next_autogen_insert_is_one(monkeypatch, tmp_path):
    """An empty identity table has max(id) IS NULL -- the sequence reset
    must fall back to 1 (COALESCE(...,0)+1), not crash, so the first
    auto-generated insert after copying an empty table lands at id=1."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_identity_empty.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE widget (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT);")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute(
            "CREATE TABLE widget (id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, v TEXT)"
        )
        pcx.commit()

    res = copy.copy_all(src)
    by_table = {r["table"]: r for r in res}
    assert by_table["widget"]["source_rows"] == 0

    with db.connect(src) as pcx:
        new_id = pcx.execute(
            'INSERT INTO "widget" (v) VALUES (\'first\') RETURNING id'
        ).fetchone()[0]
        assert new_id == 1


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_copy_all_no_identity_column_table_does_not_error(monkeypatch, tmp_path):
    """A table with no identity column at all (plain TEXT primary key) must
    not error when copy_all's post-copy sequence reset runs against it --
    `reset_identity_sequences` should be a no-op (the information_schema
    is_identity='YES' lookup returns zero columns)."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_no_identity.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE token (id TEXT PRIMARY KEY, v TEXT);")
    cx.executescript("INSERT INTO token (id,v) VALUES ('abc','x'),('def','y');")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS token CASCADE")
        pcx.execute("CREATE TABLE token (id TEXT PRIMARY KEY, v TEXT)")
        pcx.commit()

    res = copy.copy_all(src)
    by_table = {r["table"]: r for r in res}
    assert by_table["token"]["source_rows"] == 2
    assert by_table["token"]["inserted"] == 2
    assert "errors" not in by_table["token"]
    assert verify.all_ok(verify.parity(src))


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_reset_identity_sequences_directly(monkeypatch, tmp_path):
    """Unit-level check of `reset_identity_sequences` itself (not via
    copy_all): call it directly against a table already holding rows with
    a gap, and confirm the sequence lands at max+1."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_identity_direct.db")
    sqlite3.connect(src).close()  # sqlite side unused by this test
    schema = "pgmig_identity_direct"

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute(
            "CREATE TABLE widget (id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, v TEXT)"
        )
        pcx.execute('INSERT INTO "widget" (id, v) VALUES (3, \'c\'), (7, \'g\')')
        pcx.commit()

    with db.connect(src) as pcx:
        copy.reset_identity_sequences(pcx, schema, "widget")
        pcx.commit()

    with db.connect(src) as pcx:
        new_id = pcx.execute(
            'INSERT INTO "widget" (v) VALUES (\'new\') RETURNING id'
        ).fetchone()[0]
        assert new_id == 8


def by_table_inserted(results, table):
    return next(r["inserted"] for r in results if r["table"] == table)


def test_copy_all_asserts_postgres_backend(monkeypatch, tmp_path):
    """M4: copy_all must refuse to run against a non-Postgres backend rather
    than silently issuing information_schema queries against a SQLite handle.

    Unguarded (no PG skip-mark): the `db.backend() != "postgres"` assert this
    tests fires BEFORE any PG connection is attempted, so it must run (and
    pass) in the SQLite-only harness too."""
    monkeypatch.delenv("DB_BACKEND", raising=False)
    src = str(tmp_path / "pgmig_backend_guard.db")
    sqlite3.connect(src).close()

    with pytest.raises(RuntimeError, match="postgres"):
        copy.copy_all(src)
