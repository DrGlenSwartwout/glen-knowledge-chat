"""Operator CLI: preflight / copy / verify / full (Task 3, P05).

SQLite-only tests (preflight, env-guard) run unconditionally. Anything that
touches Postgres (copy/verify/full, and the PG-target unique-index
cross-check) is skip-guarded on PG_DSN, mirroring test_pgmig_copy_verify.py.
"""
import os
import sqlite3

import pytest

from scripts import migrate_sqlite_to_pg as cli
from scripts.pgmig import dedup
from dashboard import db

pg = bool(os.environ.get("PG_DSN"))


def _mk_dirty_source(path, table="accounts", col="code"):
    """A table with a GENUINE non-NULL duplicate-key group under a live
    unique index. (Pre-P05.5-Task-4 this fixture used two NULL keys --
    that was exactly the false-positive shape Task 4 correctly stopped
    reporting, since NULLs are distinct under UNIQUE; using NULLs here would
    now just prove exit 0, not exit 3.)

    A LIVE unique index still can't have a real duplicate INSERTed directly
    (SQLite enforces it at write time) -- so this scopes the index to a
    PARTIAL predicate (`WHERE active = 1`): SQLite's PRAGMA index_list still
    reports it as unique (verified), and `scan_collisions`/`scan_db` don't
    account for the partial predicate (out of Task 4's/this task's scope --
    a pre-existing property, not a defect introduced here), so two INACTIVE
    rows sharing the same code insert cleanly (the partial index never sees
    them) while still surfacing as a genuine, non-NULL duplicate-key finding."""
    cx = sqlite3.connect(path)
    cx.executescript(
        f'CREATE TABLE {table} (id INTEGER PRIMARY KEY, {col} TEXT, active INTEGER);'
        f'CREATE UNIQUE INDEX ux_{table}_{col} ON {table} ({col}) WHERE active = 1;'
        f"INSERT INTO {table} ({col}, active) VALUES ('dup', 0), ('dup', 0), ('keeper', 1);"
    )
    cx.commit()
    cx.close()


def _mk_clean_source(path, table="accounts", col="code"):
    cx = sqlite3.connect(path)
    cx.executescript(
        f'CREATE TABLE {table} (id INTEGER PRIMARY KEY, {col} TEXT);'
        f'CREATE UNIQUE INDEX ux_{table}_{col} ON {table} ({col});'
        f"INSERT INTO {table} ({col}) VALUES ('a'), ('b');"
    )
    cx.commit()
    cx.close()


def test_preflight_dirty_source_exits_3_and_names_table(tmp_path, capsys):
    src = str(tmp_path / "pgmig_cli_dirty.db")
    _mk_dirty_source(src)
    code = cli.main(["preflight", src])
    out = capsys.readouterr().out
    assert code == 3
    assert "accounts" in out


def test_preflight_clean_source_exits_0(tmp_path, capsys):
    src = str(tmp_path / "pgmig_cli_clean.db")
    _mk_clean_source(src)
    code = cli.main(["preflight", src])
    out = capsys.readouterr().out
    assert code == 0
    assert "clean" in out.lower() or "0" in out


@pytest.mark.parametrize("command", ["copy", "verify", "full"])
def test_pg_commands_require_pg_env(monkeypatch, tmp_path, capsys, command):
    """copy/verify/full must refuse to run (clear error, nonzero exit) when
    DB_BACKEND/PG_DSN aren't configured -- preflight alone is sqlite-only and
    doesn't require this."""
    monkeypatch.delenv("DB_BACKEND", raising=False)
    monkeypatch.delenv("PG_DSN", raising=False)
    src = str(tmp_path / "pgmig_cli_env.db")
    sqlite3.connect(src).close()
    code = cli.main([command, src])
    out = capsys.readouterr().out
    assert code != 0
    assert "DB_BACKEND" in out or "PG_DSN" in out


def test_preflight_incomplete_when_pg_target_crosscheck_raises(monkeypatch, tmp_path, capsys):
    """I1: PG IS configured (DB_BACKEND=postgres + PG_DSN) but the PG-target
    unique-index introspection itself fails -- preflight must NOT silently
    degrade to a sqlite-only scan and report PREFLIGHT CLEAN/exit 0 (that is
    false assurance on the exact cross-check the P06 cutover gate relies
    on). It must report PREFLIGHT INCOMPLETE and exit a distinct nonzero
    code (5), different from both a clean run (0) and a dirty one (3).

    Needs NO live Postgres: DB_BACKEND/PG_DSN are set only to satisfy
    `_pg_configured()`; `db_mod.connect` is stubbed so no real connection is
    attempted, and `introspect.pg_unique_indexes` is monkeypatched to raise,
    simulating the introspection query itself failing against an
    otherwise-configured target. Runs unconditionally (no pg skip-guard).
    """
    monkeypatch.setenv("DB_BACKEND", "postgres")
    monkeypatch.setenv("PG_DSN", "postgresql://fake-unused-in-this-test/db")

    class _FakeCx:
        def close(self):
            pass

    monkeypatch.setattr(cli.db_mod, "connect", lambda *_a, **_k: _FakeCx())

    def _raise(*_a, **_k):
        raise RuntimeError("simulated introspection failure")

    monkeypatch.setattr(cli.introspect, "pg_unique_indexes", _raise)

    src = str(tmp_path / "pgmig_cli_incomplete.db")
    _mk_clean_source(src)  # a legitimately clean sqlite source

    code = cli.main(["preflight", src])
    out = capsys.readouterr().out
    assert code == cli.EXIT_PREFLIGHT_INCOMPLETE
    assert code != 0
    assert "INCOMPLETE" in out
    assert "simulated introspection failure" in out
    assert "PREFLIGHT CLEAN" not in out


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_full_command_synthetic_source_exit_0(monkeypatch, tmp_path, capsys):
    """The CLI smoke test: `full --no-schema` against an isolated synthetic
    source (its own PG schema via a distinct basename), with the target
    table pre-created by hand, exits 0 and prints the row-count parity-ok
    message. Uses --no-schema because this test's whole point is exercising
    copy/verify against an operator-pre-created schema, not the P05.5
    create-schema-from-DDL path (that path has its own dedicated end-to-end
    test below)."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_full.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE widget (id INTEGER PRIMARY KEY, name TEXT);")
    cx.executescript("INSERT INTO widget (id,name) VALUES (1,'a'),(2,'b');")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT)")
        pcx.commit()

    code = cli.main(["full", "--no-schema", src])
    out = capsys.readouterr().out
    assert code == 0
    assert "--no-schema set: skipping schema creation" in out
    assert "ROW-COUNT PARITY OK (counts only -- content not compared)" in out


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_preflight_detects_pg_target_only_unique_index(monkeypatch, tmp_path, capsys):
    """CORRECTNESS-CRITICAL cross-check: a source table with NO sqlite unique
    index but a matching unique index already in the fresh Postgres target
    schema, containing duplicate values -- preflight must still catch it
    (exit 3), even though scan_db (sqlite-source-only) can't see it."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_pgtarget.db")
    cx = sqlite3.connect(src)
    cx.executescript(
        "CREATE TABLE sessions (id INTEGER PRIMARY KEY, token_key TEXT);"
        "INSERT INTO sessions (token_key) VALUES ('dup'), ('dup'), ('unique1');"
    )
    cx.commit()
    cx.close()

    # Negative control: proves the sqlite-source-only scan really can't see
    # this collision (no unique index on token_key in the sqlite DDL above).
    assert not any(f["table"] == "sessions" for f in dedup.scan_db(src))

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS sessions CASCADE")
        pcx.execute("CREATE TABLE sessions (id BIGINT PRIMARY KEY, token_key TEXT)")
        pcx.execute("CREATE UNIQUE INDEX ux_sessions_token_key ON sessions (token_key)")
        pcx.commit()

    code = cli.main(["preflight", src])
    out = capsys.readouterr().out
    assert code == 3
    assert "sessions" in out


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_full_nonzero_when_copy_has_errors(monkeypatch, tmp_path, capsys):
    """A row that fails to persist (copy.any_errors True) must fail the
    `full` run even though preflight was clean and parity would otherwise
    be checked -- a failed row must fail the run. Uses --no-schema: the
    NOT NULL constraint that forces the row error here is deliberately
    hand-added to the PG target (the sqlite source itself has no such
    constraint), so the P05.5 schema-from-DDL step -- which would rebuild
    this table WITHOUT that constraint -- must be skipped for this test to
    still exercise copy's row-error path."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_copyerr.db")
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

    code = cli.main(["full", "--no-schema", src])
    out = capsys.readouterr().out
    assert code != 0
    assert "error" in out.lower()


# --- P05.5 Task 5: create-schema command + full building the schema from
# source DDL first (self-contained -- no `import app`). ------------------

def _mk_ddl_synthetic_source(path):
    """widget: AUTOINCREMENT PK (identity-insert proof), a REAL col, and a
    loose-typed INTEGER col that actually holds a UUID string on one row
    (data-driven TEXT-widening proof) -- plus explicit, non-contiguous ids
    (5, 7) so a successful copy proves ORIGINAL ids survived (not
    renumbered 1, 2)."""
    cx = sqlite3.connect(path)
    cx.executescript(
        "CREATE TABLE widget ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  price REAL,"
        "  ext_ref INTEGER,"
        "  name TEXT"
        ");"
    )
    cx.executescript(
        "INSERT INTO widget (id, price, ext_ref, name) VALUES "
        "(5, 3.14, '123e4567-e89b-12d3-a456-426614174000', 'a'),"
        "(7, 2.5, '42', 'b');"
    )
    cx.commit()
    cx.close()


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_create_schema_standalone_exit_0(monkeypatch, tmp_path, capsys):
    """`create-schema <src>` alone (no preflight/copy/verify) builds the PG
    schema from the source DDL and exits 0, printing tables/indexes/widened
    counts."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_create_schema.db")
    _mk_ddl_synthetic_source(src)

    code = cli.main(["create-schema", "--drop", src])
    out = capsys.readouterr().out
    assert code == 0
    assert "SCHEMA CREATE OK." in out
    assert "tables_created=1" in out
    assert "indexes_created=0" in out
    assert "widget.ext_ref" in out  # the widened (loose-int -> TEXT) column, named


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_full_end_to_end_builds_schema_from_ddl_then_copies_and_verifies(
        monkeypatch, tmp_path, capsys):
    """The P05.5 end-to-end proof: `full` with NO pre-existing target schema
    at all -- it must build the schema itself from the source DDL (identity
    PK, REAL widening, loose-int-to-TEXT widening), then copy the ORIGINAL
    explicit ids, then verify row-count AND content-checksum parity, all in
    one exit-0 run. No `import app` involved anywhere."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_full_ddl.db")
    _mk_ddl_synthetic_source(src)

    code = cli.main(["full", "--checksum", src])
    out = capsys.readouterr().out
    assert code == 0, out

    # Schema was built from DDL, not pre-existing.
    assert "SCHEMA CREATE OK." in out
    assert "tables_created=1" in out
    assert "widget.ext_ref" in out  # widened_cols surfaced prominently

    # Data copied, parity + checksum both OK.
    assert "ROW-COUNT PARITY OK (counts only -- content not compared)" in out
    assert "CONTENT CHECKSUM PARITY OK" in out
    assert "FULL MIGRATION OK." in out

    # Original explicit ids (5, 7) survived the copy -- not renumbered.
    with db.connect(src) as pcx:
        ids = sorted(r[0] for r in pcx.execute("SELECT id FROM widget").fetchall())
        assert ids == [5, 7]
        # Identity sequence was reset past the highest copied id: a fresh
        # auto-gen insert (no explicit id) must not collide with 5 or 7.
        pcx.execute("INSERT INTO widget (price, ext_ref, name) VALUES (1.0, '1', 'c')")
        pcx.commit()
        new_id = pcx.execute("SELECT id FROM widget WHERE name = 'c'").fetchone()[0]
        assert new_id > 7


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_full_no_schema_skips_creation_and_fails_against_absent_schema(
        monkeypatch, tmp_path, capsys):
    """`full --no-schema` against a target with NO pre-created schema at all
    must NOT silently build one -- proving --no-schema really does skip
    creation. `dashboard.db.connect` still auto-creates the (empty) SCHEMA
    namespace itself, but with no tables in it copy has nothing to insert
    into and verify's row-count parity must catch the resulting mismatch
    (sqlite has rows, postgres has no such table) -- the run must exit
    nonzero, not report a false OK."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_full_no_schema.db")
    _mk_ddl_synthetic_source(src)

    # Belt-and-braces: make sure no prior run left a schema/table behind for
    # this distinct basename.
    from dashboard.dbschema import schema_for_path
    schema = schema_for_path(src)
    with db.connect(src) as pcx:
        pcx.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        pcx.commit()

    code = cli.main(["full", "--no-schema", src])
    out = capsys.readouterr().out
    assert code != 0
    assert "--no-schema set: skipping schema creation" in out
    assert "SCHEMA CREATE OK." not in out  # proves the schema step was truly skipped
    assert "MISMATCH" in out or "no matching postgres target table" in out


# --- Direct-invocation regression guard (the absolute `scripts.pgmig` imports
# fail with ModuleNotFoundError when the script is run as `python3
# scripts/migrate_sqlite_to_pg.py ...` without the repo root on sys.path; the
# CLI bootstraps sys.path itself. pytest's conftest hid this, so guard it via a
# real subprocess run, not an in-process import.) -----------------------------
import os as _os
import subprocess as _sp
import sys as _sys
import sqlite3 as _sqlite3

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_CLI = _os.path.join(_REPO_ROOT, "scripts", "migrate_sqlite_to_pg.py")


def test_cli_runs_as_direct_subprocess_clean(tmp_path):
    dbp = str(tmp_path / "clean.db")
    cx = _sqlite3.connect(dbp)
    cx.executescript("CREATE TABLE u (k TEXT); CREATE UNIQUE INDEX ux ON u(k);"
                     "INSERT INTO u(k) VALUES ('a'),('b');")
    cx.commit(); cx.close()
    env = dict(_os.environ); env.pop("DB_BACKEND", None); env.pop("PG_DSN", None)
    r = _sp.run([_sys.executable, _CLI, "preflight", dbp],
                capture_output=True, text=True, cwd=_REPO_ROOT, env=env)
    assert "ModuleNotFoundError" not in r.stderr, r.stderr
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "PREFLIGHT CLEAN" in r.stdout


def test_cli_runs_as_direct_subprocess_dirty_exit3(tmp_path):
    dbp = str(tmp_path / "dirty.db")
    cx = _sqlite3.connect(dbp)
    # A genuine non-NULL duplicate-key group under a live (partial) unique
    # index -- see _mk_dirty_source's docstring above for why NULL-key dupes
    # no longer trigger exit 3 after Task 4 (correctly: NULLs are distinct
    # under UNIQUE, so that was always a false positive).
    cx.executescript(
        "CREATE TABLE d (k TEXT, active INTEGER);"
        "CREATE UNIQUE INDEX ux ON d(k) WHERE active = 1;"
        "INSERT INTO d(k, active) VALUES ('dup',0),('dup',0),('a',1);")
    cx.commit(); cx.close()
    env = dict(_os.environ); env.pop("DB_BACKEND", None); env.pop("PG_DSN", None)
    r = _sp.run([_sys.executable, _CLI, "preflight", dbp],
                capture_output=True, text=True, cwd=_REPO_ROOT, env=env)
    assert "ModuleNotFoundError" not in r.stderr, r.stderr
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    assert "d" in r.stdout
