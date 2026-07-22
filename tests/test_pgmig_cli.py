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
    """A table WITH a unique index in its DDL that nonetheless has a genuine
    duplicate-key group -- two NULLs. SQLite's UNIQUE index does not treat
    NULL as equal to itself for constraint-enforcement purposes, so this is
    a legitimate way to get scan_db to find a real collision without needing
    to bypass SQLite's own index (same trick as test_scan_db_direct_file_path
    in test_pgmig_dedup.py)."""
    cx = sqlite3.connect(path)
    cx.executescript(
        f'CREATE TABLE {table} (id INTEGER PRIMARY KEY, {col} TEXT);'
        f'CREATE UNIQUE INDEX ux_{table}_{col} ON {table} ({col});'
        f"INSERT INTO {table} ({col}) VALUES (NULL), (NULL), ('keeper');"
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
    """The CLI smoke test: `full` against an isolated synthetic source (its
    own PG schema via a distinct basename) exits 0 and prints the row-count
    parity-ok message."""
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

    code = cli.main(["full", src])
    out = capsys.readouterr().out
    assert code == 0
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
    be checked -- a failed row must fail the run."""
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

    code = cli.main(["full", src])
    out = capsys.readouterr().out
    assert code != 0
    assert "error" in out.lower()
