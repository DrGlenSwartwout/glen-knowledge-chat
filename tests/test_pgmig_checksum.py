"""Content-checksum parity (Task C1, P06 pre-cutover integrity gate).

Row-count parity (verify.parity) cannot detect content corruption that
preserves cardinality (truncation, type coercion, encoding). These tests
prove the checksum catches exactly that: an identical copy -> ok=True; a
same-row-count but content-corrupted copy -> ok=False (while row-count
parity for the same table would still say ok=True); a column-set mismatch
-> ok=False with a clear reason, no crash.

Skip-guarded on PG_DSN, mirroring test_pgmig_copy_verify.py.
"""
import math
import os
import sqlite3
from datetime import datetime, timezone

import pytest

from scripts import migrate_sqlite_to_pg as cli
from scripts.pgmig import verify
from dashboard import db

pg = bool(os.environ.get("PG_DSN"))


def _mk_sqlite(path, rows):
    cx = sqlite3.connect(path)
    cx.executescript(
        "CREATE TABLE widget (id INTEGER PRIMARY KEY, name TEXT, qty INTEGER, active INTEGER);"
    )
    cx.executemany("INSERT INTO widget (id,name,qty,active) VALUES (?,?,?,?)", rows)
    cx.commit()
    cx.close()


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_checksum_table_identical_copy_is_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cksum_ok.db")
    rows = [(1, "alice", 10, 1), (2, "bob", 20, 0)]
    _mk_sqlite(src, rows)

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT, qty BIGINT, active BOOLEAN)")
        for r in rows:
            pcx.execute("INSERT INTO widget (id,name,qty,active) VALUES (?,?,?,?)",
                        (r[0], r[1], r[2], bool(r[3])))
        pcx.commit()

    sqlite_cx = sqlite3.connect(src)
    pg_cx = db.connect(src)
    try:
        result = verify.checksum_table(sqlite_cx, pg_cx, "widget")
    finally:
        sqlite_cx.close()
        pg_cx.close()

    assert result["table"] == "widget"
    assert result["ok"] is True
    assert result["sqlite_digest"] == result["pg_digest"]
    assert result["sqlite_digest"] is not None


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_checksum_table_detects_corruption_that_row_count_misses(monkeypatch, tmp_path):
    """Same row COUNT on both sides (row-count parity passes), but one value
    is corrupted (a truncated name) on the PG side -- checksum must catch
    what row-count parity structurally cannot."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cksum_corrupt.db")
    rows = [(1, "alice", 10, 1), (2, "bob", 20, 0)]
    _mk_sqlite(src, rows)

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT, qty BIGINT, active BOOLEAN)")
        # id=1's name truncated ("alice" -> "ali") -- same row count, corrupted content.
        pcx.execute("INSERT INTO widget (id,name,qty,active) VALUES (1,'ali',10,true)")
        pcx.execute("INSERT INTO widget (id,name,qty,active) VALUES (2,'bob',20,false)")
        pcx.commit()

    sqlite_cx = sqlite3.connect(src)
    pg_cx = db.connect(src)
    try:
        # Row-count parity: same count both sides -> would report ok.
        s_count = sqlite_cx.execute("SELECT COUNT(*) FROM widget").fetchone()[0]
        p_count = pg_cx.execute("SELECT COUNT(*) FROM widget").fetchone()[0]
        assert s_count == p_count == 2

        result = verify.checksum_table(sqlite_cx, pg_cx, "widget")
    finally:
        sqlite_cx.close()
        pg_cx.close()

    assert result["ok"] is False
    assert result["sqlite_digest"] != result["pg_digest"]


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_checksum_table_column_set_differs_reports_ok_false_with_reason(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cksum_colmismatch.db")
    rows = [(1, "alice", 10, 1)]
    _mk_sqlite(src, rows)

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        # Missing 'active' column entirely on the PG side.
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT, qty BIGINT)")
        pcx.execute("INSERT INTO widget (id,name,qty) VALUES (1,'alice',10)")
        pcx.commit()

    sqlite_cx = sqlite3.connect(src)
    pg_cx = db.connect(src)
    try:
        result = verify.checksum_table(sqlite_cx, pg_cx, "widget")
    finally:
        sqlite_cx.close()
        pg_cx.close()

    assert result["ok"] is False
    assert result.get("reason")
    assert "column" in result["reason"].lower()


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_checksum_table_order_independent(monkeypatch, tmp_path):
    """Rows returned in a different order on each backend must still match --
    the digest must not depend on row order."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cksum_order.db")
    rows = [(1, "alice", 10, 1), (2, "bob", 20, 0), (3, "cleo", 30, 1)]
    _mk_sqlite(src, rows)

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT, qty BIGINT, active BOOLEAN)")
        # Insert in a DIFFERENT order than the sqlite source.
        for r in [rows[2], rows[0], rows[1]]:
            pcx.execute("INSERT INTO widget (id,name,qty,active) VALUES (?,?,?,?)",
                        (r[0], r[1], r[2], bool(r[3])))
        pcx.commit()

    sqlite_cx = sqlite3.connect(src)
    pg_cx = db.connect(src)
    try:
        result = verify.checksum_table(sqlite_cx, pg_cx, "widget")
    finally:
        sqlite_cx.close()
        pg_cx.close()

    assert result["ok"] is True


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_checksum_table_null_vs_value_not_confused(monkeypatch, tmp_path):
    """A NULL on one side and a real (falsy-looking) value on the other must
    NOT collide into the same normalized rendering."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cksum_null.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE widget (id INTEGER PRIMARY KEY, name TEXT);")
    cx.executescript("INSERT INTO widget (id,name) VALUES (1,NULL);")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT)")
        # PG side has the STRING "None" instead of a real NULL -- a corruption
        # a naive str(value) normalization would fail to distinguish.
        pcx.execute("INSERT INTO widget (id,name) VALUES (1,'None')")
        pcx.commit()

    sqlite_cx = sqlite3.connect(src)
    pg_cx = db.connect(src)
    try:
        result = verify.checksum_table(sqlite_cx, pg_cx, "widget")
    finally:
        sqlite_cx.close()
        pg_cx.close()

    assert result["ok"] is False


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_checksum_table_timestamp_faithful_copy_is_ok(monkeypatch, tmp_path):
    """Regression test for the datetime false-positive: a real Postgres
    TIMESTAMP column (readback -> Python `datetime`) vs the SQLite source
    storing the app's ISO-8601 string ('...T...Z') for the SAME instant.
    Before the fix, str(datetime) ("2026-07-10 22:14:03", space separator)
    never equalled the ISO string ("2026-07-10T22:14:03Z"), so a faithful
    copy was wrongly reported ok=False. Must now converge to ok=True."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cksum_ts_ok.db")
    cx = sqlite3.connect(src)
    cx.executescript(
        "CREATE TABLE evt (id INTEGER PRIMARY KEY, happened_at TEXT);"
    )
    cx.execute("INSERT INTO evt (id, happened_at) VALUES (1, '2026-07-10T22:14:03Z')")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS evt CASCADE")
        pcx.execute("CREATE TABLE evt (id BIGINT PRIMARY KEY, happened_at TIMESTAMP)")
        # A naive datetime representing the SAME UTC instant as the SQLite
        # ISO string above -- this is how a real "TIMESTAMP WITHOUT TIME
        # ZONE" column ends up holding a UTC instant (psycopg would convert
        # an aware value to the session timezone and drop tzinfo, which is
        # not what either side intends here).
        pcx.execute("INSERT INTO evt (id, happened_at) VALUES (1, ?)",
                     (datetime(2026, 7, 10, 22, 14, 3),))
        pcx.commit()

    sqlite_cx = sqlite3.connect(src)
    pg_cx = db.connect(src)
    try:
        result = verify.checksum_table(sqlite_cx, pg_cx, "evt")
    finally:
        sqlite_cx.close()
        pg_cx.close()

    assert result["ok"] is True
    assert result["sqlite_digest"] == result["pg_digest"]


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_checksum_table_timestamp_different_instants_still_mismatch(monkeypatch, tmp_path):
    """Sanity companion to the faithful-copy test above: the datetime
    canonicalization must not OVER-normalize -- two genuinely different
    instants must still produce different digests (ok=False)."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cksum_ts_bad.db")
    cx = sqlite3.connect(src)
    cx.executescript(
        "CREATE TABLE evt (id INTEGER PRIMARY KEY, happened_at TEXT);"
    )
    cx.execute("INSERT INTO evt (id, happened_at) VALUES (1, '2026-07-10T22:14:03Z')")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS evt CASCADE")
        pcx.execute("CREATE TABLE evt (id BIGINT PRIMARY KEY, happened_at TIMESTAMP)")
        # A DIFFERENT instant (one second later) -- must NOT match.
        pcx.execute("INSERT INTO evt (id, happened_at) VALUES (1, ?)",
                     (datetime(2026, 7, 10, 22, 14, 4),))
        pcx.commit()

    sqlite_cx = sqlite3.connect(src)
    pg_cx = db.connect(src)
    try:
        result = verify.checksum_table(sqlite_cx, pg_cx, "evt")
    finally:
        sqlite_cx.close()
        pg_cx.close()

    assert result["ok"] is False
    assert result["sqlite_digest"] != result["pg_digest"]


def test_normalize_value_datetime_and_iso_string_converge():
    """Pure-Python unit test (no PG needed): `_normalize_value` must map a
    tz-aware `datetime` and its equivalent ISO-8601 'Z' string to the
    IDENTICAL canonical output."""
    dt = datetime(2026, 7, 10, 22, 14, 3, tzinfo=timezone.utc)
    iso = "2026-07-10T22:14:03Z"
    assert verify._normalize_value(dt) == verify._normalize_value(iso)
    # And it should also converge with the "+00:00" spelling of the same instant.
    assert verify._normalize_value(dt) == verify._normalize_value("2026-07-10T22:14:03+00:00")


def test_normalize_value_non_date_string_unchanged():
    """A string that does not parse as an ISO-8601 datetime must be
    returned UNCHANGED (no accidental transformation of ordinary text)."""
    assert verify._normalize_value("just some text") == "just some text"
    assert verify._normalize_value("SKU-12345") == "SKU-12345"
    # Date-only (no time component) is out of scope for this fix -- left as-is.
    assert verify._normalize_value("2026-07-10") == "2026-07-10"


def test_normalize_value_nan_inf_do_not_raise():
    """A non-finite float must normalize to a fixed sentinel instead of
    crashing (the old `v == int(v)` branch raised on NaN/Inf)."""
    nan_out = verify._normalize_value(float("nan"))
    pos_inf_out = verify._normalize_value(float("inf"))
    neg_inf_out = verify._normalize_value(float("-inf"))

    assert nan_out != pos_inf_out != neg_inf_out
    assert nan_out != neg_inf_out
    # All distinct from ordinary numeric/text renderings.
    assert nan_out not in ("nan", "inf", "-inf")
    # Deterministic and comparable (no exception raised above already proves
    # this, but assert idempotence too).
    assert verify._normalize_value(float("nan")) == nan_out


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_checksum_parity_folds_into_all_ok_style_check(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cksum_parity.db")
    rows = [(1, "alice", 10, 1), (2, "bob", 20, 0)]
    _mk_sqlite(src, rows)

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT, qty BIGINT, active BOOLEAN)")
        for r in rows:
            pcx.execute("INSERT INTO widget (id,name,qty,active) VALUES (?,?,?,?)",
                        (r[0], r[1], r[2], bool(r[3])))
        pcx.commit()

    results = verify.checksum_parity(src)
    by_table = {r["table"]: r for r in results}
    assert "widget" in by_table
    assert by_table["widget"]["ok"] is True
    assert verify.checksum_all_ok(results) is True

    # Now corrupt it and re-run: the fold-in must go False.
    with db.connect(src) as pcx:
        pcx.execute("UPDATE widget SET name = 'ZZZ' WHERE id = 1")
        pcx.commit()

    results2 = verify.checksum_parity(src)
    assert verify.checksum_all_ok(results2) is False


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_cli_verify_checksum_ok_on_identical_copy(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_cksum_ok.db")
    rows = [(1, "alice", 10, 1), (2, "bob", 20, 0)]
    _mk_sqlite(src, rows)

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT, qty BIGINT, active BOOLEAN)")
        for r in rows:
            pcx.execute("INSERT INTO widget (id,name,qty,active) VALUES (?,?,?,?)",
                        (r[0], r[1], r[2], bool(r[3])))
        pcx.commit()

    code = cli.main(["verify", src, "--checksum"])
    out = capsys.readouterr().out
    assert code == 0
    assert "CONTENT CHECKSUM PARITY OK" in out


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_cli_verify_checksum_exit4_on_mismatch(monkeypatch, tmp_path, capsys):
    """`verify --checksum` must exit 4 (same code as a row-count mismatch)
    when the content digest differs, even though row-count parity alone
    would pass."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_cksum_bad.db")
    rows = [(1, "alice", 10, 1), (2, "bob", 20, 0)]
    _mk_sqlite(src, rows)

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT, qty BIGINT, active BOOLEAN)")
        pcx.execute("INSERT INTO widget (id,name,qty,active) VALUES (1,'ali',10,true)")  # truncated
        pcx.execute("INSERT INTO widget (id,name,qty,active) VALUES (2,'bob',20,false)")
        pcx.commit()

    # Row-count verify alone (no --checksum) would pass -- prove that first.
    code_plain = cli.main(["verify", src])
    assert code_plain == 0

    code = cli.main(["verify", src, "--checksum"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_VERIFY_MISMATCH
    assert code == 4
    assert "CONTENT CHECKSUM PARITY FAILED" in out


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_cli_full_checksum_exit4_on_mismatch(monkeypatch, tmp_path, capsys):
    """`full --checksum` must FAIL (exit 4) on a digest mismatch even though
    preflight/copy/row-count-verify all otherwise succeed."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_full_cksum_bad.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE widget (id INTEGER PRIMARY KEY, name TEXT);")
    cx.executescript("INSERT INTO widget (id,name) VALUES (1,'alice'),(2,'bob');")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT)")
        pcx.commit()

    # `full --checksum` copies fresh (clean) -- should be OK first.
    code_ok = cli.main(["full", src, "--truncate", "--checksum"])
    out_ok = capsys.readouterr().out
    assert code_ok == 0
    assert "CONTENT CHECKSUM PARITY OK" in out_ok
    assert "FULL MIGRATION OK." in out_ok

    # Now corrupt the PG side post-copy and re-verify via `verify --checksum`
    # (re-running `full` with --truncate would just re-copy cleanly again,
    # so the corruption must be introduced AFTER copy, then checked directly).
    with db.connect(src) as pcx:
        pcx.execute("UPDATE widget SET name = 'ZZZ' WHERE id = 1")
        pcx.commit()

    code_bad = cli.main(["verify", src, "--checksum"])
    out_bad = capsys.readouterr().out
    assert code_bad == cli.EXIT_VERIFY_MISMATCH
    assert "CONTENT CHECKSUM PARITY FAILED" in out_bad


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_cli_verify_without_checksum_flag_unchanged(monkeypatch, tmp_path, capsys):
    """Preserve existing behavior: `verify` with NO --checksum never prints
    the content-checksum section and never runs it."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    src = str(tmp_path / "pgmig_cli_no_cksum.db")
    cx = sqlite3.connect(src)
    cx.executescript("CREATE TABLE widget (id INTEGER PRIMARY KEY, name TEXT);")
    cx.executescript("INSERT INTO widget (id,name) VALUES (1,'a'),(2,'b');")
    cx.commit()
    cx.close()

    with db.connect(src) as pcx:
        pcx.execute("DROP TABLE IF EXISTS widget CASCADE")
        pcx.execute("CREATE TABLE widget (id BIGINT PRIMARY KEY, name TEXT)")
        pcx.commit()

    code = cli.main(["full", src, "--truncate"])
    out = capsys.readouterr().out
    assert code == 0
    assert "CONTENT CHECKSUM" not in out
