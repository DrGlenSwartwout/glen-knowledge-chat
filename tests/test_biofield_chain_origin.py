import sqlite3
from dashboard.biofield_authoring import add_chain_row, init_auth_tables, create_test
from dashboard.biofield_reveal_import import import_layers_to_test


def _col_names(cx, table):
    return {r[1] for r in cx.execute(f"PRAGMA table_info({table})").fetchall()}


def test_chain_has_origin_column_default_live(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    assert "origin" in _col_names(cx, "biofield_auth_chain")
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    rid = add_chain_row(cx, tid, 1, "Head", "Most", "Remedy")
    row = cx.execute("SELECT origin FROM biofield_auth_chain WHERE id=?", (rid,)).fetchone()
    assert row[0] == "live"


def test_add_chain_row_accepts_origin(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    rid = add_chain_row(cx, tid, 1, "H", "M", "R", origin="scan")
    assert cx.execute("SELECT origin FROM biofield_auth_chain WHERE id=?", (rid,)).fetchone()[0] == "scan"


def test_import_marks_rows_scan(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    import_layers_to_test(cx, tid, [{"n": 1, "title": "Ox", "most_affected": "A", "remedy_name": "Neuro Magnesium"}])
    origins = [r[0] for r in cx.execute("SELECT origin FROM biofield_auth_chain WHERE test_id=?", (int(str(tid).lstrip('a')),)).fetchall()]
    assert origins == ["scan"]
