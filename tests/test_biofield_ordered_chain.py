import sqlite3
from dashboard.biofield_authoring import (
    add_chain_row, authored_report, confirm_row, create_test, init_auth_tables, ordered_chain)


def _seed(cx):
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    # two unbalanced scan layers + one live layer added between them
    add_chain_row(cx, tid, 1, "ScanA", "a", "R1", confirmed=0, origin="scan")
    add_chain_row(cx, tid, 2, "ScanB", "b", "R2", confirmed=0, origin="scan")
    live = add_chain_row(cx, tid, 1, "Live", "c", "R3", confirmed=1, origin="live")
    return tid, live


def test_live_on_top_unbalanced_scan_trails_renumbered(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    tid, _ = _seed(cx)
    rows = ordered_chain(cx, tid)
    assert [r["head"] for r in rows] == ["Live", "ScanA", "ScanB"]
    assert [r["layer"] for r in rows] == [1, 2, 3]          # contiguous display
    assert [r["zone"] for r in rows] == ["top", "bottom", "bottom"]


def test_confirming_scan_layer_promotes_it(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    tid, _ = _seed(cx)
    sa = cx.execute("SELECT id FROM biofield_auth_chain WHERE head='ScanA'").fetchone()[0]
    confirm_row(cx, sa)
    rows = ordered_chain(cx, tid)
    # ScanA now top-zone (confirmed); ordered by stored layer (Live layer=1, ScanA layer=1 -> tie broken by id: Live first)
    assert [r["zone"] for r in rows] == ["top", "top", "bottom"]
    assert "ScanB" == rows[-1]["head"]


def test_authored_report_uses_display_numbering(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    tid, _ = _seed(cx)
    rep = authored_report(cx, tid)
    assert [l["layer"] for l in rep["layers"]] == [1, 2, 3]
    assert rep["layers"][0]["head"] == "Live"
