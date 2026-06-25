import sqlite3
from dashboard.biofield_authoring import (
    add_chain_row, create_test, init_auth_tables, ordered_chain, reorder_chain)


def test_insert_at_n_renumbers_top_zone(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    a = add_chain_row(cx, tid, 1, "A", "", "R1")
    b = add_chain_row(cx, tid, 2, "B", "", "R2")
    c = add_chain_row(cx, tid, 3, "C", "", "R3")
    reorder_chain(cx, tid, c, 1)                 # move C to the top
    assert [r["head"] for r in ordered_chain(cx, tid)] == ["C", "A", "B"]
    assert [r["layer"] for r in ordered_chain(cx, tid)] == [1, 2, 3]


def test_reorder_leaves_unbalanced_scan_trailing(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_auth_tables(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-25")
    a = add_chain_row(cx, tid, 1, "A", "", "R1")
    b = add_chain_row(cx, tid, 2, "B", "", "R2")
    s = add_chain_row(cx, tid, 1, "Scan", "", "R3", confirmed=0, origin="scan")
    reorder_chain(cx, tid, b, 1)                 # B to top among the live rows
    rows = ordered_chain(cx, tid)
    assert [r["head"] for r in rows] == ["B", "A", "Scan"]
    assert rows[-1]["zone"] == "bottom"
