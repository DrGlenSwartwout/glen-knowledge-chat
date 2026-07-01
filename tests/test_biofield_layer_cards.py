"""Layer-card grouping + drag-to-reorder for the causal-chain editor."""
import sqlite3

import pytest

from dashboard.biofield_report_html import group_layers
from dashboard.biofield_authoring import set_layer_order, add_chain_row, ordered_chain
from biofield_local_app import create_app


def test_group_layers_merges_shared_head():
    layers = [
        {"rid": 5, "layer": 1, "head": "Lymph", "most_affected": "Groin", "remedy": "Lymph Flow"},
        {"rid": 6, "layer": 2, "head": "Neural", "most_affected": "CNS", "remedy": "Nerve Pulse"},
        {"rid": 7, "layer": 3, "head": "Neural", "most_affected": "CNS", "remedy": "Mag Glycinate"},
    ]
    g = group_layers(layers)
    assert [x["layer"] for x in g] == [1, 2]                 # two cards, renumbered
    assert [r["rid"] for r in g[0]["rows"]] == [5]
    assert [r["rid"] for r in g[1]["rows"]] == [6, 7]        # same head -> one card, 2 remedies
    assert g[1]["head"] == "Neural" and g[1]["most_affected"] == "CNS"


def test_group_layers_empty_head_rows_stand_alone():
    layers = [{"rid": 1, "head": "", "remedy": "A"}, {"rid": 2, "head": "", "remedy": "B"}]
    g = group_layers(layers)
    assert len(g) == 2                                        # empty heads never merge


def test_set_layer_order_reassigns_layer_by_group_position():
    cx = sqlite3.connect(":memory:")
    a = add_chain_row(cx, "9", 1, "Head A", "", "Rem A", "", "", "")
    b = add_chain_row(cx, "9", 2, "Head B", "", "Rem B1", "", "", "")
    b2 = add_chain_row(cx, "9", 2, "Head B", "", "Rem B2", "", "", "")
    # move layer B (both its rows) ahead of layer A
    set_layer_order(cx, "9", [[b, b2], [a]])
    order = [(l["head"], l["remedy"]) for l in ordered_chain(cx, "9")]
    assert order[0][0] == "Head B" and order[-1][0] == "Head A"
    # B's two remedies stay adjacent
    assert order[0][0] == "Head B" and order[1][0] == "Head B"


def test_reorder_frees_unconfirmed_scan_row():
    cx = sqlite3.connect(":memory:")
    live = add_chain_row(cx, "9", 1, "Live", "", "R1", confirmed=1, origin="live")
    scan = add_chain_row(cx, "9", 2, "Scan", "", "R2", confirmed=0, origin="scan")
    # by default the unconfirmed scan row trails
    assert [l["head"] for l in ordered_chain(cx, "9")] == ["Live", "Scan"]
    # dragging it to the top must stick (and promote it out of the bottom zone)
    set_layer_order(cx, "9", [[scan], [live]])
    rows = ordered_chain(cx, "9")
    assert [l["head"] for l in rows] == ["Scan", "Live"]
    assert all(l["zone"] == "top" for l in rows)          # scan row promoted -> stays put


def test_reorder_layers_route(tmp_path, monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: {"status": "none", "found": False,
                                                    "findings": [], "fresh": False}).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    r1 = client.post(f"/author/{tid}/row", json={"layer": 1, "head": "First", "remedy": "R1"}).get_json()["rid"]
    r2 = client.post(f"/author/{tid}/row", json={"layer": 2, "head": "Second", "remedy": "R2"}).get_json()["rid"]
    j = client.post(f"/author/{tid}/reorder-layers", json={"order": [[r2], [r1]]}).get_json()
    assert j["ok"] is True
    with sqlite3.connect(db) as cx:
        heads = [l["head"] for l in ordered_chain(cx, tid)]
    assert heads == ["Second", "First"]                      # order flipped + persisted
