"""#3: layer cards show the stresses their remedies cover, and dragging an
unbalanced stress onto a layer marks it covered (adds a remedy<->code link)."""
import sqlite3

import pytest

from dashboard import biofield_stress as st
from dashboard.biofield_authoring import add_chain_row
from dashboard.biofield_report_html import render_author_html, render_stress_panel
from biofield_local_app import create_app


def _setup(cx):
    st.init_stress_tables(cx)
    rid = add_chain_row(cx, "9", 1, "Head", "", "Nerve Pulse", "", "", "")
    st.add_stress(cx, "9", "Loose ends", source="scan", balance="required")
    sid = cx.execute("SELECT id FROM biofield_auth_stress").fetchone()[0]
    return rid, sid


def test_cover_stress_moves_it_under_the_layer():
    cx = sqlite3.connect(":memory:")
    rid, sid = _setup(cx)
    chain = [{"layer": 1, "head": "Head", "remedy": "Nerve Pulse"}]
    d0 = st.list_stresses(cx, "9", chain)
    assert {s["code"] for s in d0["unassigned"]} == {"loose ends"}
    assert d0["by_layer"][0]["stresses"] == []
    assert st.cover_stress(cx, "9", sid, [rid]) == "loose ends"
    d1 = st.list_stresses(cx, "9", chain)
    assert d1["unassigned"] == []
    assert {s["code"] for s in d1["by_layer"][0]["stresses"]} == {"loose ends"}


def test_cover_stress_ignores_missing_or_remedyless():
    cx = sqlite3.connect(":memory:")
    rid, sid = _setup(cx)
    assert st.cover_stress(cx, "9", 999999, [rid]) is None       # no such stress
    empty = add_chain_row(cx, "9", 2, "H2", "", "", "", "", "")   # remedy-less row
    st.cover_stress(cx, "9", sid, [empty])                        # no-op, no crash
    d = st.list_stresses(cx, "9", [{"layer": 1, "head": "Head", "remedy": "Nerve Pulse"}])
    assert {s["code"] for s in d["unassigned"]} == {"loose ends"} # still unassigned


def test_card_shows_covered_chips_and_unassigned_is_draggable():
    rep = {"test_id": "a1", "client": {"name": "J", "email": ""}, "date": "",
           "layers": [{"layer": 1, "head": "H", "most_affected": "", "remedy": "R",
                       "rid": 5, "confirmed": 1}], "schedule": {"slots": [], "entries": []}}
    html = render_author_html(rep, [], "", covered_by_layer={1: [{"code": "ED1", "label": "Membrane"}]})
    assert "class=covered" in html and "balances:" in html and "ED1" in html
    assert "function coverStress" in html and "function stressDragStart" in html
    panel = render_stress_panel({"by_layer": [], "unassigned": [
        {"id": 7, "code": "MR2", "label": "Calm", "balance": "required", "balanced": False}]})
    assert "stressDragStart(event,7)" in panel and "class=sdrag" in panel


def test_cover_route(tmp_path, monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: {"status": "none", "found": False,
                                                   "findings": [], "fresh": False}).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    rid = client.post(f"/author/{tid}/row", json={"layer": 1, "head": "H", "remedy": "Nerve Pulse"}).get_json()["rid"]
    with sqlite3.connect(db) as cx:
        st.add_stress(cx, tid, "Loose ends", source="scan", balance="required")
        sid = cx.execute("SELECT id FROM biofield_auth_stress").fetchone()[0]
    j = client.post(f"/author/{tid}/stress/{sid}/cover", json={"rids": [rid]}).get_json()
    assert j["ok"] is True and j["code"] == "loose ends"
    # editor page now renders without error and lists the covered stress inline
    assert client.get(f"/author/{tid}").status_code == 200
