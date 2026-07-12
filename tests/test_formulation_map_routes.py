"""Formulation-map curation page + API routes (add / remove / reorder)."""
import sqlite3
import pytest
from biofield_local_app import create_app

_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


def _e4l(tmp_path):
    p = str(tmp_path / "e4l.db")
    cx = sqlite3.connect(p)
    cx.execute("CREATE TABLE e4l_items(code TEXT, category TEXT, subcategory TEXT, name TEXT, "
               "full_name TEXT, e4l_description TEXT, clinical_notes TEXT, sort_order INT)")
    cx.execute("INSERT INTO e4l_items(code,name,sort_order) VALUES('ED5','Circulation Driver',1)")
    cx.commit()
    cx.close()
    return p


def _client(tmp_path):
    return create_app(str(tmp_path / "c.db"), e4l_db=_e4l(tmp_path),
                      scan_lookup=lambda e: _NONE).test_client()


def test_page_renders_codes(tmp_path):
    r = _client(tmp_path).get("/formulation-map")
    assert r.status_code == 200
    assert b"Formulation map" in r.data and b"ED5" in r.data


def test_add_reorder_remove_roundtrip(tmp_path):
    c = _client(tmp_path)
    c.post("/api/formulation-map/add", json={"code": "ED5", "remedy": "Heart Health"})
    j = c.post("/api/formulation-map/add", json={"code": "ED5", "remedy": "Vein Support"}).get_json()
    assert [m["name"] for m in j["mappings"]] == ["Heart Health", "Vein Support"]
    fids = [m["formulation_id"] for m in j["mappings"]]
    j2 = c.post("/api/formulation-map/reorder", json={"code": "ED5", "order": [fids[1], fids[0]]}).get_json()
    assert [m["name"] for m in j2["mappings"]] == ["Vein Support", "Heart Health"]
    j3 = c.post("/api/formulation-map/remove", json={"code": "ED5", "formulation_id": fids[0]}).get_json()
    assert [m["name"] for m in j3["mappings"]] == ["Vein Support"]
