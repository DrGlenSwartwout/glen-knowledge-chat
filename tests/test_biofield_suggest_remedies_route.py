# tests/test_biofield_suggest_remedies_route.py
import sqlite3
import pytest
from biofield_local_app import create_app
from dashboard.biofield_stress import init_stress_tables, seed_from_scan


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}


def _seed(db, tid_num):
    cx = sqlite3.connect(db)
    init_stress_tables(cx)
    seed_from_scan(cx, "a" + str(tid_num),
                   [{"code": "ED1", "name": "Membrane"}, {"code": "ES3", "name": "Lymph"}],
                   {"neuro magnesium": {"ED1", "ES3"}})
    cx.close()


def test_route_returns_suggestion(tmp_path):
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    _seed(db, int(tid.lstrip("a")))
    j = client.get(f"/author/{tid}/suggest-remedies").get_json()
    assert j["picks"] == [{"remedy": "neuro magnesium", "covers": ["Membrane", "Lymph"]}]
    assert j["uncovered"] == [] and "html" in j


def test_route_reflects_live_chain(tmp_path):
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    _seed(db, int(tid.lstrip("a")))
    # put the covering remedy on the chain -> stresses balanced -> nothing to suggest
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "x", "remedy": "Neuro Magnesium"})
    j = client.get(f"/author/{tid}/suggest-remedies").get_json()
    assert j["picks"] == [] and j["uncovered"] == []
