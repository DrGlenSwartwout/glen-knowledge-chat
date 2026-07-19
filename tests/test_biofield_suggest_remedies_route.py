# tests/test_biofield_suggest_remedies_route.py
import sqlite3
import pytest
from biofield_local_app import create_app
from dashboard.biofield_stress import init_stress_tables, seed_from_scan


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)


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


def test_route_returns_layer_candidates(tmp_path):
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    _seed(db, int(tid.lstrip("a")))
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Membrane", "remedy": "Neuro Magnesium"})
    j = client.get(f"/author/{tid}/suggest-remedies").get_json()
    assert "layer_candidates" in j and isinstance(j["layer_candidates"], list)
    L1 = next(L for L in j["layer_candidates"] if L["n"] == 1)
    assert any(c["remedy"].lower() == "neuro magnesium" and c.get("is_default")
               for c in L1["candidates"])


def test_layer_select_swaps_chain_remedy_and_learns(tmp_path):
    from dashboard.biofield_stress import get_saved_remedy_set
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    _seed(db, int(tid.lstrip("a")))
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Membrane", "remedy": "Neuro Magnesium"})
    j = client.post(f"/author/{tid}/layer/1/select", json={"remedy": "Nerve Star"}).get_json()
    assert j["ok"] and "layer_candidates" in j
    cx = sqlite3.connect(db)
    rem = cx.execute("SELECT remedy FROM biofield_auth_chain WHERE test_id=? AND layer=1",
                     (int(tid.lstrip("a")),)).fetchone()[0]
    assert "nerve star" in rem.lower()                        # chain row swapped
    saved = get_saved_remedy_set(cx, tid) or []
    assert any("nerve star" in s.lower() for s in saved)      # pick fed the learning loop


def test_route_reflects_live_chain(tmp_path):
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    _seed(db, int(tid.lstrip("a")))
    # put the covering remedy on the chain -> stresses balanced -> nothing to suggest
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "x", "remedy": "Neuro Magnesium"})
    j = client.get(f"/author/{tid}/suggest-remedies").get_json()
    assert j["picks"] == [] and j["uncovered"] == []
