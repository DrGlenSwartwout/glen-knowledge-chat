"""Pathway-review page + API routes (decide / undo / new canonical / direction)."""
import sqlite3

import pytest

from biofield_local_app import create_app

_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    import dashboard as _d
    monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)


def _ingredients(tmp_path):
    from dashboard import pathway_review as pr
    p = str(tmp_path / "ingredients.db")
    cx = sqlite3.connect(p)
    pr.init_tables(cx)
    cx.execute("INSERT INTO ingredient_pathways(id,ingredient_id,pathway,effect) "
               "VALUES(1,1,'NF-κB nuclear translocation','inhibits')")
    cx.execute("INSERT INTO canonical_pathways(id,slug,label,family,status,created_at) "
               "VALUES(1,'nf-kb','NF-κB','inflammation','proposed',datetime('now'))")
    # One atom reaching two ingredients, so it qualifies for the shared-atom batch.
    cx.executemany("INSERT INTO pathway_atoms(pathway_row_id,ingredient_id,position,"
                   "atom,atom_key,is_annotation) VALUES(?,?,?,?,?,0)",
                   [(1, 1, 0, "NF-κB", "nf kappab"), (1, 2, 0, "NF-κB", "nf kappab")])
    cx.execute("INSERT INTO pathway_atom_map(atom_key,canonical_id,decision,source) "
               "VALUES('nf kappab',1,'proposed','ai')")
    cx.execute("INSERT INTO pathway_effect_direction(effect_key,direction,confidence,"
               "decision,n_rows) VALUES('crosses','unknown','low','proposed',3)")
    cx.commit()
    cx.close()
    return p


def _client(tmp_path):
    return create_app(str(tmp_path / "c.db"), e4l_db=str(tmp_path / "e.db"),
                      ingredients_db=_ingredients(tmp_path),
                      scan_lookup=lambda e: _NONE).test_client()


def test_page_renders_the_pending_atom(tmp_path):
    r = _client(tmp_path).get("/pathway-review")
    assert r.status_code == 200
    body = r.data.decode()
    assert "Pathway review" in body
    assert "nf kappab" in body and "2 ingredients" in body
    # the raw string it came from is shown as the evidence Glen judges against
    assert "NF-κB nuclear translocation" in body
    # and the low-confidence effect phrasing gets its own queue
    assert "crosses" in body


def test_decide_then_undo_roundtrip(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/pathway-review/decide",
               json={"atom_key": "nf kappab", "decision": "confirmed", "canonical_id": 1})
    assert r.status_code == 200 and r.get_json()["ok"]
    # settled atoms leave the queue
    assert "nf kappab" not in c.get("/pathway-review").data.decode()
    assert c.post("/api/pathway-review/undo", json={"atom_key": "nf kappab"}).get_json()["ok"]
    assert "nf kappab" in c.get("/pathway-review").data.decode()


def test_decide_rejects_an_unknown_verb(tmp_path):
    r = _client(tmp_path).post("/api/pathway-review/decide",
                               json={"atom_key": "nf kappab", "decision": "sure-why-not"})
    assert r.status_code == 400 and not r.get_json()["ok"]


def test_new_canonical_route(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/pathway-review/canonical",
               json={"label": "NLRP3 inflammasome", "family": "inflammation"})
    assert r.status_code == 200 and r.get_json()["canonical"]["slug"] == "nlrp3-inflammasome"
    assert c.post("/api/pathway-review/canonical", json={"label": ""}).status_code == 400


def test_direction_route_sets_and_validates(tmp_path):
    c = _client(tmp_path)
    assert c.post("/api/pathway-review/direction",
                  json={"effect_key": "crosses", "direction": "substrate"}).get_json()["ok"]
    assert c.post("/api/pathway-review/direction",
                  json={"effect_key": "crosses", "direction": "sideways"}).status_code == 400


def test_queue_route_paginates(tmp_path):
    c = _client(tmp_path)
    j = c.get("/api/pathway-review/queue?offset=0").get_json()
    assert j["ok"] and j["count"] == 1 and "nf kappab" in j["html"]
    assert c.get("/api/pathway-review/queue?offset=50").get_json()["count"] == 0
