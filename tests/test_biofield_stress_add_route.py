import sqlite3
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}


def _app(db):
    return create_app(db, scan_lookup=lambda e: _NONE)


def _new(client):
    return client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]


def _stress_state(client, tid):
    return client.get(f"/author/{tid}/stresses").get_json()["data"]


def test_add_active_stress_unassigned(tmp_path):
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    j = client.post(f"/author/{tid}/stress/add", json={"label": "Geopathic Stress"}).get_json()
    assert j["ok"] and isinstance(j["sid"], int) and j["layer"] is None
    s = _stress_state(client, tid)
    active = {x["label"] for x in s["active"]}
    assert "Geopathic Stress" in active


def test_add_new_term_persisted_to_vocab(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db).test_client()
    tid = _new(client)
    client.post(f"/author/{tid}/stress/add", json={"label": "Scalar Interference"})
    cx = sqlite3.connect(db)
    assert cx.execute("SELECT 1 FROM custom_stress_vocab WHERE LOWER(term)='scalar interference'").fetchone()


def test_add_balanced_on_layer(tmp_path):
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    # Give layer 1 a remedy row so the stress can be balanced by it.
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Liver", "remedy": "Liver Support"})
    j = client.post(f"/author/{tid}/stress/add",
                    json={"label": "Liver Congestion", "layer": 1}).get_json()
    assert j["ok"] and j["layer"] == 1
    s = _stress_state(client, tid)
    balanced = {x["label"] for x in s["balanced"]}
    assert "Liver Congestion" in balanced
    assert "Liver Congestion" not in {x["label"] for x in s["active"]}


def test_add_empty_label_rejected(tmp_path):
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    resp = client.post(f"/author/{tid}/stress/add", json={"label": "   "})
    assert resp.status_code == 400
