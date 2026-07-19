import sqlite3
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}


def _app(db, stresses):
    # interpret_complete returns a fixed stresses payload; scan_lookup finds nothing (no seeding)
    import json as _j
    return create_app(db, scan_lookup=lambda e: _NONE,
                      interpret_complete=lambda s, u: _j.dumps({"stresses": stresses}))


def _new(client):
    return client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]


def test_capture_adds_voice_stresses(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, ["Liver Congestion", "Adrenal Fatigue"]).test_client()
    tid = _new(client)
    client.post(f"/author/{tid}/session", json={"transcript": "liver congestion, adrenal fatigue"})
    j = client.post(f"/author/{tid}/capture-stresses", json={}).get_json()
    assert j["added"] == 2
    s = client.get(f"/author/{tid}/stresses").get_json()["data"]
    labels = {x["label"] for x in s["active"] + s["balanced"]}
    assert labels == {"Liver Congestion", "Adrenal Fatigue"}


def test_capture_empty_transcript_errors(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, ["X"]).test_client()
    tid = _new(client)
    j = client.post(f"/author/{tid}/capture-stresses", json={}).get_json()
    assert j["added"] == 0 and "error" in j


def test_capture_then_chain_row_balances_by_label(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, ["Liver Congestion"]).test_client()
    tid = _new(client)
    client.post(f"/author/{tid}/session", json={"transcript": "liver congestion"})
    client.post(f"/author/{tid}/capture-stresses", json={})
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Liver Congestion", "remedy": "Hepato Tonic"})
    s = client.get(f"/author/{tid}/stresses").get_json()["data"]
    assert [x["label"] for x in s["balanced"]] == ["Liver Congestion"]
    assert s["balanced"][0]["balanced_by"] == "Hepato Tonic"
