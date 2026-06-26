# tests/test_biofield_mine_comms_routes.py
import sqlite3
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}
_COMMS = {"intake_summary": "first name: Jane\n  Energy?: Low",
          "recent_inquiries": [{"main_challenge": "fatigue", "main_goal": "more energy"}],
          "recent_queries": [{"question": "why tired"}]}


def _app(db, comms, stresses):
    import json as _j
    return create_app(db, scan_lookup=lambda e: _NONE,
                      fetch_profile=lambda e: {},
                      fetch_recent_comms=lambda e: comms if e == "j@x.com" else {},
                      interpret_complete=lambda s, u: _j.dumps({"stresses": stresses}))


def _new(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "J", "email": email, "date": "2026-06-25"})
    return tid


def test_mine_comms_adds_comm_stresses(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _COMMS, ["Chronic fatigue", "Poor sleep"]).test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/mine-comms", json={}).get_json()
    assert "error" not in j
    data = client.get(f"/author/{tid}/stresses").get_json()["data"]
    labels = {x["label"] for x in data["active"] + data["balanced"]}
    sources = {x["source"] for x in data["active"] + data["balanced"]}
    assert {"Chronic fatigue", "Poor sleep"} <= labels and "comm" in sources


def test_mine_comms_no_email(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _COMMS, ["X"]).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    j = client.post(f"/author/{tid}/mine-comms", json={}).get_json()
    assert j["added"] == 0 and "error" in j


def test_mine_comms_empty(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, {}, []).test_client()
    tid = _new(client, "nobody@x.com")
    assert client.post(f"/author/{tid}/mine-comms", json={}).get_json()["added"] == 0


def test_header_save_mines_comms_run_once(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _COMMS, ["Chronic fatigue"]).test_client()
    tid = _new(client, "j@x.com")   # header-save hook should have mined comms already
    data = client.get(f"/author/{tid}/stresses").get_json()["data"]
    labels = {x["label"] for x in data["active"] + data["balanced"]}
    assert "Chronic fatigue" in labels
