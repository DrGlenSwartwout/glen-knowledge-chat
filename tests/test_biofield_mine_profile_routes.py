# tests/test_biofield_mine_profile_routes.py
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
_PROFILE = {"email": "j@x.com", "tags": ["Inflammation"], "conditions": "Eczema",
            "challenges": "always tired"}


def _app(db, profile, stresses):
    import json as _j
    return create_app(db, scan_lookup=lambda e: _NONE,
                      fetch_profile=lambda e: profile if e == "j@x.com" else {},
                      interpret_complete=lambda s, u: _j.dumps({"stresses": stresses}))


def _new(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "J", "email": email, "date": "2026-06-25"})
    return tid


def test_mine_profile_adds_tag_stresses(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _PROFILE, ["Chronic fatigue"]).test_client()
    tid = _new(client, "j@x.com")
    # With always-on profile mining from the header-save hook (_seed_stresses now mines
    # the profile even when no scan exists), the explicit /mine-profile call may return
    # added=0 (labels already present).  Assert end-state: no error, expected labels
    # with source='tag' visible in /stresses.
    j = client.post(f"/author/{tid}/mine-profile", json={}).get_json()
    assert "error" not in j
    data = client.get(f"/author/{tid}/stresses").get_json()["data"]
    labels = {x["label"] for x in data["active"] + data["balanced"]}
    sources = {x["source"] for x in data["active"] + data["balanced"]}
    assert {"Inflammation", "Eczema", "Chronic fatigue"} <= labels and "tag" in sources


def test_header_save_mines_profile_when_no_scan(tmp_path):
    """header-save alone seeds tag stresses even when no E4L scan exists (Return-C fix)."""
    db = str(tmp_path / "c.db")
    # scan_lookup always returns not-found; fetch_profile returns a real profile
    client = _app(db, _PROFILE, ["Chronic fatigue"]).test_client()
    tid = _new(client, "j@x.com")
    # No explicit /mine-profile call — header-save should have triggered mining
    data = client.get(f"/author/{tid}/stresses").get_json()["data"]
    labels = {x["label"] for x in data["active"] + data["balanced"]}
    sources = {x["source"] for x in data["active"] + data["balanced"]}
    assert {"Inflammation", "Eczema", "Chronic fatigue"} <= labels
    assert "tag" in sources


def test_mine_profile_no_email(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _PROFILE, []).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    j = client.post(f"/author/{tid}/mine-profile", json={}).get_json()
    assert j["added"] == 0 and "error" in j


def test_mine_profile_empty_profile(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, {}, []).test_client()
    tid = _new(client, "nobody@x.com")
    assert client.post(f"/author/{tid}/mine-profile", json={}).get_json()["added"] == 0


def test_mine_profile_failure_is_best_effort(tmp_path):
    db = str(tmp_path / "c.db")
    def boom(e):
        raise RuntimeError("people hub down")
    client = create_app(db, scan_lookup=lambda e: _NONE, fetch_profile=boom,
                        interpret_complete=lambda s, u: "{}").test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/mine-profile", json={}).get_json()
    assert j["added"] == 0 and "error" in j
