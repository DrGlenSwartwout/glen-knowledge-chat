"""Stress seed hook + stress routes + reorder wiring (Task 7).

Tests:
  - header-save triggers seed → stresses appear in GET /author/<id>/stresses
  - balance follows chain remedies (coverage lookup)
  - manual balance toggle via POST /author/<id>/stress/<sid>/balance
  - row-save with layer field reorders via reorder_chain
"""
import sqlite3
import pytest
from biofield_local_app import create_app
import dashboard.biofield_reveal_import as RI


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_FRESH = {"status": "fresh", "found": True, "scan_id": 900, "scan_date": "2026-06-24",
          "days_ago": 1, "fresh": True, "window_days": 14, "message": "ok",
          "findings": [{"code": "ED1", "name": "Membrane", "group": "infoceutical"},
                       {"code": "MR2", "name": "Calm", "group": "stress"}],
          "infoceuticals": [], "stresses": []}
_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}

# synthesis stub: ED1 covered by Neuro Magnesium -> required; MR2 optional
_SYNTH = {"found": True, "scan_id": 900, "scan_date": "2026-06-24", "days_ago": 1,
          "fresh": True, "layers": [{"n": 1, "title": "Ox", "summary": "",
          "most_affected": "Membrane", "remedy_name": "Neuro Magnesium", "codes": ["ED1"]}]}


def _app(db):
    return create_app(
        db, scan_lookup=lambda e: _FRESH if e == "j@x.com" else _NONE
    ).test_client()


def _new(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "J", "email": email, "date": "2026-06-25"})
    return tid


def test_header_save_seeds_stresses(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _SYNTH)
    db = str(tmp_path / "c.db")
    client = _app(db)
    tid = _new(client, "j@x.com")
    j = client.get(f"/author/{tid}/stresses").get_json()
    codes = {s["code"] for s in j["data"]["active"]} | {s["code"] for s in j["data"]["balanced"]}
    assert codes == {"ED1", "MR2"}
    bal = {s["balance"] for s in j["data"]["active"] + j["data"]["balanced"]}
    assert bal == {"required", "optional"}


def test_stress_lists_balance_follows_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _SYNTH)
    db = str(tmp_path / "c.db")
    client = _app(db)
    tid = _new(client, "j@x.com")
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Ox", "most_affected": "Membrane",
                                            "remedy": "Neuro Magnesium"})
    j = client.get(f"/author/{tid}/stresses").get_json()
    assert {s["code"] for s in j["data"]["balanced"]} == {"ED1"}
    assert {s["code"] for s in j["data"]["active"]} == {"MR2"}


def test_manual_balance_toggle(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _SYNTH)
    db = str(tmp_path / "c.db")
    client = _app(db)
    tid = _new(client, "j@x.com")
    sid = sqlite3.connect(db).execute(
        "SELECT id FROM biofield_auth_stress WHERE code='MR2'").fetchone()[0]
    client.post(f"/author/{tid}/stress/{sid}/balance", json={"value": True})
    j = client.get(f"/author/{tid}/stresses").get_json()
    assert "MR2" in {s["code"] for s in j["data"]["balanced"]}


def test_row_save_layer_change_reorders(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _NONE)  # no seed needed
    db = str(tmp_path / "c.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    r1 = client.post(f"/author/{tid}/row", json={"layer": 1, "head": "A", "remedy": "R1"}).get_json()["rid"]
    r2 = client.post(f"/author/{tid}/row", json={"layer": 2, "head": "B", "remedy": "R2"}).get_json()["rid"]
    client.post(f"/author/{tid}/row/{r2}", json={"layer": 1})   # move B to top
    from dashboard.biofield_authoring import ordered_chain
    assert [l["head"] for l in ordered_chain(sqlite3.connect(db), tid)] == ["B", "A"]
