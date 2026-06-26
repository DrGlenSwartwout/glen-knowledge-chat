"""The local app wires the E4L scan into the authoring flow: the header POST and a
GET /author/<id>/e4l return the reference panel, and narrative generation for an
authored (spoken) test feeds the scan to the LLM. A fake scan_lookup is injected so
these tests never touch the real e4l.db."""
import sqlite3
import pytest

from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_FRESH = {"status": "fresh", "found": True, "scan_id": 900, "scan_date": "2026-06-20",
          "days_ago": 4, "fresh": True, "window_days": 14,
          "message": "Recent E4L scan · 4 days ago",
          "findings": [{"rank": 1, "code": "LV3", "name": "Liver meridian",
                        "description": "detox and anger"}]}
_NONE = {"status": "none", "found": False, "scan_id": None, "scan_date": None,
         "days_ago": None, "fresh": False, "window_days": 14, "findings": [],
         "message": "No E4L scan on file"}


def _lookup(email):
    return _FRESH if (email or "").strip().lower() == "jane@x.com" else _NONE


def _app(db):
    return create_app(db, scan_lookup=_lookup)


def test_header_post_returns_panel_for_known_client(tmp_path):
    db = str(tmp_path / "chat_log.db")
    client = _app(db).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    r = client.post(f"/author/{tid}/header",
                    json={"name": "Jane", "email": "jane@x.com", "date": "2026-06-24"})
    j = r.get_json()
    assert j["ok"] is True
    assert j["e4l"]["status"] == "fresh"
    assert "Recent E4L scan" in j["html"] and "LV3" in j["html"]


def test_e4l_get_endpoint_reflects_stored_email(tmp_path):
    db = str(tmp_path / "chat_log.db")
    client = _app(db).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"email": "jane@x.com"})
    j = client.get(f"/author/{tid}/e4l").get_json()
    assert j["e4l"]["status"] == "fresh" and "LV3" in j["html"]


def test_e4l_get_none_when_no_email(tmp_path):
    db = str(tmp_path / "chat_log.db")
    client = _app(db).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    j = client.get(f"/author/{tid}/e4l").get_json()
    assert j["e4l"]["status"] == "none"
    assert "No E4L scan on file" in j["html"]


def test_author_page_loads_without_real_e4l_db(tmp_path, monkeypatch):
    # default scan_lookup + no e4l.db present -> page still renders, panel is "none"
    monkeypatch.setenv("E4L_DB", str(tmp_path / "absent-e4l.db"))
    db = str(tmp_path / "chat_log.db")
    client = create_app(db).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"email": "whoever@x.com"})
    assert client.get(f"/author/{tid}/e4l").get_json()["e4l"]["status"] == "none"


def test_narrative_generate_feeds_scan_for_authored_test(tmp_path):
    db = str(tmp_path / "chat_log.db")
    seen = {}
    app = create_app(db, complete=lambda s, u: seen.setdefault("u", u) or "Aloha Jane,",
                     scan_lookup=_lookup)
    client = app.test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "Jane", "email": "jane@x.com"})
    # an authored chain row so the report has content
    client.post(f"/author/{tid}/row",
                json={"layer": 1, "head": "Night", "remedy": "TMG", "dosage": "1 scoop"})
    r = client.post(f"/test/{tid}/generate", json={"notes": "mercury hx"})
    assert r.status_code == 200
    assert "LV3" in seen["u"]            # scan findings reached the LLM prompt
    assert "TMG" in seen["u"]            # authored chain reached the LLM prompt too
