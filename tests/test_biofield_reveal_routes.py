# tests/test_biofield_reveal_routes.py
"""Begin #4a - reveal ingest + reveal route."""
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals
    import begin_funnel
    with sqlite3.connect(db) as cx:
        biofield_reveals.init_table(cx)
        begin_funnel.init_journey_tables(cx)
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)
    return db


def test_ingest_stores_draft(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    client = app_module.app.test_client()
    r = client.post("/api/e4l/reveal-draft",
                    json={"email": "a@x.com", "scan_date": "2026-06-19",
                          "top_match": {"name": "Cistus", "slug": "cistus", "meaning": "Calm."},
                          "blurred": [{"kind": "binder"}], "source": "e4l-matcher"},
                    headers={"X-Cron-Secret": "k"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    from dashboard import biofield_reveals
    with sqlite3.connect(db) as cx:
        d = biofield_reveals.list_drafts(cx)
    assert len(d) == 1 and d[0]["top"]["name"] == "Cistus"


def test_ingest_requires_auth(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    client = app_module.app.test_client()
    r = client.post("/api/e4l/reveal-draft", json={"email": "a@x.com", "scan_date": "d",
                    "top_match": {"name": "X"}}, headers={"X-Cron-Secret": "wrong"})
    assert r.status_code == 401


def test_ingest_missing_email_400(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    client = app_module.app.test_client()
    r = client.post("/api/e4l/reveal-draft", json={"scan_date": "d", "top_match": {"name": "X"}},
                    headers={"X-Cron-Secret": "k"})
    assert r.status_code == 400


def test_console_list_drafts(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "ck", raising=False)
    from dashboard import biofield_reveals
    with sqlite3.connect(db) as cx:
        biofield_reveals.upsert_draft(cx, "a@x.com", "2026-06-19", {"name": "Cistus"}, [], "s")
    client = app_module.app.test_client()
    r = client.get("/api/console/biofield-reveals?key=ck")
    assert r.status_code == 200
    body = r.get_json()
    assert body["drafts"][0]["top"]["name"] == "Cistus"


def test_console_page_served(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "ck", raising=False)
    client = app_module.app.test_client()
    r = client.get("/console/biofield-reveals?key=ck")
    assert r.status_code == 200
    assert b"biofield" in r.data.lower()
