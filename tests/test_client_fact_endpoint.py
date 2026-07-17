import importlib, sqlite3, sys
from pathlib import Path
import pytest
from dashboard import condition_programs, client_facts

def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try: return importlib.import_module("app")
    except Exception as e: pytest.skip(f"app not importable: {e}")

EMAIL = "pat@x.com"

@pytest.fixture
def wired(monkeypatch, tmp_path):
    app = _app()
    db = tmp_path / "chat_log.db"
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        condition_programs.init_table(cx)
        client_facts.init_table(cx)
        condition_programs.upsert(cx, "dry-amd", "Dry AMD", False,
            items=[{"slug": "macular-wellness-lutein", "name": "Macular Wellness Lutein"},
                   {"slug": "wholomega", "name": "WholOmega"}],
            modifiers=[{"when": "on_areds2", "action": "remove", "source": "client-reported",
                        "client_default": False,
                        "items": [{"slug": "macular-wellness-lutein", "name": "Macular Wellness Lutein"}]}])
    monkeypatch.setattr(app, "LOG_DB", db)
    monkeypatch.setattr(app, "_support_programs_enabled", lambda: True)
    monkeypatch.setattr(app, "_client_condition_for", lambda e: "dry-amd")
    # token -> email, ignoring any body email
    monkeypatch.setattr(app, "_portal_record_for", lambda cx, tok: {"email": EMAIL})
    return app

def test_unknown_key_rejected(wired):
    c = wired.app.test_client()
    r = c.post("/api/portal/tok/client-fact", json={"key": "evil", "value": True})
    assert r.status_code == 400

def test_set_removes_trio_and_uses_token_identity(wired):
    c = wired.app.test_client()
    r = c.post("/api/portal/tok/client-fact",
               json={"key": "on_areds2", "value": True, "email": "attacker@x.com"})
    assert r.status_code == 200
    slugs = [i["name"] for i in r.get_json()["support_program"]["items"]]
    assert "Macular Wellness Lutein" not in slugs  # removed
    # fact stored under the TOKEN email, not the body email
    with sqlite3.connect(wired.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        assert client_facts.get_facts(cx, EMAIL) == {"on_areds2": True}

def test_unset_restores_trio(wired):
    c = wired.app.test_client()
    c.post("/api/portal/tok/client-fact", json={"key": "on_areds2", "value": True})
    r = c.post("/api/portal/tok/client-fact", json={"key": "on_areds2", "value": False})
    slugs = [i["name"] for i in r.get_json()["support_program"]["items"]]
    assert "Macular Wellness Lutein" in slugs

def test_flag_off_404(wired, monkeypatch):
    monkeypatch.setattr(wired, "_support_programs_enabled", lambda: False)
    c = wired.app.test_client()
    r = c.post("/api/portal/tok/client-fact", json={"key": "on_areds2", "value": True})
    assert r.status_code == 404
