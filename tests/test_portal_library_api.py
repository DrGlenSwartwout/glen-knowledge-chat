import importlib, sqlite3, sys
from pathlib import Path

def _app(tmp_path, monkeypatch, hub="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PORTAL_HUB_ENABLED", hub)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod

def _seed(appmod, email):
    from dashboard import client_portal as cp, portal_library as lib
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx); lib.init_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    lib.grant(cx, email, "healing-glaucoma-starter", "healingglaucoma.com")
    cx.commit()
    return tok

def test_library_lists_granted_items_with_urls(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, hub="1")
    tok = _seed(appmod, "c@x.com")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/library")
    assert r.status_code == 200
    body = r.get_json()
    assert body["enabled"] is True and len(body["items"]) == 1
    it = body["items"][0]
    assert it["slug"] == "healing-glaucoma-starter"
    assert it["pdf_url"] == f"/api/portal/{tok}/library/healing-glaucoma-starter/pdf"
    assert it["audio_url"] == f"/api/portal/{tok}/library/healing-glaucoma-starter/audio"

def test_library_enabled_false_when_flag_off(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, hub="0")
    tok = _seed(appmod, "d@x.com")
    body = appmod.app.test_client().get(f"/api/portal/{tok}/library").get_json()
    assert body["enabled"] is False

def test_library_unknown_token_404(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert appmod.app.test_client().get("/api/portal/nope/library").status_code == 404
