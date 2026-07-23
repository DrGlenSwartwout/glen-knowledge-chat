import importlib, sqlite3, sys
from pathlib import Path

def _app(tmp_path, monkeypatch, hub="1", onboarding="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PORTAL_HUB_ENABLED", hub)
    if onboarding is None:
        monkeypatch.delenv("PORTAL_ONBOARDING_ENABLED", raising=False)
    else:
        monkeypatch.setenv("PORTAL_ONBOARDING_ENABLED", onboarding)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod

def _seed(appmod, email):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    cx.commit()
    return tok

def test_onboarding_status_rewrites_anchor_hrefs(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, hub="1", onboarding="1")
    tok = _seed(appmod, "c@x.com")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/onboarding")
    assert r.status_code == 200
    body = r.get_json()
    assert body["enabled"] is True
    status = body["status"]
    assert "phases" in status and len(status["phases"]) > 0
    hrefs = [
        st["href"]
        for ph in status["phases"]
        for st in ph["steps"]
    ]
    rewritten = [h for h in hrefs if h.startswith(f"/portal/{tok}#")]
    assert rewritten, f"expected a rewritten anchor href, got: {hrefs}"
    assert not any(h.startswith("#") for h in hrefs)

def test_onboarding_unknown_token_404(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert appmod.app.test_client().get("/api/portal/nope/onboarding").status_code == 404

def test_onboarding_disabled_by_default_when_sub_flag_unset(tmp_path, monkeypatch):
    """Dedicated dark-launch flag: PORTAL_HUB_ENABLED alone must NOT flip the
    onboarding tile on -- PORTAL_ONBOARDING_ENABLED must be set independently."""
    appmod = _app(tmp_path, monkeypatch, hub="1", onboarding=None)
    tok = _seed(appmod, "c@x.com")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/onboarding")
    assert r.status_code == 200
    assert r.get_json()["enabled"] is False
