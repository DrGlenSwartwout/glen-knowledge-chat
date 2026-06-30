import importlib


def _reload_app(monkeypatch, tmp_path, enabled="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FIRESIDE_ENABLED", enabled)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_fireside_page_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="false")
    r = appmod.app.test_client().get("/begin/fireside")
    assert r.status_code == 404


def test_fireside_page_served_when_on(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    r = appmod.app.test_client().get("/begin/fireside")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "fireside" in body.lower()
    # sets the anonymous session cookie
    assert any("amg_session" in (h or "") for h in r.headers.getlist("Set-Cookie"))
