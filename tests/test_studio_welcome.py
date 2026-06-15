import importlib

def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app as _app
    importlib.reload(_app)
    _app.app.config["TESTING"] = True
    return _app

def test_studio_welcome_served_no_store_and_sets_source(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    r = c.get("/studio")
    assert r.status_code == 200
    assert b"Studio" in r.data
    assert "no-store" in r.headers.get("Cache-Control", "")
    joined = " ".join(r.headers.getlist("Set-Cookie"))
    assert "rm_ref=studio" in joined
    assert "amg_session=" in joined

def test_studio_welcome_does_not_clobber_existing_ref(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    # seed an existing affiliate ref cookie on the client (adapt to your Werkzeug's set_cookie signature)
    try:
        c.set_cookie("rm_ref", "someaffiliate")
    except TypeError:
        c.set_cookie("localhost", "rm_ref", "someaffiliate")
    r = c.get("/studio")
    assert r.status_code == 200
    assert "rm_ref=studio" not in " ".join(r.headers.getlist("Set-Cookie"))
