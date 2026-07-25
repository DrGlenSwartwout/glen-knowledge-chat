import importlib, io, sqlite3, sys
from pathlib import Path


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONSOLE_SECRET", "test-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _token(appmod, email):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    cx.commit(); cx.close()
    return tok


def _seed_photo(appmod, email, system, side=""):
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    bmp.put(cx, email, system, side, b"img", "image/jpeg", "portal-self"); cx.commit(); cx.close()


T = {"mx": 1.5, "my": -0.5, "tx": 300.0, "ty": 12.25}


def test_put_then_get_transform(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com"); _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put(f"/api/portal/{tok}/bodymap-transform?system=face", json=T).status_code == 200
    assert c.get(f"/api/portal/{tok}/bodymap-transform?system=face").get_json() == T


def test_get_missing_transform_404(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com"); _seed_photo(appmod, "c@x.com", "face")
    assert appmod.app.test_client().get(
        f"/api/portal/{tok}/bodymap-transform?system=face").status_code == 404


def test_put_malformed_transform_400(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com"); _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put(f"/api/portal/{tok}/bodymap-transform?system=face",
                 json={"mx": 1, "my": 0, "tx": 2}).status_code == 400
    assert c.get(f"/api/portal/{tok}/bodymap-transform?system=face").status_code == 404


def test_transform_token_scoping(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _token(appmod, "other@x.com"); _seed_photo(appmod, "other@x.com", "face")
    tok_a = _token(appmod, "a@x.com"); _seed_photo(appmod, "a@x.com", "face")
    appmod.app.test_client().put(f"/api/portal/{tok_a}/bodymap-transform?system=face", json=T)
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    assert bmp.get_transform(cx, "a@x.com", "face", "") == T
    assert bmp.get_transform(cx, "other@x.com", "face", "") is None


def test_unknown_system_400(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    assert appmod.app.test_client().put(
        f"/api/portal/{tok}/bodymap-transform?system=notasystem", json=T).status_code == 400


def test_console_transform_requires_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert appmod.app.test_client().put(
        "/api/console/bodymap-transform?email=c@x.com&system=face", json=T).status_code == 401


def test_console_transform_roundtrip(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put("/api/console/bodymap-transform?key=test-secret&email=c@x.com&system=face",
                 json=T).status_code == 200
    assert c.get("/api/console/bodymap-transform?key=test-secret&email=c@x.com&system=face"
                 ).get_json() == T


def test_portal_put_unparseable_body_does_not_clear(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com"); _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put(f"/api/portal/{tok}/bodymap-transform?system=face", json=T).status_code == 200
    resp = c.put(f"/api/portal/{tok}/bodymap-transform?system=face",
                 data=b"not json", content_type="text/plain")
    assert resp.status_code == 400
    assert c.get(f"/api/portal/{tok}/bodymap-transform?system=face").get_json() == T


def test_portal_put_non_object_json_400(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com"); _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put(f"/api/portal/{tok}/bodymap-transform?system=face",
                 json=[1, 2, 3, 4]).status_code == 400
    assert c.put(f"/api/portal/{tok}/bodymap-transform?system=face",
                 json="x").status_code == 400


def test_console_put_unparseable_body_does_not_clear(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put("/api/console/bodymap-transform?key=test-secret&email=c@x.com&system=face",
                 json=T).status_code == 200
    resp = c.put("/api/console/bodymap-transform?key=test-secret&email=c@x.com&system=face",
                 data=b"not json", content_type="text/plain")
    assert resp.status_code == 400
    assert c.get("/api/console/bodymap-transform?key=test-secret&email=c@x.com&system=face"
                 ).get_json() == T


def test_console_put_non_object_json_400(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _seed_photo(appmod, "c@x.com", "face")
    c = appmod.app.test_client()
    assert c.put("/api/console/bodymap-transform?key=test-secret&email=c@x.com&system=face",
                 json=[1, 2, 3, 4]).status_code == 400
    assert c.put("/api/console/bodymap-transform?key=test-secret&email=c@x.com&system=face",
                 json="x").status_code == 400
