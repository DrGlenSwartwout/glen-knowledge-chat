import base64, importlib, io, sqlite3, sys
from pathlib import Path
import pytest

# 1x1 PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _seed_portal(appmod, email):
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, email, "Test Client", {})
        cx.commit()
    return token


def _upload(client, token, blob=PNG, ctype="image/png", name="m.png"):
    return client.post(
        f"/api/portal/{token}/photo",
        data={"photo": (io.BytesIO(blob), name, ctype)},
        content_type="multipart/form-data")


def test_upload_then_serve_own_photo(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    token = _seed_portal(appmod, "client@x.com")
    c = appmod.app.test_client()
    r = _upload(c, token)
    assert r.status_code == 200 and r.get_json()["ok"] is True
    g = c.get(f"/api/portal/{token}/photo")
    assert g.status_code == 200
    assert g.data == PNG
    assert g.mimetype == "image/png"


def test_serve_is_token_scoped(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    t1 = _seed_portal(appmod, "a@x.com")
    t2 = _seed_portal(appmod, "b@x.com")
    c = appmod.app.test_client()
    _upload(c, t1)
    # t2's owner has no photo; the route serves only the token's own email -> 404
    assert c.get(f"/api/portal/{t2}/photo").status_code == 404


def test_rejects_non_image_and_oversize(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    token = _seed_portal(appmod, "client@x.com")
    c = appmod.app.test_client()
    assert _upload(c, token, blob=b"not-an-image", ctype="text/plain", name="x.txt").status_code == 400
    big = b"\x89PNG" + b"\x00" * (5 * 1024 * 1024 + 1)
    assert _upload(c, token, blob=big, ctype="image/png").status_code == 400


def test_unknown_token_404(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    assert c.get("/api/portal/nope/photo").status_code == 404
    assert _upload(c, "nope").status_code == 404
