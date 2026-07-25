# tests/test_bodymap_photo_routes.py
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


def _up(client, url, data=b"\xff\xd8\xffimg", name="face.jpg", ctype="image/jpeg"):
    return client.post(url, data={"photo": (io.BytesIO(data), name, ctype)},
                       content_type="multipart/form-data")


def test_upload_and_serve_a_slot(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    c = appmod.app.test_client()
    assert _up(c, f"/api/portal/{tok}/bodymap-photo?system=hand").status_code == 200
    r = c.get(f"/api/portal/{tok}/bodymap-photo?system=hand")
    assert r.status_code == 200 and r.data == b"\xff\xd8\xffimg"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Cache-Control"] == "private, no-store"


def test_side_makes_distinct_slots(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    c = appmod.app.test_client()
    _up(c, f"/api/portal/{tok}/bodymap-photo?system=iridology&side=left", data=b"LEYE")
    _up(c, f"/api/portal/{tok}/bodymap-photo?system=iridology&side=right", data=b"REYE")
    assert c.get(f"/api/portal/{tok}/bodymap-photo?system=iridology&side=left").data == b"LEYE"
    assert c.get(f"/api/portal/{tok}/bodymap-photo?system=iridology&side=right").data == b"REYE"


def test_face_falls_back_to_client_photos(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    from dashboard import client_photos as cph
    cx = sqlite3.connect(appmod.LOG_DB)
    cph.put(cx, "c@x.com", b"PORTRAIT", "image/jpeg", source="fmp"); cx.commit(); cx.close()
    # no body_map_photos face row -> face serves the client_photos portrait
    r = appmod.app.test_client().get(f"/api/portal/{tok}/bodymap-photo?system=face")
    assert r.status_code == 200 and r.data == b"PORTRAIT"


def test_face_slot_wins_over_client_photos(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    from dashboard import client_photos as cph
    cx = sqlite3.connect(appmod.LOG_DB)
    cph.put(cx, "c@x.com", b"PORTRAIT", "image/jpeg", source="fmp"); cx.commit(); cx.close()
    _up(appmod.app.test_client(), f"/api/portal/{tok}/bodymap-photo?system=face", data=b"FACESLOT")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/bodymap-photo?system=face")
    assert r.data == b"FACESLOT"


def test_nonface_missing_slot_404s(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    assert appmod.app.test_client().get(
        f"/api/portal/{tok}/bodymap-photo?system=hand").status_code == 404


def test_token_scoping(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _token(appmod, "other@x.com")
    tok_a = _token(appmod, "a@x.com")
    _up(appmod.app.test_client(), f"/api/portal/{tok_a}/bodymap-photo?system=hand", data=b"AHAND")
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    assert bmp.get(cx, "a@x.com", "hand", "") is not None
    assert bmp.get(cx, "other@x.com", "hand", "") is None


def test_unknown_system_400(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    assert _up(appmod.app.test_client(),
               f"/api/portal/{tok}/bodymap-photo?system=notasystem").status_code == 400


def test_size_and_type_rejected(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    c = appmod.app.test_client()
    assert _up(c, f"/api/portal/{tok}/bodymap-photo?system=hand", data=b"").status_code == 400
    big = b"x" * (5 * 1024 * 1024 + 1)
    assert _up(c, f"/api/portal/{tok}/bodymap-photo?system=hand", data=big).status_code == 400
    assert _up(c, f"/api/portal/{tok}/bodymap-photo?system=hand",
               data=b"pdf", ctype="application/pdf").status_code == 400


def test_html_slot_served_as_attachment(tmp_path, monkeypatch):
    # A slot somehow stored as text/html must never render inline.
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    bmp.put(cx, "c@x.com", "hand", "", b"<script>", "text/html", "console"); cx.commit(); cx.close()
    r = appmod.app.test_client().get(f"/api/portal/{tok}/bodymap-photo?system=hand")
    assert r.headers["Content-Type"].startswith("application/octet-stream")
    assert "attachment" in r.headers.get("Content-Disposition", "")


def test_console_photo_requires_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    r = appmod.app.test_client().post(
        "/api/console/bodymap-photo",
        data={"email": "c@x.com", "system": "hand",
              "photo": (io.BytesIO(b"H"), "h.jpg", "image/jpeg")},
        content_type="multipart/form-data")
    assert r.status_code == 401


def test_console_photo_with_key_stores_console_source(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    r = appmod.app.test_client().post(
        "/api/console/bodymap-photo?key=test-secret",
        data={"email": "C@x.com", "system": "hand",
              "photo": (io.BytesIO(b"H"), "h.jpg", "image/jpeg")},
        content_type="multipart/form-data")
    assert r.status_code == 200
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    assert bmp.get(cx, "c@x.com", "hand", "")["source"] == "console"
