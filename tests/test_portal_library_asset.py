import importlib, io, os, sqlite3, sys
from pathlib import Path

def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    # redirect the served asset dir to tmp_path so stubs never touch the real static/ tree
    monkeypatch.setattr(appmod, "STATIC", str(tmp_path))
    # ensure the pilot asset dir exists with tiny stand-in files for the test
    d = Path(tmp_path) / "ebooks" / "healing-glaucoma-starter"
    d.mkdir(parents=True, exist_ok=True)
    (d / "starter.pdf").write_bytes(b"%PDF-1.4 test")
    (d / "starter.mp3").write_bytes(b"ID3 test")
    return appmod

def _grant(appmod, email):
    from dashboard import client_portal as cp, portal_library as lib
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx); lib.init_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    lib.grant(cx, email, "healing-glaucoma-starter", "healingglaucoma.com")
    cx.commit()
    return tok

def test_serves_pdf_and_audio_when_entitled(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _grant(appmod, "c@x.com")
    # force the deterministic local-fallback path -- never let this test reach
    # a real R2 client (which would otherwise be constructed with live creds).
    monkeypatch.setattr(appmod, "_r2", lambda: _FakeR2(raise_exc=Exception("no r2 in test")))
    c = appmod.app.test_client()
    p = c.get(f"/api/portal/{tok}/library/healing-glaucoma-starter/pdf")
    assert p.status_code == 200 and p.data.startswith(b"%PDF")
    a = c.get(f"/api/portal/{tok}/library/healing-glaucoma-starter/audio")
    assert a.status_code == 200 and a.data.startswith(b"ID3")

def test_denied_when_not_entitled(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB); cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, "nolib@x.com", "T"); cx.commit()   # portal but no grant
    r = appmod.app.test_client().get(f"/api/portal/{tok}/library/healing-glaucoma-starter/pdf")
    assert r.status_code == 404

def test_bad_asset_and_unknown_token_404(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _grant(appmod, "c@x.com")
    c = appmod.app.test_client()
    assert c.get(f"/api/portal/{tok}/library/healing-glaucoma-starter/exe").status_code == 404
    assert c.get("/api/portal/nope/library/healing-glaucoma-starter/pdf").status_code == 404

class _FakeBody:
    def __init__(self, data):
        self._data = data

    def iter_chunks(self, chunk_size=65536):
        yield self._data


class _FakeR2:
    """Stand-in R2 client. `get_object` returns a canned object dict, or raises
    if `raise_exc` is set (simulating a missing object / network failure)."""

    def __init__(self, body=b"", content_type="audio/mpeg", content_length=None,
                 content_range=None, raise_exc=None):
        self.body = body
        self.content_type = content_type
        self.content_length = content_length if content_length is not None else len(body)
        self.content_range = content_range
        self.raise_exc = raise_exc
        self.calls = []

    def get_object(self, **kw):
        self.calls.append(kw)
        if self.raise_exc:
            raise self.raise_exc
        obj = {
            "Body": _FakeBody(self.body),
            "ContentType": self.content_type,
            "ContentLength": self.content_length,
        }
        if self.content_range:
            obj["ContentRange"] = self.content_range
        return obj


def test_audio_streams_from_r2_when_audio_key_set(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _grant(appmod, "c@x.com")
    fake = _FakeR2(body=b"R2AUDIOBYTES", content_type="audio/mpeg")
    monkeypatch.setattr(appmod, "_r2", lambda: fake)
    # sabotage the local fallback path so a pass here proves R2 (not disk) served it
    local_path = Path(tmp_path) / "ebooks" / "healing-glaucoma-starter" / "starter.mp3"
    local_path.unlink()
    c = appmod.app.test_client()
    r = c.get(f"/api/portal/{tok}/library/healing-glaucoma-starter/audio")
    assert r.status_code == 200
    assert r.data == b"R2AUDIOBYTES"
    assert r.headers.get("Content-Type") == "audio/mpeg"
    assert r.headers.get("Cache-Control") == "private, no-store"
    assert fake.calls and fake.calls[0]["Key"] == "ebooks/healing-glaucoma/starter/audio.mp3"


def test_audio_range_request_yields_206_with_content_range(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _grant(appmod, "c@x.com")
    fake = _FakeR2(body=b"PARTIAL", content_range="bytes 0-6/100")
    monkeypatch.setattr(appmod, "_r2", lambda: fake)
    c = appmod.app.test_client()
    r = c.get(f"/api/portal/{tok}/library/healing-glaucoma-starter/audio",
              headers={"Range": "bytes=0-6"})
    assert r.status_code == 206
    assert r.headers.get("Content-Range") == "bytes 0-6/100"
    assert fake.calls[0].get("Range") == "bytes=0-6"


def test_audio_falls_back_to_local_when_r2_raises(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _grant(appmod, "c@x.com")
    fake = _FakeR2(raise_exc=RuntimeError("object not found"))
    monkeypatch.setattr(appmod, "_r2", lambda: fake)
    c = appmod.app.test_client()
    r = c.get(f"/api/portal/{tok}/library/healing-glaucoma-starter/audio")
    assert r.status_code == 200
    assert r.data.startswith(b"ID3")  # the local stand-in seeded by _app()
    assert r.headers.get("Cache-Control") == "private, no-store"


def test_entitlement_checked_before_any_r2_call(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    calls = []

    class _Boom:
        def get_object(self, **kw):
            calls.append(kw)
            raise AssertionError("R2 should not be called before entitlement check")

    monkeypatch.setattr(appmod, "_r2", lambda: _Boom())
    c = appmod.app.test_client()
    # unknown token
    r1 = c.get("/api/portal/nope/library/healing-glaucoma-starter/audio")
    assert r1.status_code == 404
    # bad asset name
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, "noent@x.com", "T")
    cx.commit()
    r2 = c.get(f"/api/portal/{tok}/library/healing-glaucoma-starter/exe")
    assert r2.status_code == 404
    # not entitled
    r3 = c.get(f"/api/portal/{tok}/library/healing-glaucoma-starter/audio")
    assert r3.status_code == 404
    assert calls == []


def test_cross_client_entitlement_isolation(tmp_path, monkeypatch):
    """A's grant must not leak to B's token — the library asset route is
    email-keyed authorization, not merely 'any valid portal token'."""
    appmod = _app(tmp_path, monkeypatch)
    # force the deterministic local-fallback path -- never let this test reach
    # a real R2 client (which would otherwise be constructed with live creds).
    monkeypatch.setattr(appmod, "_r2", lambda: _FakeR2(raise_exc=Exception("no r2 in test")))
    from dashboard import client_portal as cp, portal_library as lib
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx); lib.init_table(cx)
    tok_a = cp.ensure_token(cx, "a@x.com", "A")
    tok_b = cp.ensure_token(cx, "b@x.com", "B")
    lib.grant(cx, "a@x.com", "healing-glaucoma-starter", "healingglaucoma.com")
    cx.commit()
    c = appmod.app.test_client()
    r_a = c.get(f"/api/portal/{tok_a}/library/healing-glaucoma-starter/pdf")
    assert r_a.status_code == 200 and r_a.data.startswith(b"%PDF")
    r_b = c.get(f"/api/portal/{tok_b}/library/healing-glaucoma-starter/pdf")
    assert r_b.status_code == 404
