import importlib, os, sqlite3, sys
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

def test_cross_client_entitlement_isolation(tmp_path, monkeypatch):
    """A's grant must not leak to B's token — the library asset route is
    email-keyed authorization, not merely 'any valid portal token'."""
    appmod = _app(tmp_path, monkeypatch)
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
