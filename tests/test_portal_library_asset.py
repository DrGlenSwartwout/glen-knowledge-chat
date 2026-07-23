import importlib, os, sqlite3, sys
from pathlib import Path

def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    # ensure the pilot asset dir exists with tiny stand-in files for the test
    d = Path(appmod.STATIC) / "ebooks" / "healing-glaucoma-starter"
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
