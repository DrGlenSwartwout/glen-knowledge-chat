"""Token-gated triage submit + prefill route: POST seeds condition programs
via condition_triage.seed_from_triage; GET returns stored answers for prefill."""
import importlib
import sqlite3
import sys
from pathlib import Path


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed(appmod, email):
    from dashboard import client_portal as cp, condition_programs as cprog
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    cp.init_client_portal_table(cx)
    cprog.init_table(cx)
    cprog.upsert(cx, "glaucoma-elevated-iop", "Glaucoma - Elevated IOP", False,
                 [{"slug": "neuroprotect", "name": "Neuroprotect"}])
    cprog.upsert(cx, "glaucoma-normal-iop", "Glaucoma - Normal IOP", False,
                 [{"slug": "eye-calm", "name": "Eye Calm"}])
    tok = cp.ensure_token(cx, email, "T")
    cx.commit()
    return tok


def test_post_triage_seeds_programs(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-a@x.com")
    r = appmod.app.test_client().post(
        f"/api/portal/{tok}/triage", json={"condition": "glaucoma", "iop_od": 25})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["programs"] == ["glaucoma-elevated-iop"]
    assert body["seeded"] > 0


def test_get_triage_returns_stored_answers(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _seed(appmod, "triage-b@x.com")
    appmod.app.test_client().post(
        f"/api/portal/{tok}/triage", json={"condition": "glaucoma", "iop_od": 25})
    r = appmod.app.test_client().get(f"/api/portal/{tok}/triage?condition=glaucoma")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    triage = body["triage"]
    assert triage["iop_od"] == "25"
    assert triage["resolved_programs"] == ["glaucoma-elevated-iop"]


def test_unknown_token_404(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert appmod.app.test_client().post(
        "/api/portal/nope/triage", json={"condition": "glaucoma"}).status_code == 404
    assert appmod.app.test_client().get(
        "/api/portal/nope/triage").status_code == 404
