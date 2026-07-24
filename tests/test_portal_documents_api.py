import importlib, sqlite3, sys
from pathlib import Path
from dashboard import client_documents as cd
from dashboard import document_extractions as dx


def _app(tmp_path, monkeypatch, hub="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PORTAL_HUB_ENABLED", hub)
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed(appmod, email, blob=b"%PDF bytes", confirm=False):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    doc_id = cd.put(cx, email, blob, "labs.pdf", "application/pdf", "console")["id"]
    eid = dx.put_draft(cx, doc_id, email, "Your summary.", [], [], [], "m")
    if confirm:
        dx.confirm(cx, eid, "Your summary.", "glen")
    cx.commit(); cx.close()
    return tok, doc_id


def test_before_approval_status_is_under_review_and_no_narrative(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed(appmod, "c@x.com")
    body = appmod.app.test_client().get(f"/api/portal/{tok}/documents").get_json()
    assert body["enabled"] is True
    it = body["items"][0]
    assert it["status"] == "under_review"
    assert it["narrative_md"] == ""
    assert it["file_url"] == f"/api/portal/{tok}/documents/{doc_id}/file"


def test_after_approval_narrative_is_visible(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, _ = _seed(appmod, "c@x.com", confirm=True)
    it = appmod.app.test_client().get(
        f"/api/portal/{tok}/documents").get_json()["items"][0]
    assert it["status"] == "ready"
    assert it["narrative_md"] == "Your summary."


def test_payload_never_exposes_attributes_or_facts(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed(appmod, "c@x.com", confirm=True)
    cx = sqlite3.connect(appmod.LOG_DB)
    dx.put_draft(cx, doc_id, "c@x.com", "n",
                 [{"field": "conditions", "value": "Glaucoma", "source_quote": "q"}],
                 [{"fact_key": "on_areds2", "value": True, "source_quote": "q"}],
                 [{"label": "HbA1c", "value": "6.4", "source_quote": "q"}], "m")
    cx.commit(); cx.close()
    raw = appmod.app.test_client().get(f"/api/portal/{tok}/documents").get_data(as_text=True)
    assert "Glaucoma" not in raw and "on_areds2" not in raw and "HbA1c" not in raw


def test_owner_can_download_their_own_raw_file(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed(appmod, "c@x.com", blob=b"%PDF-1.4 real bytes")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert r.status_code == 200
    assert r.data == b"%PDF-1.4 real bytes"
    assert r.headers["Cache-Control"] == "private, no-store"


def test_another_token_cannot_download_the_file(tmp_path, monkeypatch):
    """Cross-token isolation — the test that matters most."""
    appmod = _app(tmp_path, monkeypatch)
    _, doc_id = _seed(appmod, "owner@x.com")
    other_tok, _ = _seed(appmod, "other@x.com", blob=b"other bytes")
    r = appmod.app.test_client().get(
        f"/api/portal/{other_tok}/documents/{doc_id}/file")
    assert r.status_code == 404


def test_another_token_does_not_see_the_document_in_its_list(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    _seed(appmod, "owner@x.com")
    other_tok, _ = _seed(appmod, "other@x.com", blob=b"other bytes")
    items = appmod.app.test_client().get(
        f"/api/portal/{other_tok}/documents").get_json()["items"]
    assert len(items) == 1


def test_enabled_false_when_hub_flag_off(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, hub="0")
    tok, _ = _seed(appmod, "c@x.com")
    assert appmod.app.test_client().get(
        f"/api/portal/{tok}/documents").get_json()["enabled"] is False


def test_unknown_token_404s(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert appmod.app.test_client().get("/api/portal/nope/documents").status_code == 404
