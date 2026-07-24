import importlib, sqlite3, sys
from pathlib import Path
from dashboard import client_documents as cd
from dashboard import document_extractions as dx


def _app(tmp_path, monkeypatch, hub="1", console_secret=None):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PORTAL_HUB_ENABLED", hub)
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    if console_secret:
        monkeypatch.setenv("CONSOLE_SECRET", console_secret)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed(appmod, email, blob=b"%PDF bytes", confirm=False, source="portal-self"):
    """Defaults to source='portal-self' -- a client's own upload, visible to
    them immediately -- so the pre-existing tests below (which only care
    about the owner-vs-other-token isolation, not the console visibility
    gate) keep exercising a document the owning token can actually see.
    Tests that specifically cover the console staff-only gate pass
    source='console' explicitly."""
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    doc_id = cd.put(cx, email, blob, "labs.pdf", "application/pdf", source)["id"]
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


def _seed_with(appmod, email, filename, content_type, blob=b"bytes", source="portal-self"):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    doc_id = cd.put(cx, email, blob, filename, content_type, source)["id"]
    cx.commit(); cx.close()
    return tok, doc_id


def test_crlf_in_filename_downloads_without_500(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed_with(
        appmod, "c@x.com", "evil\r\nX-Injected: yes.pdf", "application/pdf")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert r.status_code == 200
    cd_header = r.headers["Content-Disposition"]
    assert "\r" not in cd_header and "\n" not in cd_header
    assert "X-Injected" not in dict(r.headers)


def test_html_content_type_served_as_attachment_octet_stream(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed_with(appmod, "c@x.com", "page.html", "text/html")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert r.status_code == 200
    assert r.mimetype == "application/octet-stream"
    assert r.headers["Content-Disposition"].startswith("attachment")


def test_svg_content_type_not_served_inline(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed_with(appmod, "c@x.com", "pic.svg", "image/svg+xml")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert r.status_code == 200
    assert r.mimetype == "application/octet-stream"
    assert r.headers["Content-Disposition"].startswith("attachment")


def test_pdf_still_served_inline_with_own_type_and_unchanged_bytes(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed_with(
        appmod, "c@x.com", "labs.pdf", "application/pdf", blob=b"%PDF-1.4 real bytes")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert r.status_code == 200
    assert r.mimetype == "application/pdf"
    assert r.headers["Content-Disposition"].startswith("inline")
    assert r.data == b"%PDF-1.4 real bytes"


def test_nosniff_header_present(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed(appmod, "c@x.com")
    r = appmod.app.test_client().get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert r.headers["X-Content-Type-Options"] == "nosniff"


# ── client_visible gate (Gap 1) ─────────────────────────────────────────────

def test_console_uploaded_document_hidden_from_portal_and_file_404s(tmp_path, monkeypatch):
    """A record Glen (or staff) uploads on the console stays staff-only until
    he explicitly marks it visible -- it must not appear in the owner's own
    portal list, and its file route must 404 exactly like a nonexistent doc."""
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed(appmod, "c@x.com", source="console")
    items = appmod.app.test_client().get(f"/api/portal/{tok}/documents").get_json()["items"]
    assert items == []
    r = appmod.app.test_client().get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert r.status_code == 404


def test_client_self_uploaded_document_appears_and_downloads(tmp_path, monkeypatch):
    """A record the CLIENT uploads themselves is their own file and is
    visible/downloadable immediately -- no console action required."""
    appmod = _app(tmp_path, monkeypatch)
    tok, doc_id = _seed(appmod, "c@x.com", source="portal-self",
                        blob=b"%PDF-1.4 mine")
    items = appmod.app.test_client().get(f"/api/portal/{tok}/documents").get_json()["items"]
    assert len(items) == 1 and items[0]["id"] == doc_id
    r = appmod.app.test_client().get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert r.status_code == 200
    assert r.data == b"%PDF-1.4 mine"


def test_visibility_route_makes_console_document_visible_and_downloadable(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, console_secret="s3cret")
    tok, doc_id = _seed(appmod, "c@x.com", source="console", blob=b"%PDF-1.4 staff")
    client = appmod.app.test_client()
    r = client.post(f"/api/console/client-document/{doc_id}/visibility",
                    json={"visible": True}, headers={"X-Console-Key": "s3cret"})
    assert r.status_code == 200
    items = client.get(f"/api/portal/{tok}/documents").get_json()["items"]
    assert len(items) == 1 and items[0]["id"] == doc_id
    fr = client.get(f"/api/portal/{tok}/documents/{doc_id}/file")
    assert fr.status_code == 200
    assert fr.data == b"%PDF-1.4 staff"


def test_visibility_route_requires_console_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, console_secret="s3cret")
    _, doc_id = _seed(appmod, "c@x.com", source="console")
    r = appmod.app.test_client().post(
        f"/api/console/client-document/{doc_id}/visibility", json={"visible": True})
    assert r.status_code == 401


def test_console_review_payload_still_shows_non_visible_documents(tmp_path, monkeypatch):
    """The console screen is exactly where Glen decides visibility -- it must
    keep showing every document regardless of client_visible."""
    appmod = _app(tmp_path, monkeypatch, console_secret="s3cret")
    _, doc_id = _seed(appmod, "c@x.com", source="console")
    r = appmod.app.test_client().get(
        "/api/console/client-documents?email=c@x.com",
        headers={"X-Console-Key": "s3cret"})
    items = r.get_json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == doc_id
    assert items[0]["client_visible"] is False
