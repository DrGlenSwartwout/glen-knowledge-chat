# tests/test_document_upload_routes.py
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
    # These tests verify the SYNCHRONOUS upload/storage contract (validation,
    # scoping, dedup, the stored extract_status). The upload also fires a
    # fire-and-forget daemon thread that immediately moves an extractable doc
    # pending -> extracting -> failed, which makes 'pending' a transient state
    # and races any assertion on it. Stub the trigger so the stored status is
    # deterministic; the trigger's real behavior is covered by
    # tests/test_document_extract_wiring.py.
    monkeypatch.setattr(appmod, "_trigger_document_extraction", lambda *a, **k: None)
    return appmod


def _token(appmod, email):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    tok = cp.ensure_token(cx, email, "T")
    cx.commit(); cx.close()
    return tok


def _upload(client, url, data=b"%PDF-1.4 fake", name="labs.pdf",
            ctype="application/pdf"):
    return client.post(url, data={"file": (io.BytesIO(data), name, ctype)},
                       content_type="multipart/form-data")


def test_portal_upload_stores_against_the_token_owner(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "owner@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents")
    assert r.status_code == 200 and r.get_json()["ok"] is True
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    rows = cd.list_for_email(cx, "owner@x.com")
    assert len(rows) == 1
    assert rows[0]["source"] == "portal-self"
    assert rows[0]["extract_status"] == "pending"


def test_portal_upload_cannot_reach_another_clients_documents(tmp_path, monkeypatch):
    """Token scoping: a token writes ONLY its own owner's email."""
    appmod = _app(tmp_path, monkeypatch)
    _token(appmod, "other@x.com")
    tok_a = _token(appmod, "a@x.com")
    _upload(appmod.app.test_client(), f"/api/portal/{tok_a}/documents")
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    assert len(cd.list_for_email(cx, "a@x.com")) == 1
    assert cd.list_for_email(cx, "other@x.com") == []


def test_portal_upload_unknown_token_404s(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    r = _upload(appmod.app.test_client(), "/api/portal/nope/documents")
    assert r.status_code == 404


def test_upload_rejects_empty_file(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents", data=b"")
    assert r.status_code == 400


def test_upload_rejects_over_cap(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    big = b"x" * (30 * 1024 * 1024 + 1)
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents", data=big)
    assert r.status_code == 400
    assert "too large" in r.get_json()["error"]


def test_unreadable_type_is_stored_but_marked_skipped(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents",
                data=b"PK zip", name="records.zip", ctype="application/zip")
    assert r.status_code == 200
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    rows = cd.list_for_email(cx, "c@x.com")
    assert rows[0]["extract_status"] == "skipped-unreadable"


def test_image_type_is_extractable(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents",
            data=b"\xff\xd8\xff", name="record.jpg", ctype="image/jpeg")
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cd.list_for_email(cx, "c@x.com")[0]["extract_status"] == "pending"


def test_reupload_of_identical_bytes_is_deduped(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _token(appmod, "c@x.com")
    c = appmod.app.test_client()
    _upload(c, f"/api/portal/{tok}/documents")
    r = _upload(c, f"/api/portal/{tok}/documents")
    assert r.status_code == 200 and r.get_json()["deduped"] is True
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    assert len(cd.list_for_email(cx, "c@x.com")) == 1


def test_console_upload_requires_the_console_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-document",
               data={"email": "c@x.com",
                     "file": (io.BytesIO(b"%PDF"), "a.pdf", "application/pdf")},
               content_type="multipart/form-data")
    assert r.status_code == 401


def test_console_upload_with_key_stores_with_console_source(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-document?key=test-secret",
               data={"email": "C@x.com",
                     "file": (io.BytesIO(b"%PDF"), "a.pdf", "application/pdf")},
               content_type="multipart/form-data")
    assert r.status_code == 200
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cd.list_for_email(cx, "c@x.com")[0]["source"] == "console"


def test_console_upload_requires_an_email(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-document?key=test-secret",
               data={"file": (io.BytesIO(b"%PDF"), "a.pdf", "application/pdf")},
               content_type="multipart/form-data")
    assert r.status_code == 400
