import importlib, sqlite3, sys
from pathlib import Path
import pytest
from dashboard import client_documents as cd
from dashboard import document_extractions as dx


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


def _seed(appmod, email="c@x.com"):
    cx = sqlite3.connect(appmod.LOG_DB)
    doc_id = cd.put(cx, email, b"%PDF", "labs.pdf", "application/pdf", "console")["id"]
    dx.put_draft(cx, doc_id, email, "Draft narrative.",
                 attributes=[{"field": "conditions", "value": "Glaucoma",
                              "source_quote": "q"},
                             {"field": "body_systems", "value": "Liver",
                              "source_quote": "q"}],
                 facts=[{"fact_key": "on_areds2", "value": True,
                         "source_quote": "q"}],
                 unstructured=[{"label": "HbA1c", "value": "6.4",
                                "source_quote": "q"}],
                 model="m")
    cx.commit(); cx.close()
    return doc_id


def _approve(appmod, doc_id, **body):
    payload = {"narrative_md": "Final narrative.", "attributes": [0, 1],
               "facts": [0], "reviewed_by": "glen"}
    payload.update(body)
    return appmod.app.test_client().post(
        f"/api/console/client-document/{doc_id}/approve?key=test-secret",
        json=payload)


def test_approve_writes_checked_attributes_with_document_provenance(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    assert _approve(appmod, doc_id).status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB)
    rows = cx.execute("SELECT field, value, source FROM person_attributes "
                      "WHERE email='c@x.com' ORDER BY field").fetchall()
    assert ("body_systems", "Liver", f"document:{doc_id}") in rows
    assert ("conditions", "Glaucoma", f"document:{doc_id}") in rows


def test_approve_writes_checked_boolean_facts(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    _approve(appmod, doc_id)
    from dashboard import client_facts as cf
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    assert cf.get_facts(cx, "c@x.com")["on_areds2"] is True


def test_approve_does_not_write_client_conditions(tmp_path, monkeypatch):
    """Guards the single eye-condition support-program override from being
    clobbered by extracted diagnoses."""
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    from dashboard import client_conditions as cc
    cx = sqlite3.connect(appmod.LOG_DB)
    cc.init_table(cx)
    cc.set(cx, "c@x.com", "glaucoma-support", "operator")
    cx.commit(); cx.close()
    _approve(appmod, doc_id)
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cc.get(cx, "c@x.com") == "glaucoma-support"


def test_unchecked_items_are_not_written(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    _approve(appmod, doc_id, attributes=[0], facts=[])
    cx = sqlite3.connect(appmod.LOG_DB)
    fields = [r[0] for r in cx.execute(
        "SELECT field FROM person_attributes WHERE email='c@x.com'").fetchall()]
    assert fields == ["conditions"]
    n = cx.execute("SELECT COUNT(*) FROM client_facts "
                   "WHERE email='c@x.com'").fetchone()[0]
    assert n == 0


def test_approve_saves_the_edited_narrative_and_confirms(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    _approve(appmod, doc_id, narrative_md="Glen's edited words.")
    cx = sqlite3.connect(appmod.LOG_DB)
    got = dx.get_for_document(cx, doc_id)
    assert got["status"] == "confirmed"
    assert got["narrative_md"] == "Glen's edited words."
    assert got["reviewed_by"] == "glen"


def test_second_approve_is_an_idempotent_noop(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    assert _approve(appmod, doc_id).status_code == 200
    r2 = _approve(appmod, doc_id, narrative_md="second")
    assert r2.status_code == 200 and r2.get_json()["already"] is True
    cx = sqlite3.connect(appmod.LOG_DB)
    assert dx.get_for_document(cx, doc_id)["narrative_md"] == "Final narrative."


def test_reject_discards_the_draft_and_keeps_the_file(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = appmod.app.test_client().post(
        f"/api/console/client-document/{doc_id}/reject?key=test-secret",
        json={"reviewed_by": "glen"})
    assert r.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB)
    assert dx.get_for_document(cx, doc_id)["status"] == "rejected"
    assert cd.get(cx, doc_id) is not None
    n = cx.execute("SELECT COUNT(*) FROM person_attributes").fetchone()[0]
    assert n == 0


def test_approve_requires_the_console_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = appmod.app.test_client().post(
        f"/api/console/client-document/{doc_id}/approve", json={})
    assert r.status_code == 401


def test_approve_404s_for_a_document_with_no_draft(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    assert _approve(appmod, 4242).status_code == 404


def test_console_review_payload_carries_quotes_and_file_url(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = appmod.app.test_client().get(
        "/api/console/client-documents?email=c@x.com&key=test-secret")
    assert r.status_code == 200
    it = r.get_json()["items"][0]
    assert it["id"] == doc_id
    assert it["filename"] == "labs.pdf"
    assert it["file_url"] == f"/admin/client-document?id={doc_id}"
    d = it["draft"]
    assert d["narrative_md"] == "Draft narrative."
    assert d["attributes"][0]["source_quote"] == "q"
    assert d["facts"][0]["fact_key"] == "on_areds2"
    assert d["unstructured"][0]["label"] == "HbA1c"


def test_console_review_payload_draft_is_null_when_not_extracted(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB)
    cd.put(cx, "c@x.com", b"%PDF", "raw.pdf", "application/pdf", "console")
    cx.commit(); cx.close()
    it = appmod.app.test_client().get(
        "/api/console/client-documents?email=c@x.com&key=test-secret"
    ).get_json()["items"][0]
    assert it["draft"] is None


def test_console_review_requires_the_console_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    r = appmod.app.test_client().get("/api/console/client-documents?email=c@x.com")
    assert r.status_code == 401


def test_console_document_file_serves_bytes_with_the_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = appmod.app.test_client().get(f"/admin/client-document?id={doc_id}&key=test-secret")
    assert r.status_code == 200 and r.data == b"%PDF"


def test_console_document_file_requires_the_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    assert appmod.app.test_client().get(
        f"/admin/client-document?id={doc_id}").status_code == 401


def test_console_document_html_content_type_served_as_attachment_octet_stream(
        tmp_path, monkeypatch):
    """Hardening: the console viewer must reuse the Task 6 allowlist, not the
    stored (attacker-influenced) content_type, else a text/html document would
    render inline at the app's origin with console privileges."""
    appmod = _app(tmp_path, monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB)
    doc_id = cd.put(cx, "c@x.com", b"<script>1</script>", "page.html",
                    "text/html", "console")["id"]
    cx.commit(); cx.close()
    r = appmod.app.test_client().get(
        f"/admin/client-document?id={doc_id}&key=test-secret")
    assert r.status_code == 200
    assert r.mimetype == "application/octet-stream"
    assert r.headers["Content-Disposition"].startswith("attachment")
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_console_document_file_nosniff_header_present(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = appmod.app.test_client().get(
        f"/admin/client-document?id={doc_id}&key=test-secret")
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_out_of_vocabulary_fact_key_is_skipped_not_written(tmp_path, monkeypatch):
    """A model-proposed fact_key outside ALLOWED_CLIENT_FACT_KEYS must never
    reach client_facts -- but the rest of the approval (attributes, confirm)
    still succeeds."""
    appmod = _app(tmp_path, monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB)
    doc_id = cd.put(cx, "c@x.com", b"%PDF", "labs.pdf", "application/pdf",
                    "console")["id"]
    dx.put_draft(cx, doc_id, "c@x.com", "Draft narrative.",
                 attributes=[{"field": "conditions", "value": "Glaucoma",
                              "source_quote": "q"}],
                 facts=[{"fact_key": "has_melanoma", "value": True,
                         "source_quote": "q"}],
                 unstructured=[], model="m")
    cx.commit(); cx.close()
    r = _approve(appmod, doc_id, attributes=[0], facts=[0])
    assert r.status_code == 200
    got = r.get_json()
    assert got["written"]["facts"] == 0
    assert got["written"]["attributes"] == 1
    assert got["skipped_facts"] == 1
    from dashboard import client_facts as cf
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    assert cf.get_facts(cx, "c@x.com") == {}
    assert dx.get_for_document(cx, doc_id)["status"] == "confirmed"


def test_missing_fact_key_is_skipped_not_500(tmp_path, monkeypatch):
    """A fact item with no fact_key must be skipped, never reach set_fact
    (which would raise IntegrityError on the NOT NULL column)."""
    appmod = _app(tmp_path, monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB)
    doc_id = cd.put(cx, "c@x.com", b"%PDF", "labs.pdf", "application/pdf",
                    "console")["id"]
    dx.put_draft(cx, doc_id, "c@x.com", "Draft narrative.",
                 attributes=[], facts=[{"value": True, "source_quote": "q"}],
                 unstructured=[], model="m")
    cx.commit(); cx.close()
    r = _approve(appmod, doc_id, attributes=[], facts=[0])
    assert r.status_code == 200
    got = r.get_json()
    assert got["written"]["facts"] == 0
    assert got["skipped_facts"] == 1
    cx = sqlite3.connect(appmod.LOG_DB)
    assert dx.get_for_document(cx, doc_id)["status"] == "confirmed"


def test_a_failed_write_leaves_the_draft_ai_draft_and_narrative_not_live(
        tmp_path, monkeypatch):
    """Reorder-under-test: if a live write raises partway through, the draft
    must still be ai_draft (not confirmed) -- proving writes run BEFORE
    confirm()."""
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    from dashboard import canonical_tags as ct

    def _boom(cx, email, field, value, *, source):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(ct, "set_attr", _boom)
    appmod.app.testing = True
    with pytest.raises(RuntimeError):
        _approve(appmod, doc_id)
    cx = sqlite3.connect(appmod.LOG_DB)
    got = dx.get_for_document(cx, doc_id)
    assert got["status"] == "ai_draft"
    assert got["narrative_md"] == "Draft narrative."


def test_retry_after_a_failed_write_completes_successfully(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    from dashboard import canonical_tags as ct
    real_set_attr = ct.set_attr
    calls = {"n": 0}

    def _boom_once(cx, email, field, value, *, source):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated write failure")
        return real_set_attr(cx, email, field, value, source=source)

    monkeypatch.setattr(ct, "set_attr", _boom_once)
    appmod.app.testing = True
    with pytest.raises(RuntimeError):
        _approve(appmod, doc_id)
    cx = sqlite3.connect(appmod.LOG_DB)
    assert dx.get_for_document(cx, doc_id)["status"] == "ai_draft"

    monkeypatch.setattr(ct, "set_attr", real_set_attr)
    r = _approve(appmod, doc_id)
    assert r.status_code == 200
    got = r.get_json()
    assert got["written"]["attributes"] == 2
    assert got["written"]["facts"] == 1
    cx = sqlite3.connect(appmod.LOG_DB)
    d = dx.get_for_document(cx, doc_id)
    assert d["status"] == "confirmed"
    rows = cx.execute("SELECT field FROM person_attributes "
                      "WHERE email='c@x.com' ORDER BY field").fetchall()
    assert [r[0] for r in rows] == ["body_systems", "conditions"]
    from dashboard import client_facts as cf
    cx.row_factory = sqlite3.Row
    assert cf.get_facts(cx, "c@x.com")["on_areds2"] is True


def test_in_vocabulary_fact_still_writes_exactly_as_before(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _seed(appmod)
    r = _approve(appmod, doc_id)
    assert r.status_code == 200
    got = r.get_json()
    assert got["written"]["facts"] == 1
    assert got["skipped_facts"] == 0
    from dashboard import client_facts as cf
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    assert cf.get_facts(cx, "c@x.com")["on_areds2"] is True
