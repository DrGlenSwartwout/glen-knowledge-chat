# tests/test_document_extract_wiring.py
"""Wiring tests for the portal document-extraction pipeline: the inline
upload trigger, the console backfill route, and the console requeue route.

extract_document/run_pending themselves are already fully tested in
test_document_extract.py -- these tests only cover that app.py actually
calls them, at the right moments, without ever blocking or breaking the
upload response.
"""
import importlib, io, sqlite3, sys, time
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


def _upload(client, url, data=b"%PDF-1.4 fake", name="labs.pdf",
            ctype="application/pdf"):
    return client.post(url, data={"file": (io.BytesIO(data), name, ctype)},
                       content_type="multipart/form-data")


def _fake_call_model(payload):
    """A fake call_model matching document_extract's call signature
    (blob, content_type) -> dict, returning a fixed, well-formed payload."""
    def _call(blob, content_type):
        return payload
    return _call


def _wait_until(fn, timeout=2.0, interval=0.02):
    """Poll fn() until truthy or timeout. Used instead of a flat sleep so the
    test is deterministic and fast on a machine where the background thread
    finishes quickly, but still tolerant of a slow one."""
    deadline = time.time() + timeout
    result = fn()
    while not result and time.time() < deadline:
        time.sleep(interval)
        result = fn()
    return result


_GOOD_PAYLOAD = {
    "document_text": "Patient has elevated fasting glucose of 130 mg/dL noted.",
    "narrative_md": "Your recent labs showed a slightly elevated blood sugar.",
    "attributes": [],
    "facts": [],
    "unstructured": [],
}


def _doc_status(appmod, doc_id):
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    doc = cd.get(cx, doc_id)
    cx.close()
    return doc["extract_status"] if doc else None


# ---------------------------------------------------------------------------
# Inline trigger on upload
# ---------------------------------------------------------------------------

def test_upload_triggers_extraction_to_drafted(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_DOC_EXTRACT_CALL_MODEL",
                        _fake_call_model(_GOOD_PAYLOAD))
    tok = _token(appmod, "owner@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents")
    assert r.status_code == 200
    doc_id = r.get_json()["id"]
    ok = _wait_until(lambda: _doc_status(appmod, doc_id) == "drafted")
    assert ok, f"expected 'drafted', got {_doc_status(appmod, doc_id)!r}"
    from dashboard import document_extractions as dx
    cx = sqlite3.connect(appmod.LOG_DB)
    draft = dx.get_for_document(cx, doc_id)
    assert draft is not None
    assert draft["status"] == "ai_draft"
    assert "blood sugar" in draft["narrative_md"]


def test_deduped_reupload_does_not_trigger_a_second_extraction(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    calls = []

    def _counting_call(blob, content_type):
        calls.append(1)
        return _GOOD_PAYLOAD

    monkeypatch.setattr(appmod, "_DOC_EXTRACT_CALL_MODEL", _counting_call)
    tok = _token(appmod, "c@x.com")
    c = appmod.app.test_client()
    r1 = _upload(c, f"/api/portal/{tok}/documents")
    doc_id = r1.get_json()["id"]
    _wait_until(lambda: _doc_status(appmod, doc_id) == "drafted")
    assert len(calls) == 1

    r2 = _upload(c, f"/api/portal/{tok}/documents")
    assert r2.status_code == 200 and r2.get_json()["deduped"] is True
    # Give any (unwanted) second trigger a chance to run, then confirm it
    # never fired a second model call.
    time.sleep(0.3)
    assert len(calls) == 1


def test_unreadable_type_does_not_trigger_extraction(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    calls = []

    def _counting_call(blob, content_type):
        calls.append(1)
        return _GOOD_PAYLOAD

    monkeypatch.setattr(appmod, "_DOC_EXTRACT_CALL_MODEL", _counting_call)
    tok = _token(appmod, "c@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents",
               data=b"PK zip", name="records.zip", ctype="application/zip")
    assert r.status_code == 200
    doc_id = r.get_json()["id"]
    assert _doc_status(appmod, doc_id) == "skipped-unreadable"
    time.sleep(0.3)
    assert len(calls) == 0


def test_extraction_failure_does_not_break_upload_response(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)

    def _boom(blob, content_type):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(appmod, "_DOC_EXTRACT_CALL_MODEL", _boom)
    tok = _token(appmod, "c@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    doc_id = r.get_json()["id"]
    ok = _wait_until(lambda: _doc_status(appmod, doc_id) == "failed")
    assert ok, f"expected 'failed', got {_doc_status(appmod, doc_id)!r}"


# ---------------------------------------------------------------------------
# Console: run pending extractions (backstop)
# ---------------------------------------------------------------------------

def test_extract_pending_route_requires_console_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/documents/extract-pending")
    assert r.status_code == 401


def test_extract_pending_route_drafts_pending_documents(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    # No inline trigger for this test: leave the call_model seam unset (None)
    # would hit the real SDK, so point it at the fake -- but disable the
    # inline trigger's effect by asserting the backstop route ALSO drafts a
    # document that is still pending (e.g. inline trigger's thread hasn't
    # run yet is not testable deterministically, so instead: store a document
    # directly via client_documents.put + set pending, bypassing the route,
    # to exercise the backstop route in isolation).
    monkeypatch.setattr(appmod, "_DOC_EXTRACT_CALL_MODEL",
                        _fake_call_model(_GOOD_PAYLOAD))
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    res = cd.put(cx, "p@x.com", b"%PDF-1.4 fake", "labs.pdf", "application/pdf",
                 "console")
    cd.set_extract_status(cx, res["id"], "pending")
    cx.close()

    c = appmod.app.test_client()
    r = c.post("/api/console/documents/extract-pending?key=test-secret")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["drafted"] == 1
    assert _doc_status(appmod, res["id"]) == "drafted"


# ---------------------------------------------------------------------------
# Console: requeue
# ---------------------------------------------------------------------------

def _store_doc(appmod, email="r@x.com", status="pending"):
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    res = cd.put(cx, email, b"%PDF-1.4 fake", "labs.pdf", "application/pdf",
                 "console")
    cd.set_extract_status(cx, res["id"], status)
    cx.close()
    return res["id"]


def test_requeue_route_requires_console_key(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _store_doc(appmod, status="failed")
    c = appmod.app.test_client()
    r = c.post(f"/api/console/client-document/{doc_id}/requeue")
    assert r.status_code == 401


def test_requeue_resets_a_failed_document_to_pending(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _store_doc(appmod, status="failed")
    c = appmod.app.test_client()
    r = c.post(f"/api/console/client-document/{doc_id}/requeue?key=test-secret")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert _doc_status(appmod, doc_id) == "pending"


def test_requeue_recovers_a_document_stuck_in_extracting(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _store_doc(appmod, status="extracting")
    c = appmod.app.test_client()
    r = c.post(f"/api/console/client-document/{doc_id}/requeue?key=test-secret")
    assert r.status_code == 200
    assert _doc_status(appmod, doc_id) == "pending"


def test_requeue_refuses_a_confirmed_draft(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    doc_id = _store_doc(appmod, status="drafted")
    from dashboard import document_extractions as dx
    cx = sqlite3.connect(appmod.LOG_DB)
    ext_id = dx.put_draft(cx, doc_id, "r@x.com", "narrative", [], [], [], "m")
    dx.confirm(cx, ext_id, "narrative", "glen")
    cx.close()

    c = appmod.app.test_client()
    r = c.post(f"/api/console/client-document/{doc_id}/requeue?key=test-secret")
    assert r.status_code == 409
    assert r.get_json()["ok"] is False
    # Refused: status must be unchanged, still 'drafted', not reset to pending.
    assert _doc_status(appmod, doc_id) == "drafted"


def test_requeue_unknown_document_404s(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/console/client-document/999999/requeue?key=test-secret")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# CRITICAL 1: _db_lock must never be held across the (multi-second) model
# call. Deterministic, not timing-based: the injected call_model itself
# tries a NON-BLOCKING acquire of the real _db_lock -- that can only
# succeed if nothing else (i.e. the caller) is holding it at that moment.
# ---------------------------------------------------------------------------

def test_lock_not_held_during_model_call_extract_pending_route(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import client_documents as cd
    cx = sqlite3.connect(appmod.LOG_DB)
    res = cd.put(cx, "lock@x.com", b"%PDF-1.4 fake", "labs.pdf",
                "application/pdf", "console")
    cd.set_extract_status(cx, res["id"], "pending")
    cx.close()

    acquired = {}

    def _call(blob, content_type):
        got = appmod._db_lock.acquire(blocking=False)
        acquired["ok"] = got
        if got:
            appmod._db_lock.release()
        return _GOOD_PAYLOAD

    monkeypatch.setattr(appmod, "_DOC_EXTRACT_CALL_MODEL", _call)
    c = appmod.app.test_client()
    r = c.post("/api/console/documents/extract-pending?key=test-secret")
    assert r.status_code == 200
    assert r.get_json()["drafted"] == 1
    assert acquired.get("ok") is True, (
        "call_model could not acquire _db_lock -- it was held during the "
        "model call")


def test_lock_not_held_during_model_call_inline_trigger(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    acquired = {}

    def _call(blob, content_type):
        got = appmod._db_lock.acquire(blocking=False)
        acquired["ok"] = got
        if got:
            appmod._db_lock.release()
        return _GOOD_PAYLOAD

    monkeypatch.setattr(appmod, "_DOC_EXTRACT_CALL_MODEL", _call)
    tok = _token(appmod, "lock2@x.com")
    r = _upload(appmod.app.test_client(), f"/api/portal/{tok}/documents")
    doc_id = r.get_json()["id"]
    ok = _wait_until(lambda: _doc_status(appmod, doc_id) == "drafted")
    assert ok, f"expected 'drafted', got {_doc_status(appmod, doc_id)!r}"
    assert acquired.get("ok") is True, (
        "call_model could not acquire _db_lock -- it was held during the "
        "model call")
