import importlib, sqlite3, sys
from pathlib import Path
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
