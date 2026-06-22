# tests/test_biofield_reveal_send.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load(mod):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"{mod} not importable: {e}")


def _app_db(monkeypatch, tmp_path):
    app_module = _load("app")
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens "
                   "(token_hash TEXT, email TEXT, purpose TEXT, extra TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return app_module, db


def test_send_reveal_link_mints_sends_marks(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "s@x.com", "2026-06-20", {"body": "x"},
                           [{"name": "Top", "slug": "top", "meaning": "m"}], "src")
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body: sent.append((to, body)) or True)
    ok = app_module._send_reveal_link(rid)
    assert ok is True and len(sent) == 1 and "/begin/biofield/" in sent[0][1]
    with sqlite3.connect(db) as cx:
        assert br.get(cx, rid)["notified_at"]
        n = cx.execute("SELECT COUNT(*) FROM auth_tokens WHERE email='s@x.com' AND purpose='biofield_reveal'").fetchone()[0]
    assert n == 1


def test_send_reveal_link_failed_send_not_marked(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "f@x.com", "2026-06-20", {"body": "x"}, [], "src")
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: False)
    ok = app_module._send_reveal_link(rid)
    assert ok is False
    with sqlite3.connect(db) as cx:
        assert not br.get(cx, rid)["notified_at"]


def test_ingest_notify_true_marks_notified(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.os.environ.get("CRON_SECRET") or app_module.CONSOLE_SECRET or ""
    if not key: pytest.skip("no secret")
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    c = app_module.app.test_client()
    c.post("/api/e4l/reveal-draft", headers={"X-Console-Key": key},
           json={"email": "n@x.com", "scan_date": "2026-06-20", "interpretation": {"body": "x"},
                 "layers": [{"n": 1, "title": "L", "summary": "s", "patterns": [], "remedy": None}]})
    c.post("/api/e4l/reveal-draft", headers={"X-Console-Key": key},
           json={"email": "q@x.com", "scan_date": "2026-06-20", "interpretation": {"body": "x"},
                 "layers": [{"n": 1, "title": "L", "summary": "s", "patterns": [], "remedy": None}],
                 "notify": False})
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rows = {r["email"]: r["notified_at"] for r in br.list_pending(cx)}
    assert rows.get("n@x.com")          # notify true -> marked
    assert not rows.get("q@x.com")      # notify false -> unmarked


def test_set_notified_and_list_approved_unnotified(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        r1, _ = br.upsert(cx, "a@x.com", "2026-06-20", {"body": "x"}, [], "s")  # approved, unnotified
        r2, _ = br.upsert(cx, "b@x.com", "2026-06-20", {"body": "x"}, [], "s")  # approved, notified
        r3, _ = br.upsert(cx, "c@x.com", "2026-06-20", {"body": "x"}, [], "s")  # not approved
        br.approve_first(cx, r1, "glen")
        br.approve_first(cx, r2, "glen")
        br.set_notified(cx, r2)
        ids = [r["id"] for r in br.list_approved_unnotified(cx)]
    assert r1 in ids and r2 not in ids and r3 not in ids
    with sqlite3.connect(db) as cx:
        row = br.get(cx, r2)
    assert row["notified_at"]


def _spine_db(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sek", raising=False)
    from dashboard import events as _ev
    with sqlite3.connect(db) as cx:
        _ev.init_event_tables(cx)
        cx.commit()
    return app_module, db


def test_send_action_approved_only(monkeypatch, tmp_path):
    app_module, db = _spine_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "a@x.com", "2026-06-20", {"body": "x"}, [], "s")  # not approved
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    c = app_module.app.test_client()
    r = c.post("/api/action/biofield_reveal.send", json={"id": rid}, headers={"X-Console-Key": "sek"})
    assert r.get_json()["result"]["sent"] is False           # unapproved -> not sent
    with sqlite3.connect(db) as cx:
        br.approve_first(cx, rid, "glen")
    r2 = c.post("/api/action/biofield_reveal.send", json={"id": rid}, headers={"X-Console-Key": "sek"})
    assert r2.get_json()["result"]["sent"] is True
    with sqlite3.connect(db) as cx:
        assert br.get(cx, rid)["notified_at"]


def test_console_page_ships_send_controls(monkeypatch, tmp_path):
    app_module, _ = _spine_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "sek", raising=False)
    html = app_module.app.test_client().get(
        "/console/biofield-reveals", headers={"X-Console-Key": "sek"}).data.decode()
    assert "biofield_reveal.send_all" in html
    assert "Send reveal link" in html


def test_send_all_batches_approved_unnotified(monkeypatch, tmp_path):
    app_module, db = _spine_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        a, _ = br.upsert(cx, "a@x.com", "2026-06-20", {"body": "x"}, [], "s")
        b, _ = br.upsert(cx, "b@x.com", "2026-06-20", {"body": "x"}, [], "s")
        c_, _ = br.upsert(cx, "c@x.com", "2026-06-20", {"body": "x"}, [], "s")  # stays unapproved
        br.approve_first(cx, a, "glen"); br.approve_first(cx, b, "glen")
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda to, *a, **k: sent.append(to) or True)
    r = app_module.app.test_client().post("/api/action/biofield_reveal.send_all",
                                          json={}, headers={"X-Console-Key": "sek"})
    res = r.get_json()["result"]
    assert res["sent"] == 2 and res["of"] == 2
    assert set(sent) == {"a@x.com", "b@x.com"}
