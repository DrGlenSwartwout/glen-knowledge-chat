# tests/test_begin_entry_points.py
"""Begin #3 - entry points into the one record + predicates + PB wiring."""
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS inbound_leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT, source TEXT, email TEXT,
                first_name TEXT, last_name TEXT, phone TEXT,
                raw_json TEXT, ghl_contact_id TEXT,
                ghl_opp_id TEXT, ghl_error TEXT,
                last_outbound_at TEXT, tags TEXT, status TEXT
            )
        """)
        cx.commit()
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_log_inbound_lead", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_attribute_conversion_by_email", lambda *a, **k: None)
    return db


def test_record_entry_unlock_writes_quiz_by_email(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    app_module._record_entry_unlock("quiz", "Ann@Example.com", first_name="Ann")
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="ann@example.com")
    assert "quiz" in st["unlocked_gates"]
    assert st["first_name"] == "Ann"


def test_record_entry_unlock_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    app_module._record_entry_unlock("scan", "b@x.com")
    app_module._record_entry_unlock("scan", "b@x.com")
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM journey_events WHERE trigger='scan' AND email='b@x.com'").fetchone()[0]
    assert n == 1


def test_entry_unions_with_real_session(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.record_unlock(cx, session_id="sessA", trigger="name",
                                   email="c@x.com", first_name="Cee")
    app_module._record_entry_unlock("scan", "c@x.com")
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, session_id="sessA", email="c@x.com")
    assert "scan" in st["unlocked_gates"]


def test_record_entry_unlock_never_raises(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    import begin_funnel
    monkeypatch.setattr(begin_funnel, "record_unlock",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    app_module._record_entry_unlock("scan", "d@x.com")  # must not raise


def test_e4l_freshness_ingest_records_scan(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")   # endpoint reads CRON_SECRET first
    client = app_module.app.test_client()
    r = client.post("/api/e4l/scan-freshness", json={"rows": [{"email": "e@x.com", "last_scan_date": "2026-06-19"}]},
                    headers={"X-Cron-Secret": "k"})
    assert r.status_code == 200
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="e@x.com")
    assert "scan" in st["unlocked_gates"]


def test_state_predicates_light_give(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS affiliate_signups (email TEXT, slug TEXT, status TEXT)")
        cx.execute("INSERT INTO affiliate_signups (email, slug, status) VALUES (?,?,?)",
                   ("amb@x.com", "amb", "approved"))
        cx.commit()
    client = app_module.app.test_client(); client.set_cookie("amg_session", "s1")
    # activate so email is on the session row, then read state with that email
    app_module._record_entry_unlock("quiz", "amb@x.com", first_name="Amb")
    with sqlite3.connect(db) as cx:
        import begin_funnel
        begin_funnel.record_unlock(cx, session_id="s1", trigger="tos",
                                   email="amb@x.com", tos=True)
    body = client.get("/begin/state").get_json()
    give = [c for c in body["journey_map"] if c["key"] == "give"][0]
    assert give["steps"][0]["done"] is True


def test_state_scan_gate_routes_to_portal(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client(); client.set_cookie("amg_session", "s2")
    app_module._record_entry_unlock("scan", "p@x.com")
    with sqlite3.connect(db) as cx:
        import begin_funnel
        begin_funnel.record_unlock(cx, session_id="s2", trigger="tos",
                                   email="p@x.com", tos=True)
    body = client.get("/begin/state").get_json()
    scan = [c for c in body["journey_map"] if c["key"] == "scan"][0]
    assert scan["href"].startswith("https://portal.e4l.com")


def test_pb_completion_sets_gate(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "WEBHOOK_SECRET", "", raising=False)
    client = app_module.app.test_client()
    r = client.post("/webhook/practice-better",
                    json={"event_type": "wellness-whispering.completed", "email": "g@x.com", "name": "Gee Aitch"})
    assert r.status_code == 200
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="g@x.com")
    assert "course_ww" in st["unlocked_gates"]


def test_pb_unmapped_event_sets_no_gate(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "WEBHOOK_SECRET", "", raising=False)
    client = app_module.app.test_client()
    r = client.post("/webhook/practice-better",
                    json={"event_type": "client.created", "email": "h@x.com", "name": "H"})
    assert r.status_code == 200
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="h@x.com")
    assert "course_ww" not in st["unlocked_gates"]
