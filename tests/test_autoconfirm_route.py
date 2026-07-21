import json, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest
try:
    import app, dashboard
    from dashboard import client_portal as _cp
except Exception as e:
    pytest.skip(f"app import needs secrets: {e}", allow_module_level=True)

def _auth(mp, tmp):
    db = str(tmp / "chat_log.db"); mp.setenv("DATA_DIR", str(tmp))
    mp.setattr(app, "LOG_DB", db, raising=False)
    for obj in (app, dashboard):
        mp.setattr(obj, "CONSOLE_SECRET", "sek", raising=False)
    mp.setattr(app, "ANALYSIS_AUTOCONFIRM_ENABLED", True, raising=False)
    mp.setattr(app, "ANALYSIS_AUTOCONFIRM_SAMPLE_PCT", "0", raising=False)  # no sampling in test
    cx = sqlite3.connect(db); _cp.init_client_portal_table(cx); cx.close()
    return db

def _publish_draft(payload):
    # /admin/portal/upsert (admin_client_portal_upsert) is the actual hand-off
    # bridge that writes a fresh ai_draft (it carries the never-un-publish guard
    # this task's hook sits after). /api/console/biofield-portal is the OPERATOR
    # "Publish" action, which always force-confirms regardless of draft status,
    # so it can't exercise the auto-confirm hook.
    return app.app.test_client().post("/admin/portal/upsert",
        headers={"X-Console-Key": "sek", "Content-Type": "application/json"},
        data=json.dumps(payload))

def _clean_draft_body(email):
    return {"email": email, "name": "T", "scan_date": "2026-07-01",
            "content": {"biofield_status": "ai_draft", "greeting": "Aloha.",
                        "layers": [{"title": "Cellular", "remedy": "Vitality",
                                    "dosage": "1 cap", "frequency": "daily"}]}}

def test_clean_ai_draft_auto_confirms(monkeypatch, tmp_path):
    # Force resolver to accept any remedy so the test doesn't depend on catalog data.
    from dashboard import biofield_portal_publish as bpp
    monkeypatch.setattr(bpp, "resolve_remedy_slug", lambda n, c: "slug-x")
    db = _auth(monkeypatch, tmp_path)
    _publish_draft(_clean_draft_body("free@x.com"))
    cx = sqlite3.connect(db)
    st = (json.loads(cx.execute("SELECT content_json FROM client_portals WHERE email='free@x.com'")
          .fetchone()[0]) or {}).get("biofield_status")
    assert st == "confirmed"   # auto-confirmed

def test_low_quality_draft_stays_ai_draft(monkeypatch, tmp_path):
    from dashboard import biofield_portal_publish as bpp
    monkeypatch.setattr(bpp, "resolve_remedy_slug", lambda n, c: None)  # nothing resolves → held
    db = _auth(monkeypatch, tmp_path)
    _publish_draft(_clean_draft_body("held@x.com"))
    cx = sqlite3.connect(db)
    st = (json.loads(cx.execute("SELECT content_json FROM client_portals WHERE email='held@x.com'")
          .fetchone()[0]) or {}).get("biofield_status")
    assert st == "ai_draft"    # held for human review

def test_backfill_dryrun_counts_without_confirming(monkeypatch, tmp_path):
    from dashboard import biofield_portal_publish as bpp
    monkeypatch.setattr(bpp, "resolve_remedy_slug", lambda n, c: "slug-x")
    monkeypatch.setattr(app, "ANALYSIS_AUTOCONFIRM_ENABLED", True, raising=False)
    monkeypatch.setattr(app, "ANALYSIS_AUTOCONFIRM_SAMPLE_PCT", "0", raising=False)
    db = _auth(monkeypatch, tmp_path)
    # seed one clean ai_draft directly
    cx = sqlite3.connect(db)
    _cp.upsert_portal(cx, "b@x.com", "B", {"biofield_status": "ai_draft",
        "layers": [{"title": "C", "remedy": "V", "dosage": "1"}], "greeting": "hi"})
    cx.close()
    r = app.app.test_client().post("/api/console/autoconfirm/backfill",
        headers={"X-Console-Key": "sek", "Content-Type": "application/json"},
        data=json.dumps({"commit": False}))
    j = r.get_json()
    assert j["ok"] and j["would_confirm"] == 1
    cx = sqlite3.connect(db)
    st = (json.loads(cx.execute("SELECT content_json FROM client_portals WHERE email='b@x.com'")
          .fetchone()[0]) or {}).get("biofield_status")
    assert st == "ai_draft"   # dry run did NOT change anything

def test_backfill_commit_confirms_only_clean(monkeypatch, tmp_path):
    from dashboard import biofield_portal_publish as bpp
    # Monkeypatch resolver: "Vitality" resolves, "Bogus" does not
    monkeypatch.setattr(bpp, "resolve_remedy_slug",
                        lambda n, c: "slug-x" if n == "Vitality" else None)
    monkeypatch.setattr(app, "ANALYSIS_AUTOCONFIRM_SAMPLE_PCT", "0", raising=False)
    db = _auth(monkeypatch, tmp_path)
    # Seed TWO ai_draft portals directly
    cx = sqlite3.connect(db)
    _cp.upsert_portal(cx, "clean@x.com", "Clean", {"biofield_status": "ai_draft",
        "greeting": "hi", "layers": [{"title": "C", "remedy": "Vitality", "dosage": "1 cap"}]})
    _cp.upsert_portal(cx, "held@x.com", "Held", {"biofield_status": "ai_draft",
        "greeting": "hi", "layers": [{"title": "C", "remedy": "Bogus", "dosage": "1 cap"}]})
    cx.close()
    # POST with commit=true to actually confirm
    r = app.app.test_client().post("/api/console/autoconfirm/backfill",
        headers={"X-Console-Key": "sek", "Content-Type": "application/json"},
        data=json.dumps({"commit": True}))
    j = r.get_json()
    assert j["ok"] and j["would_confirm"] == 1
    # Re-open db and verify only clean was confirmed
    cx = sqlite3.connect(db)
    clean_st = (json.loads(cx.execute("SELECT content_json FROM client_portals WHERE email='clean@x.com'")
          .fetchone()[0]) or {}).get("biofield_status")
    held_st = (json.loads(cx.execute("SELECT content_json FROM client_portals WHERE email='held@x.com'")
          .fetchone()[0]) or {}).get("biofield_status")
    assert clean_st == "confirmed"  # was confirmed by backfill
    assert held_st == "ai_draft"    # held because remedy unresolvable
    # Verify audit log recorded the confirm
    log_rec = cx.execute("SELECT decision FROM analysis_autoconfirm_log WHERE email='clean@x.com'").fetchone()
    assert log_rec and log_rec[0] == "confirmed"
    cx.close()
