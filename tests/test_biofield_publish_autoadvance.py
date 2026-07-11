import pytest, sqlite3

@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    # never actually send during this test
    monkeypatch.setattr(appmod, "_send_full_report_email", lambda *a, **k: ("console-log", None))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod

def _pub(c, appmod, scan_date):
    return c.post("/api/console/biofield-portal?key=test-secret",
                  json={"email": "a@x.com", "name": "A", "scan_date": scan_date,
                        "scan_id": scan_date, "content": {"greeting": scan_date}})

def test_optout_keeps_pointer_but_still_stores_report(client):
    c, appmod = client
    _pub(c, appmod, "2026-07-02")
    from dashboard import client_portal as cp, portal_biofield_reports as pbr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.set_auto_advance(cx, "a@x.com", False)
    _pub(c, appmod, "2026-07-09")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cp.get_current_scan(cx, "a@x.com") == "2026-07-02"        # pointer unmoved
        assert "2026-07-09" in pbr.list_report_dates(cx, "a@x.com")      # but report stored

    # The opt-out must survive more than one publish. Each publish REPLACES
    # content_json wholesale (upsert_portal), so if auto_advance itself isn't
    # re-injected into `content` on every publish, it silently reverts to the
    # default (True) after exactly one publish post-opt-out — and the pointer
    # starts moving again on this THIRD publish.
    _pub(c, appmod, "2026-07-16")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cp.get_auto_advance(cx, "a@x.com") is False                 # opt-out still in effect
        assert cp.get_current_scan(cx, "a@x.com") == "2026-07-02"          # pointer still unmoved
        assert "2026-07-16" in pbr.list_report_dates(cx, "a@x.com")        # but report still stored
