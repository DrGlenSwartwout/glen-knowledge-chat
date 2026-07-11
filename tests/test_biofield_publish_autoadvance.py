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


def test_empty_scan_date_publish_preserves_optout_and_pin(client):
    # T5 regression: a publish with NO scan_date (e.g. console editor saving just
    # a greeting/content tweak) used to skip the whole prefs-preservation block,
    # so upsert_portal's wholesale content_json replace silently reverted an
    # opted-out client's auto_advance to the default True and dropped their pin.
    c, appmod = client
    from dashboard import client_portal as cp
    _pub(c, appmod, "2026-07-02")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.set_auto_advance(cx, "a@x.com", False)
        cp.set_current_scan(cx, "a@x.com", "2026-07-02")
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "a@x.com", "name": "A",
                     "content": {"greeting": "updated greeting only"}})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cp.get_auto_advance(cx, "a@x.com") is False       # opt-out preserved
        assert cp.get_current_scan(cx, "a@x.com") == "2026-07-02"  # pin preserved


def test_published_report_row_content_has_no_pref_pollution(client):
    # The per-scan report row (portal_biofield_reports) must hold the dated
    # clinical content only — auto_advance/current_scan_date are portal-level
    # prefs and must never be stamped into a report's content.
    c, appmod = client
    _pub(c, appmod, "2026-07-02")
    from dashboard import portal_biofield_reports as pbr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rep = pbr.get_report(cx, "a@x.com", "2026-07-02")
    report_content = rep.get("content") or {}
    assert "auto_advance" not in report_content
    assert "current_scan_date" not in report_content
