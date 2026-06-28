# tests/test_console_biofield_portal.py
import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


_LAYER = {"n": 1, "title": "Calm", "meaning": "settle", "remedy": "Terrain Restore",
          "dosing": "10 drops 3x/day"}


def test_post_creates_portal_and_returns_url(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "x@y.com", "name": "X",
                     "content": {"greeting": "Aloha", "layers": [_LAYER]}})
    assert r.status_code == 200
    j = r.get_json()
    assert j["token"] and j["url"].endswith(j["token"])
    # content round-trips through the public portal API
    r2 = c.get(f"/api/portal/{j['token']}")
    assert r2.get_json()["layers"][0]["title"] == "Calm"


def test_post_requires_console_key(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal",
               json={"email": "x@y.com", "content": {"layers": [_LAYER]}})
    assert r.status_code == 401


def test_post_requires_email(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"content": {"layers": [_LAYER]}})
    assert r.status_code == 400


def test_post_requires_some_content(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "x@y.com", "name": "X", "content": {}})
    assert r.status_code == 400


def test_post_send_emails_link(client, monkeypatch):
    c, appmod = client
    sent = {}
    # _send_full_report_email returns (sent_via, error); a delivering path => emailed.
    def _fake(to, name, subj, body, **k):
        sent.update(to=to, body=body)
        return ("gmail-api", None)
    monkeypatch.setattr(appmod, "_send_full_report_email", _fake)
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "e@y.com", "name": "E", "send": True,
                     "content": {"greeting": "hi", "layers": [_LAYER]}})
    j = r.get_json()
    tok = j["token"]
    assert sent["to"] == "e@y.com" and tok in sent["body"]
    assert j["emailed"] is True and j["email_status"] == "gmail-api"


def test_post_send_console_log_not_marked_emailed(client, monkeypatch):
    # If the send falls through to console-log (nothing actually delivered), the
    # response must NOT claim the client was emailed.
    c, appmod = client
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda to, name, subj, body, **k: ("console-log", "no email-send mechanism configured"))
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "e@y.com", "name": "E", "send": True,
                     "content": {"greeting": "hi", "layers": [_LAYER]}})
    j = r.get_json()
    assert j["emailed"] is False and j["email_status"] == "console-log"


def test_post_send_emails_on_republish_existing_portal(client, monkeypatch):
    # Regression: re-publishing an existing portal makes upsert return token=None
    # (only the hash is stored). The old `if token and send` guard then skipped the
    # email entirely, so "Publish & email client" sent nothing on any republish.
    c, appmod = client
    c.post("/api/console/biofield-portal?key=test-secret",
           json={"email": "karin@y.com", "name": "Karin",
                 "content": {"greeting": "hi", "layers": [_LAYER]}})
    sent = {}
    def _fake(to, name, subj, body, **k):
        sent.update(to=to, body=body)
        return ("gmail-api", None)
    monkeypatch.setattr(appmod, "_send_full_report_email", _fake)
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "karin@y.com", "name": "Karin", "send": True,
                     "content": {"greeting": "hi again", "layers": [_LAYER]}})
    j = r.get_json()
    assert j["updated"] is True                        # was an update (token None)
    assert sent.get("to") == "karin@y.com"             # email STILL attempted
    assert j["emailed"] is True
    assert j["url"] and "/portal/" in j["url"]         # usable link returned
    assert "/portal/" in sent["body"]                  # email carries a working link


def _seed(appmod, email="seed@y.com", name="Seed"):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    cp.upsert_portal(cx, email, name, {"greeting": "hello", "layers": [_LAYER]})
    cx.close()


def test_get_loads_existing_portal(client):
    c, appmod = client
    _seed(appmod, "seed@y.com", "Seed")
    j = c.get("/api/console/biofield-portal?key=test-secret&email=seed@y.com").get_json()
    assert j["found"] is True
    assert j["name"] == "Seed"
    assert j["content"]["layers"][0]["title"] == "Calm"


def test_get_unknown_returns_scaffold(client):
    c, _ = client
    j = c.get("/api/console/biofield-portal?key=test-secret&email=nobody@y.com").get_json()
    assert j["found"] is False
    assert j["content"] == {}


def test_get_requires_key(client):
    c, _ = client
    assert c.get("/api/console/biofield-portal?email=x@y.com").status_code == 401


def test_catalog_returns_products(client):
    c, _ = client
    j = c.get("/api/console/biofield-portal/catalog?key=test-secret").get_json()
    assert isinstance(j["products"], list) and j["products"]
    assert "slug" in j["products"][0] and "name" in j["products"][0]


def test_page_served(client):
    c, _ = client
    assert c.get("/console/biofield-portal").status_code == 200


# ── Import from FMP ──────────────────────────────────────────────────────────

def test_import_fmp_returns_content(client, monkeypatch):
    c, _ = client
    from dashboard import fmp_biofield
    monkeypatch.setattr(fmp_biofield, "import_content",
        lambda email, name="", tags=None: {"greeting": "G",
            "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "R", "dosing": "d"}],
            "video": {}, "reorder_items": [], "pricing_note": ""})
    r = c.post("/api/console/biofield-portal/import-fmp?key=test-secret",
               json={"email": "e@x.com", "name": "O"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["found"] is True
    assert j["content"]["layers"][0]["title"] == "Calm"


def test_import_fmp_not_found(client, monkeypatch):
    c, _ = client
    from dashboard import fmp_biofield
    monkeypatch.setattr(fmp_biofield, "import_content", lambda *a, **k: None)
    r = c.post("/api/console/biofield-portal/import-fmp?key=test-secret",
               json={"email": "nobody@x.com"})
    assert r.status_code == 200
    assert r.get_json()["found"] is False


def test_import_fmp_requires_key(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal/import-fmp", json={"email": "e@x.com"})
    assert r.status_code == 401


# ── Review queue + confirm-on-publish ────────────────────────────────────────

def test_review_queue_lists_only_requested(client):
    c, appmod = client
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    cp.upsert_portal(cx, "req@y.com", "Req", {"biofield_status": "requested", "layers": []})
    cp.upsert_portal(cx, "drft@y.com", "Drft", {"biofield_status": "ai_draft", "layers": []})
    cx.close()
    j = c.get("/api/console/biofield/review-queue?key=test-secret").get_json()
    emails = [r["email"] for r in j["queue"]]
    assert "req@y.com" in emails and "drft@y.com" not in emails


def test_review_queue_requires_key(client):
    c, _ = client
    assert c.get("/api/console/biofield/review-queue").status_code == 401


def test_publish_confirms_status(client):
    c, appmod = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "cf@y.com", "name": "CF",
                     "content": {"layers": [{"n": 1, "title": "Calm", "remedy": "R"}]}})
    assert r.status_code == 200
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    assert cp.get_biofield_status(cx, "cf@y.com") == "confirmed"


def test_corrections_logged_and_listable(client):
    c, appmod = client
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod._log_biofield_correction(cx, "corr@y.com", "2026-06-05",
                                        {"layers": [{"n": 1, "title": "T", "remedy": "Real FF"}]})
    j = c.get("/api/console/biofield/corrections?key=test-secret&since=2000-01-01").get_json()
    hit = [x for x in j["corrections"] if x["email"] == "corr@y.com" and x["scan_date"] == "2026-06-05"]
    assert hit and hit[0]["content"]["layers"][0]["remedy"] == "Real FF"


def test_corrections_requires_key(client):
    c, _ = client
    assert c.get("/api/console/biofield/corrections").status_code == 401


def test_publish_writes_dated_report_and_logs_correction(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3
    r = c.post("/api/console/biofield-portal?key=test-secret", json={
        "email": "pub@y.com", "name": "Pub", "scan_date": "2026-06-05",
        "content": {"layers": [{"n": 1, "title": "Calm", "remedy": "Real FF"}]}})
    assert r.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    rep = R.get_report(cx, "pub@y.com", "2026-06-05")
    assert rep is not None and rep["status"] == "confirmed"
    j = c.get("/api/console/biofield/corrections?key=test-secret&since=2000-01-01").get_json()
    assert any(x["email"] == "pub@y.com" and x["scan_date"] == "2026-06-05" for x in j["corrections"])


def test_review_queue_lists_requested_reports_with_dates(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3, datetime
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    today = datetime.date.today().isoformat()
    R.upsert_report(cx, "rq@y.com", today, "s1", {"layers": []}, "requested")
    R.upsert_report(cx, "rq2@y.com", today, "s2", {"layers": []}, "ai_draft")
    cx.close()
    j = c.get("/api/console/biofield/review-queue?key=test-secret").get_json()
    hits = [(x["email"], x.get("scan_date")) for x in j["queue"]]
    assert ("rq@y.com", today) in hits and all(e != "rq2@y.com" for e, _ in hits)


def test_load_returns_scan_dates_and_selected_report(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as R
    import sqlite3, datetime
    cx = sqlite3.connect(appmod.LOG_DB); R.init_table(cx)
    today = datetime.date.today().isoformat()
    old = (datetime.date.today() - datetime.timedelta(days=40)).isoformat()
    R.upsert_report(cx, "ld@y.com", today, "s1", {"layers": [{"n": 1, "title": "New", "remedy": "Y"}]}, "ai_draft")
    R.upsert_report(cx, "ld@y.com", old, "s0", {"layers": [{"n": 1, "title": "Old", "remedy": "X"}]}, "confirmed")
    cx.close()
    j = c.get("/api/console/biofield-portal?key=test-secret&email=ld@y.com").get_json()
    assert j["found"] and j["scan_dates"] == [today, old] and j["scan_date"] == today
    assert j["content"]["layers"][0]["title"] == "New" and j["content"]["layers"][0]["remedy"] == "Y"
    j2 = c.get("/api/console/biofield-portal?key=test-secret&email=ld@y.com&scan_date=" + old).get_json()
    assert j2["scan_date"] == old and j2["content"]["layers"][0]["title"] == "Old"


def test_load_legacy_no_reports_unchanged(client):
    c, appmod = client
    from dashboard import client_portal as cp
    import sqlite3
    cx = sqlite3.connect(appmod.LOG_DB); cp.init_client_portal_table(cx)
    cp.upsert_portal(cx, "lg2@y.com", "Lg2", {"layers": [{"n": 1, "title": "Legacy"}]}); cx.close()
    j = c.get("/api/console/biofield-portal?key=test-secret&email=lg2@y.com").get_json()
    assert j["found"] and j["scan_dates"] == [] and j["content"]["layers"][0]["title"] == "Legacy"
