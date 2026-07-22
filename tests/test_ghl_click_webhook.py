import sqlite3
import app as app_module
from dashboard import recommendation_events as re


def _seed(tmp_path, monkeypatch, *, secret="s3cret"):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    re.init_recommendation_events(cx)
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "WEBHOOK_SECRET", secret, raising=False)
    # catalog stub: terrain-restore resolves to itself; anything else is not a product
    monkeypatch.setattr(app_module, "_rec_valid_slug",
                        lambda s: ("terrain-restore" if s == "terrain-restore" else None),
                        raising=False)
    app_module.app.config["TESTING"] = True
    return db


def test_authed_click_records_newsletter_event(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=s3cret",
               json={"email": "A@B.com", "product_slug": "terrain-restore"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "newsletter" and e["product_key"] == "terrain-restore"
               for e in re.list_events(cx, "a@b.com"))   # normalized lower


def test_real_ghl_payload_nested_customdata_records(tmp_path, monkeypatch):
    """GHL's Outbound Webhook nests the workflow's custom data under `customData`
    (email stays top-level). Verified live 2026-07-22 with a real GHL 'Test workflow'
    fire — the flat-payload tests above missed this, so no events would record.
    Payload below is the real shape (PII synthesized)."""
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=s3cret", json={
        "contact_id": "TqB0thchnGJE54QvP7ZA",
        "first_name": "Tester", "last_name": "McTester", "full_name": "Tester McTester",
        "email": "tester@example.com",
        "location": {"name": "Remedy Match", "city": "Hilo", "state": "HI"},
        "workflow": {"id": "wf-1", "name": "Newsletter click - terrain-restore"},
        "triggerData": {},
        "customData": {"product_slug": "terrain-restore"},
    })
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "newsletter" and e["product_key"] == "terrain-restore"
               for e in re.list_events(cx, "tester@example.com"))


def test_bad_secret_is_401_and_records_nothing(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=wrong",
               json={"email": "a@b.com", "product_slug": "terrain-restore"})
    assert r.status_code == 401
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []


def test_secret_via_header_also_accepted(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click",
               headers={"X-Webhook-Secret": "s3cret"},
               json={"email": "a@b.com", "product_slug": "terrain-restore"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    assert len(re.list_events(cx, "a@b.com")) == 1


def test_invalid_slug_records_nothing_but_200(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=s3cret",
               json={"email": "a@b.com", "product_slug": "junk-slug"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []


def test_missing_email_records_nothing_but_200(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=s3cret",
               json={"product_slug": "terrain-restore"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    n = cx.execute("SELECT COUNT(*) FROM recommendation_events").fetchone()[0]
    assert n == 0


def test_disallowed_source_records_nothing_but_200(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=s3cret",
               json={"email": "a@b.com", "product_slug": "terrain-restore", "source": "chat"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []
