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


def _seed_portal(appmod, email="free@x.com", name="Test Client", content=None):
    from dashboard import client_portal as cp
    content = content or {"greeting": "hi", "layers": [], "reorder_items": []}
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, content)
    cx.close()
    return token


def test_api_program_404_when_flag_off(client, monkeypatch):
    c, appmod = client
    monkeypatch.delenv("PORTAL_PROGRAM_PAGE_ENABLED", raising=False)
    r = c.get("/api/portal/anytoken/program")
    assert r.status_code == 404


def test_api_program_returns_tiers_for_free_client(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PORTAL_PROGRAM_PAGE_ENABLED", "1")
    # Paid availability now requires the new live-gate flag AND continuous-care
    # monthly checkout being on (the actual checkout endpoint the Join button
    # posts to gates on both), so both must be set or the paid tier falls back
    # to "coming_soon".
    monkeypatch.setenv("PROGRAM_PAID_LIVE_ENABLED", "1")
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    # _active_membership_for_email opens its own connection against LOG_DB; the
    # `memberships` table is only created at import time against the real LOG_DB
    # (module-level _init_membership_tables()), not against this test's tmp_path
    # db. Matches the established pattern elsewhere (test_biofield_cart.py,
    # test_analysis_quota_gate.py): monkeypatch it directly for a free client.
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: None)
    tok = _seed_portal(appmod, "free@x.com")
    r = c.get(f"/api/portal/{tok}/program")
    assert r.status_code == 200
    body = r.get_json()
    keys = {t["key"] for t in body["tiers"]}
    assert keys == {"free", "paid", "family"}
    tiers = {t["key"]: t for t in body["tiers"]}
    assert tiers["free"]["state"] == "owned"
    assert tiers["paid"]["state"] == "available"
    assert tiers["paid"]["checkout_path"] == "/portal/offer/continuous-care/checkout"
    assert body["current_tier"] == "free"
    assert body["ambassador"]["status"] == "none"
    assert {g["key"] for g in body["grow"]} == {"practitioner", "coach", "cert"}


def test_api_program_paid_coming_soon_when_not_live(client, monkeypatch):
    # With PROGRAM_PAID_LIVE_ENABLED unset, the paid tier must not go live even
    # if continuous-care checkout itself is enabled: the program page must
    # agree with the checkout endpoint's gate or it renders a dead "Join"
    # button that POSTs into a 404.
    c, appmod = client
    monkeypatch.setenv("PORTAL_PROGRAM_PAGE_ENABLED", "1")
    monkeypatch.delenv("PROGRAM_PAID_LIVE_ENABLED", raising=False)
    monkeypatch.setattr(appmod, "CONTINUOUS_CARE_MONTHLY_ENABLED", True)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: None)
    tok = _seed_portal(appmod, "free2@x.com")
    r = c.get(f"/api/portal/{tok}/program")
    assert r.status_code == 200
    body = r.get_json()
    tiers = {t["key"]: t for t in body["tiers"]}
    assert tiers["paid"]["state"] == "coming_soon"


def test_program_page_404_when_flag_off(client, monkeypatch):
    c, appmod = client
    monkeypatch.delenv("PORTAL_PROGRAM_PAGE_ENABLED", raising=False)
    r = c.get("/portal/tok/program")
    assert r.status_code == 404


def test_program_page_served_when_flag_on(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PORTAL_PROGRAM_PAGE_ENABLED", "1")
    r = c.get("/portal/tok/program")
    assert r.status_code == 200
    assert b'id="program-root"' in r.data


def test_client_portal_payload_exposes_program_page(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("PORTAL_PROGRAM_PAGE_ENABLED", "1")
    # _seed_portal's token is minted by upsert_portal (secrets.token_urlsafe), not
    # settable by the caller, so we capture the real token rather than assume one.
    tok = _seed_portal(appmod, "pp@x.com")
    r = c.get(f"/api/portal/{tok}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["program_page"]["enabled"] is True
    assert body["program_page"]["url"] == f"/portal/{tok}/program"
