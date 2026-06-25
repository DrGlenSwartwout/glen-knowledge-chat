"""1b-A claim + wallet endpoints. App-importing → run under doppler+DATA_DIR."""
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "REWARDS_1B_ENABLED", True)
    import begin_funnel
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        begin_funnel.init_journey_tables(cx)
        coupons.init_coupons_table(cx)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod, begin_funnel


def _seed_session(begin_funnel, log_db, sid, *, email=None, tos=False, gates=()):
    """Seed a journey_state row directly (mirrors record_unlock's columns)."""
    import json
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(log_db) as cx:
        cx.execute(
            "INSERT INTO journey_state"
            "(session_id,email,unlocked_gates,tos_agreed_at,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            (sid, email or "", json.dumps(sorted(gates)),
             "2026-06-25T00:00:00" if tos else None, now, now))
        cx.commit()


def test_claim_requires_email_and_tos(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "s1", email=None, tos=False, gates=["scan", "course_ww"])
    c.set_cookie("amg_session", "s1")
    r = c.post("/api/journey/claim-coupon", json={"land": "scan"})
    assert r.status_code == 409 and r.get_json()["needs"] == "email_tos"


def test_claim_rejects_incomplete_stage(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "s2", email="m@x.com", tos=True, gates=[])  # nothing done
    c.set_cookie("amg_session", "s2")
    r = c.post("/api/journey/claim-coupon", json={"land": "scan"})
    assert r.status_code == 409 and r.get_json()["needs"] == "complete_stage"


def test_claim_mints_when_eligible_and_wallet_lists_it(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "s3", email="m@x.com", tos=True, gates=["scan", "course_ww"])
    c.set_cookie("amg_session", "s3")
    r = c.post("/api/journey/claim-coupon", json={"land": "scan"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["coupon"]["pct"] == 15 and body["coupon"]["product_slug"]
    w = c.get("/api/journey/wallet").get_json()
    assert any(x["code"] == body["coupon"]["code"] for x in w["coupons"])


def test_routes_return_404_when_flag_off(client, monkeypatch):
    c, appmod, _bf = client
    monkeypatch.setattr(appmod, "REWARDS_1B_ENABLED", False)
    assert c.post("/api/journey/claim-coupon", json={"land": "scan"}).status_code == 404
    assert c.get("/api/journey/wallet").status_code == 404


def test_double_claim_returns_same_coupon(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "s4", email="m@x.com", tos=True, gates=["scan", "course_ww"])
    c.set_cookie("amg_session", "s4")
    a = c.post("/api/journey/claim-coupon", json={"land": "scan"}).get_json()
    b = c.post("/api/journey/claim-coupon", json={"land": "scan"}).get_json()
    assert a["coupon"]["code"] == b["coupon"]["code"]
