# tests/test_gifting_activate.py
import sqlite3, json
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "REWARDS_1B_GIFT_ENABLED", True)
    import begin_funnel
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        begin_funnel.init_journey_tables(cx)
        coupons.init_coupons_table(cx)
    appmod._init_referral_tables()
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod, begin_funnel


def _seed_session(bf, log_db, sid, *, email=None, tos=False, gates=()):
    with sqlite3.connect(log_db) as cx:
        cx.execute(
            "INSERT INTO journey_state(session_id,email,unlocked_gates,tos_agreed_at,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (sid, email or "", json.dumps(sorted(gates)),
             "2026-06-25T00:00:00" if tos else None, "2026-06-25T00:00:00", "2026-06-25T00:00:00"))
        cx.commit()


def test_activate_requires_email_tos(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "g1", email=None, tos=False)
    c.set_cookie("amg_session", "g1")
    r = c.post("/api/journey/activate-gifting", json={})
    assert r.status_code == 409 and r.get_json()["needs"] == "email_tos"


def test_activate_sets_gifting_but_not_ambassador(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "g2", email="me@x.com", tos=True)
    c.set_cookie("amg_session", "g2")
    r = c.post("/api/journey/activate-gifting", json={})
    assert r.status_code == 200 and r.get_json()["gifting"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = cx.execute("SELECT status, gifting_activated_at FROM affiliate_signups WHERE email='me@x.com'").fetchone()
        assert row and row[0] == "pending" and row[1]  # pending + activated timestamp
        assert appmod._is_ambassador(cx, "me@x.com") is False  # commission still gated


def test_activate_flag_off_404(client):
    c, appmod, bf = client
    appmod.REWARDS_1B_GIFT_ENABLED = False
    _seed_session(bf, appmod.LOG_DB, "g3", email="me@x.com", tos=True)
    c.set_cookie("amg_session", "g3")
    assert c.post("/api/journey/activate-gifting", json={}).status_code == 404
