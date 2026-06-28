import sqlite3
import app as appmod
import dashboard
from dashboard import points
from dashboard.pay_it_forward import MILESTONE_REWARD_CENTS


def _client(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db)
    points.init_points_table(cx)
    cx.close()
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "testsecret")
    return appmod.app.test_client()


def test_milestone_route_requires_console_key(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/admin/pif/milestone", json={"email": "m@x.com", "milestone_key": "k"})
    assert r.status_code == 401


def test_milestone_route_credits(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/admin/pif/milestone?key=testsecret",
               json={"email": "m@x.com", "milestone_key": "program_complete_1"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["balance_cents"] == MILESTONE_REWARD_CENTS


def test_milestone_route_validates_input(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/admin/pif/milestone?key=testsecret", json={"email": "m@x.com"})
    assert r.status_code == 400


def test_milestone_route_rejects_bad_value_cents(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/admin/pif/milestone?key=testsecret",
               json={"email": "m@x.com", "milestone_key": "k", "value_cents": "abc"})
    assert r.status_code == 400


def test_summary_requires_member(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "PAY_IT_FORWARD_ENABLED", True)
    monkeypatch.setattr(appmod, "is_member", lambda session_id="", email="": False)
    r = c.get("/api/pif/summary?email=m@x.com")
    assert r.status_code == 403


def test_summary_returns_balance_and_chain(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "PAY_IT_FORWARD_ENABLED", True)
    monkeypatch.setattr(appmod, "is_member", lambda session_id="", email="": True)
    # credit a milestone so the member has a balance
    c.post("/admin/pif/milestone?key=testsecret",
           json={"email": "m@x.com", "milestone_key": "k1"})
    r = c.get("/api/pif/summary?email=m@x.com")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["balance_cents"] == 500
    assert body["chain"] == {"reached": 0, "l1": 0, "l2": 0, "levels": []}
    assert body["healer_level"] == 1
    assert body["gift_wallet"] == []


def test_summary_dark_when_flag_off(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "PAY_IT_FORWARD_ENABLED", False)
    r = c.get("/api/pif/summary?email=m@x.com")
    assert r.status_code == 404


def test_summary_gift_wallet_omits_secret_fields(monkeypatch, tmp_path):
    from dashboard import coupons
    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "PAY_IT_FORWARD_ENABLED", True)
    monkeypatch.setattr(appmod, "is_member", lambda session_id="", email="": True)
    # Mint a gift coupon so the wallet is non-empty
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    coupons.init_coupons_table(cx)
    coupons.mint_gift(cx, email="m@x.com", product_slug="neuro-magnesium", pct=15)
    cx.close()
    r = c.get("/api/pif/summary?email=m@x.com")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["gift_wallet"]) == 1
    entry = body["gift_wallet"][0]
    assert set(entry.keys()) == {"product_slug", "pct", "expires_at"}
    assert "code" not in entry
    assert "session_id" not in entry
    assert "email" not in entry
