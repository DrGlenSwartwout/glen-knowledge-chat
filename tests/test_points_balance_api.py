import sqlite3
import app as appmod
from dashboard import points


def test_balance_requires_auth(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "")
    assert appmod.app.test_client().get("/api/points/balance").status_code == 401


def test_balance_returns_cents_and_dollars(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db"); monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); points.init_points_table(cx)
    points.earn(cx, "a@x.com", full_price_cents=20000, earn_pct=0.05, order_ref="s"); cx.commit()
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    r = appmod.app.test_client().get("/api/points/balance")
    assert r.status_code == 200
    b = r.get_json()
    assert b["balance_cents"] == 1000 and b["balance_dollars"] == "10.00"
