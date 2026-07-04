import importlib
import sqlite3
from dashboard import points

KEY = "test-console-secret"


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONSOLE_SECRET", KEY)
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config["TESTING"] = True
    return appmod


def test_points_ledger_summary_and_filter(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        points.init_points_table(cx)
        points.credit(cx, "amb@x.com", value_cents=1000, reason="referral_reward", order_ref="referral:a")
        points.credit(cx, "up@x.com", value_cents=500, reason="referral_reward_l2", order_ref="referral_l2:a")
        points.credit(cx, "up@x.com", value_cents=570, reason="referral_reward_l2", order_ref="disp_l2:INV1")
        cx.commit()
    c = appmod.app.test_client()
    d = c.get("/api/console/points-ledger", headers={"X-Console-Key": KEY}).get_json()
    by = {s["reason"]: s for s in d["summary"]}
    assert by["referral_reward"]["total_cents"] == 1000 and by["referral_reward"]["entries"] == 1
    assert by["referral_reward_l2"]["total_cents"] == 1070 and by["referral_reward_l2"]["entries"] == 2
    # filtered rows
    d2 = c.get("/api/console/points-ledger?reason=referral_reward_l2", headers={"X-Console-Key": KEY}).get_json()
    assert len(d2["rows"]) == 2 and all(r["reason"] == "referral_reward_l2" for r in d2["rows"])


def test_points_ledger_requires_key(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    assert appmod.app.test_client().get("/api/console/points-ledger").status_code == 401
