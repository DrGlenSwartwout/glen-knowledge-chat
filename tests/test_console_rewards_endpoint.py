import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)  # open console auth in test (mirror sibling console tests)
    try:
        import app
        importlib.reload(app)  # CONSOLE_SECRET is read at import time; reload so delenv takes effect
    except Exception as e: pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    return app


def _pending(tmp_db, email="a@ex.com", tier=3):
    from dashboard import data_sharing_rewards as dr
    with sqlite3.connect(str(tmp_db)) as cx:
        dr.init_reward_tables(cx)
        cx.execute("INSERT INTO member_reward_grants (email, reward_type, tier, status, granted_at) "
                   "VALUES (?, 'store_credit', ?, 'pending', '2026-07-14T00:00:00Z')", (email, tier)); cx.commit()


def test_rewards_endpoint_lists_pending_with_options(monkeypatch, tmp_db):
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    app = _app(monkeypatch, tmp_db); _pending(tmp_db)
    j = app.app.test_client().get("/api/console/rewards").get_json()
    assert len(j["items"]) == 1
    it = j["items"][0]
    assert it["email"] == "a@ex.com" and it["tier"] == 3
    assert any(o["sku"] == "GIFT-SAMPLE-3" for o in it["options"])


def test_rewards_endpoint_empty_when_flag_off(monkeypatch, tmp_db):
    monkeypatch.delenv("REWARD_GIFTS_ENABLED", raising=False)
    app = _app(monkeypatch, tmp_db); _pending(tmp_db)
    r = app.app.test_client().get("/api/console/rewards")
    assert r.status_code == 200 and r.get_json()["items"] == []


def test_console_rewards_page_html_markup():
    html = (Path(__file__).resolve().parent.parent / "static" / "console-rewards.html").read_text()
    assert "/api/console/rewards" in html
    assert "reward.select_gift" in html
    assert "reward.dismiss" in html
    assert "<select" in html
