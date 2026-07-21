import sqlite3
import app as app_module
from dashboard import recommendation_events as re, wishlist as wl


def _seed(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); re.init_recommendation_events(cx); wl.init_wishlist_table(cx); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_WISHLIST_ENABLED", True, raising=False)
    app_module.app.config["TESTING"] = True
    return db


def test_identified_add_emits_self_anonymous_does_not(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    # identified via rm_reorder_email cookie -> self event
    c.set_cookie("rm_reorder_email", "a@b.com")
    c.post("/begin/wishlist/toggle", json={"slug": "neuro-magnesium"})
    cx = sqlite3.connect(db)
    assert len([e for e in re.list_events(cx, "a@b.com") if e["source_key"] == "self"]) == 1
    cx.close()
    # anonymous (no email cookie) add -> NO self event for anybody
    c2 = app_module.app.test_client()
    c2.post("/begin/wishlist/toggle", json={"slug": "immune-modulation"})
    cx = sqlite3.connect(db)
    rows = cx.execute("SELECT COUNT(*) FROM recommendation_events WHERE source_key='self' AND product_key='immune-modulation'").fetchone()[0]
    assert rows == 0
    cx.close()
