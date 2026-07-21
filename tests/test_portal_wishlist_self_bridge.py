import sqlite3
import app as app_module
from dashboard import recommendation_events as re, client_portal as cp, wishlist as wl


def _seed(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx); re.init_recommendation_events(cx); wl.init_wishlist_table(cx)
    token, _pid = cp.upsert_portal(cx, "a@b.com", "Al", {})
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_WISHLIST_ENABLED", True, raising=False)
    app_module.app.config["TESTING"] = True
    return db, token


def test_portal_wishlist_add_emits_self_remove_does_not(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    # add -> self event
    c.post(f"/api/portal/{token}/wishlist/toggle", json={"slug": "neuro-magnesium"})
    cx = sqlite3.connect(db)
    ev = re.list_events(cx, "a@b.com")
    assert len(ev) == 1 and ev[0]["source_key"] == "self"
    cx.close()
    # toggle again (remove) -> no new self event
    c.post(f"/api/portal/{token}/wishlist/toggle", json={"slug": "neuro-magnesium"})
    cx = sqlite3.connect(db)
    assert len([e for e in re.list_events(cx, "a@b.com") if e["source_key"] == "self"]) == 1
    cx.close()
