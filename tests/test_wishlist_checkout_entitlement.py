import sqlite3
import app as appmod
from dashboard import wishlist as w


def _client(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "_WISHLIST_ENABLED", True)
    monkeypatch.setattr(appmod, "_portal_record_for",
                        lambda cx, token: {"email": "buyer@x.com", "content": {}})
    monkeypatch.setattr(appmod, "_portal_entitled_slugs", lambda email: set())
    monkeypatch.setattr(appmod, "_accepted_recommendation_slugs", lambda cx, email: set())
    monkeypatch.setattr(appmod, "_portal_priced_lines",
                        lambda items, email=None: ([], [], 0))
    with sqlite3.connect(db) as cx:
        w.init_wishlist_table(cx)
        w.toggle(cx, "email:buyer@x.com", "night-vision")
    return appmod.app.test_client()


def test_wishlist_slug_passes_entitlement(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/portal/tok/checkout", json={"items": [{"slug": "night-vision", "qty": 2}]})
    body = r.get_json()
    assert "isn't available to reorder" not in (body.get("error") or "")


def test_non_wishlist_slug_still_rejected(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/portal/tok/checkout", json={"items": [{"slug": "not-on-list", "qty": 1}]})
    assert r.status_code == 400
    assert "isn't available to reorder" in (r.get_json().get("error") or "")
