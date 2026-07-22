# tests/test_email_click_redirect.py
import sqlite3
import app as app_module
from dashboard import email_click_tokens as ect
from dashboard import recommendation_events as re


def _seed(tmp_path, monkeypatch, *, email="a@b.com"):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    ect.init_email_click_tokens(cx)
    re.init_recommendation_events(cx)
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    # catalog stub: terrain-restore is real, junk-slug is not
    monkeypatch.setattr(app_module, "_rec_valid_slug", lambda s: (s if s == "terrain-restore" else None), raising=False)
    app_module.app.config["TESTING"] = True
    cx = sqlite3.connect(db)
    token = ect.token_for(cx, email)
    cx.close()
    return db, token


def test_valid_email_click_records_and_redirects(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.get(f"/r/{token}/email/terrain-restore")
    assert r.status_code in (301, 302)
    assert "/begin/product/terrain-restore" in r.headers["Location"]
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "email" and e["product_key"] == "terrain-restore"
               for e in re.list_events(cx, "a@b.com"))


def test_unknown_token_records_nothing_but_still_redirects(tmp_path, monkeypatch):
    db, _ = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.get("/r/bogustoken/email/terrain-restore")
    assert r.status_code in (301, 302)
    cx = sqlite3.connect(db)
    n = cx.execute("SELECT COUNT(*) FROM recommendation_events").fetchone()[0]
    assert n == 0


def test_disallowed_source_records_nothing(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.get(f"/r/{token}/chat/terrain-restore")   # 'chat' not allowed via email links
    assert r.status_code in (301, 302)
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []


def test_invalid_slug_records_nothing_and_redirects_home(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.get(f"/r/{token}/email/junk-slug")
    assert r.status_code in (301, 302)
    assert r.headers["Location"].endswith("/")
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []


def test_newsletter_source_is_allowed(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    c.get(f"/r/{token}/newsletter/terrain-restore")
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "newsletter" for e in re.list_events(cx, "a@b.com"))


def test_rec_valid_slug_routes_through_real_resolver():
    import app as app_module
    # active, unsuperseded slug resolves to itself
    assert app_module._rec_valid_slug("terrain-restore") == "terrain-restore"
    # a slug that is not a sellable product resolves to None
    assert app_module._rec_valid_slug("definitely-not-a-real-slug-xyz") is None
    assert app_module._rec_valid_slug("") is None
