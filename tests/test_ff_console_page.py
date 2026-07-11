"""Slice 3c.3: the console review-and-publish HTML surface for FF-match
drafts. The page itself is static; this only proves the serve route works
and mirrors the shared console pattern (mirrors test_console_stub_routes.py)."""
import app as appmod


def _client(monkeypatch):
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "", raising=False)
    return appmod.app.test_client()


def test_ff_drafts_console_page_served(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/console/ff-drafts")
    assert r.status_code == 200
    assert r.content_type.startswith("text/html")
    assert b"FF Match Review" in r.data
    assert b"op-nav.js" in r.data  # carries the shared nav
