import os
import pytest
import sqlite3

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)

import app as appmod


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setenv("PUBLIC_SURFACE_ENABLED", "1")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_api_sample_404s_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setenv("PUBLIC_SURFACE_ENABLED", "")
    c = appmod.app.test_client()
    r = c.get("/api/sample")
    assert r.status_code == 404
    assert r.data == b""


def test_api_sample_returns_demo_payload(client):
    r = client.get("/api/sample")
    assert r.status_code == 200
    body = r.get_json()
    assert body["sample"] is True
    assert body["findings"]


def test_api_sample_needs_no_token_or_cookie(client):
    """Public means public — no auth of any kind."""
    r = client.get("/api/sample")
    assert r.status_code == 200


def test_api_sample_is_noindex(client):
    assert client.get("/api/sample").headers.get("X-Robots-Tag") == "noindex"


def test_sample_page_serves_html(client):
    r = client.get("/sample")
    assert r.status_code == 200
    assert b"<html" in r.data.lower()


def test_sample_page_is_noindex(client):
    r = client.get("/sample")
    assert r.headers.get("X-Robots-Tag") == "noindex"


def test_sample_page_404s_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setenv("PUBLIC_SURFACE_ENABLED", "")
    c = appmod.app.test_client()
    r = c.get("/sample")
    assert r.status_code == 404
    assert r.data == b""


def test_sample_page_loads_no_third_party_scripts(client):
    """No trackers on public health-adjacent surfaces. Every large pixel
    settlement in this space was a private class action, not an OCR action."""
    lowered = client.get("/sample").data.decode("utf-8", "replace").lower()
    for needle in ("googletagmanager", "google-analytics", "connect.facebook",
                   "facebook.net", "hotjar", "fullstory", "segment.com",
                   "clarity.ms", "doubleclick"):
        assert needle not in lowered, f"third-party tracker present: {needle}"


def test_sample_page_loads_no_remote_assets(client):
    """No off-origin script/style/img/iframe at all — the strong form of the
    no-tracker rule, and the one a future marketing change would violate first.

    Covers: src=/href= with http://, https://, or protocol-relative "//";
    CSS url(...) references (inline <style> blocks or style= attributes);
    and @import (bare-string or url() form). Same-origin relative paths
    (e.g. href="/api/sample") are intentionally allowed."""
    import re as _re
    html = client.get("/sample").data.decode("utf-8", "replace")

    remote = []
    # src="..."/href="..." pointing off-origin (absolute http(s) or protocol-relative //)
    remote += _re.findall(
        r'(?:src|href)\s*=\s*["\'](?:https?:)?//[^"\']*', html, _re.I
    )
    # CSS url(...) references, quoted or bare, off-origin
    remote += _re.findall(
        r'url\(\s*["\']?(?:https?:)?//[^)"\']*', html, _re.I
    )
    # @import, either "url(...)" form or bare quoted-string form
    remote += _re.findall(
        r'@import\s+["\']?(?:url\(\s*)?["\']?(?:https?:)?//[^;)"\']*', html, _re.I
    )
    assert remote == [], f"off-origin assets: {remote}"


def test_sample_page_has_no_intake_elements(client):
    """No scheduling widget, symptom checker, or login form on a public page."""
    lowered = client.get("/sample").data.decode("utf-8", "replace").lower()
    assert "<form" not in lowered
    assert "<iframe" not in lowered
    assert "type=\"password\"" not in lowered
    assert "type=\"email\"" not in lowered
    assert "type=\"tel\"" not in lowered
    for needle in ("calendly", "acuityscheduling", "schedule"):
        assert needle not in lowered, f"intake widget marker present: {needle}"


def _seed_affiliate(db_path, slug="prof-jane-doe"):
    cx = sqlite3.connect(db_path)
    cx.executescript("""
      CREATE TABLE IF NOT EXISTS affiliate_signups (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, name TEXT,
        email TEXT, organization TEXT DEFAULT '', website TEXT DEFAULT '',
        promo_method TEXT DEFAULT '', slug TEXT, token TEXT,
        status TEXT DEFAULT 'approved', notes TEXT DEFAULT '',
        referred_by TEXT DEFAULT '', short_url TEXT DEFAULT '');
    """)
    cx.execute(
        "INSERT INTO affiliate_signups (created_at,name,email,organization,slug,token,status)"
        " VALUES ('2026-01-01','Jane Doe','jane@example.com','Doe Wellness',?,'tok','approved')",
        (slug,))
    cx.commit()
    cx.close()


@pytest.fixture
def client_with_affiliate(monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed_affiliate(db)
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("PUBLIC_SURFACE_ENABLED", "1")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_storefront_page_renders(client_with_affiliate):
    r = client_with_affiliate.get("/p/prof-jane-doe")
    assert r.status_code == 200
    assert r.headers.get("X-Robots-Tag") == "noindex"


def test_storefront_unknown_slug_404s(client_with_affiliate):
    assert client_with_affiliate.get("/p/no-such-person").status_code == 404


def test_storefront_sets_attribution_cookie(client_with_affiliate):
    r = client_with_affiliate.get("/p/prof-jane-doe")
    assert "rm_ref=prof-jane-doe" in r.headers.get("Set-Cookie", "")


def test_storefront_api_returns_whitelisted_payload(client_with_affiliate):
    body = client_with_affiliate.get("/api/p/prof-jane-doe").get_json()
    assert body["practitioner_name"] == "Jane Doe"
    assert "profit_disclosure" in body


def test_storefront_body_leaks_no_commercial_terms(client_with_affiliate):
    """The Thorne lesson: Thorne shipped wholesalePrice beside retailPrice in
    plain page source, letting patients compute their practitioner's margin."""
    page = client_with_affiliate.get("/p/prof-jane-doe").data.decode("utf-8", "replace").lower()
    api = client_with_affiliate.get("/api/p/prof-jane-doe").data.decode("utf-8", "replace").lower()
    for needle in ("wholesale", "margin", "markup", "msrp", "revenue", "commission"):
        assert needle not in page, f"leaked {needle} in page"
        assert needle not in api, f"leaked {needle} in api"


def test_dispensary_route_still_works(client_with_affiliate):
    """Old links must never break. /dispensary/<code> is untouched by this work."""
    assert appmod.app.url_map.bind("localhost").match("/dispensary/abc123")[0] == "dispensary_landing"
