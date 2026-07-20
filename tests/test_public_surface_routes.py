import os
import pytest

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
    assert c.get("/sample").status_code == 404


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
    no-tracker rule, and the one a future marketing change would violate first."""
    import re as _re
    html = client.get("/sample").data.decode("utf-8", "replace")
    remote = _re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)', html, _re.I)
    assert remote == [], f"off-origin assets: {remote}"


def test_sample_page_has_no_intake_elements(client):
    """No scheduling widget, symptom checker, or login form on a public page."""
    lowered = client.get("/sample").data.decode("utf-8", "replace").lower()
    assert "<form" not in lowered
    assert "type=\"password\"" not in lowered
