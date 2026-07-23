"""Endpoint-level tests for the My Healing Oasis Build-Out actions:
POST /api/portal/<token>/oasis/tool/add, .../tool/remove, .../roadmap/want.

Follows the tests/test_health_profile_endpoint.py fixture pattern: swap LOG_DB
to a tmp sqlite file so tests never touch the dev db, and use app.test_client().
Identity comes ONLY from the portal token (never the request body), same as
every other /api/portal/<token>/... route.
"""
import os
# Dummy keys so `import app` (which constructs OpenAI + Pinecone clients at import)
# succeeds under a secretless CI without doppler.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")

import json
import sqlite3
import pytest


def _repo_catalog():
    """Repo products.json, independent of $DATA_DIR -- the roadmap-want test
    asserts on a real wishlist slug (aces-eyedrops), so it must see the real
    catalog, not a stripped $DATA_DIR products.json left by the full suite."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "data", "products.json")) as f:
        return (json.load(f) or {}).get("products", {})


@pytest.fixture(autouse=True)
def _pin_repo_catalog(monkeypatch):
    monkeypatch.setattr("dashboard.products.load_products", _repo_catalog)


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "_PORTAL_OASIS_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _mint_portal(appmod, email):
    from dashboard import client_portal as _cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        token, _pid = _cp.upsert_portal(cx, email, "Test Client", {})
    return token


def test_tool_add_flag_off_returns_404(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_OASIS_ENABLED", False)
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/oasis/tool/add", json={"name": "Some Tool"})
    assert r.status_code == 404


def test_tool_add_bad_token_returns_404(client):
    c, appmod = client
    r = c.post("/api/portal/not-a-real-token/oasis/tool/add", json={"name": "Some Tool"})
    assert r.status_code == 404


def test_tool_add_appears_in_owned_external(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/oasis/tool/add",
               json={"name": "My Ionizer", "brand": "Acme"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["enabled"] is True
    names = {t["name"] for t in body["build_out"]["owned_external"]}
    assert "My Ionizer" in names


def test_tool_add_with_hero_family_slug_excludes_hero_from_roadmap(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/oasis/tool/add",
               json={"name": "5-Plate Water Ionizer", "brand": "Living Water",
                     "slug": "water-ionizer-5plate"})
    assert r.status_code == 200
    body = r.get_json()
    road_slugs = {i["slug"] for i in body["build_out"]["roadmap"]}
    assert "water-ionizer" not in road_slugs


def test_tool_remove_drops_it(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/oasis/tool/add", json={"name": "My Ionizer"})
    tool_id = r.get_json()["build_out"]["owned_external"][0]["id"]
    r2 = c.post(f"/api/portal/{token}/oasis/tool/remove", json={"tool_id": tool_id})
    assert r2.status_code == 200
    body = r2.get_json()
    names = {t["name"] for t in body["build_out"]["owned_external"]}
    assert "My Ionizer" not in names


def test_tool_remove_flag_off_returns_404(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_OASIS_ENABLED", False)
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/oasis/tool/remove", json={"tool_id": 1})
    assert r.status_code == 404


def test_tool_remove_bad_token_returns_404(client):
    c, appmod = client
    r = c.post("/api/portal/not-a-real-token/oasis/tool/remove", json={"tool_id": 1})
    assert r.status_code == 404


def test_roadmap_want_lands_on_wishlist(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/oasis/roadmap/want", json={"slug": "harmony"})
    assert r.status_code == 200
    from dashboard import wishlist
    with sqlite3.connect(appmod.LOG_DB) as cx:
        slugs = wishlist.slugs_for(cx, "email:a@b.com")
    assert "harmony" in slugs


def test_roadmap_want_is_idempotent_never_toggles_off(client):
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    c.post(f"/api/portal/{token}/oasis/roadmap/want", json={"slug": "harmony"})
    r2 = c.post(f"/api/portal/{token}/oasis/roadmap/want", json={"slug": "harmony"})
    assert r2.status_code == 200
    from dashboard import wishlist
    with sqlite3.connect(appmod.LOG_DB) as cx:
        slugs = wishlist.slugs_for(cx, "email:a@b.com")
    assert "harmony" in slugs  # still present -- a second "want" must not remove it


def test_roadmap_want_flag_off_returns_404(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_PORTAL_OASIS_ENABLED", False)
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/oasis/roadmap/want", json={"slug": "harmony"})
    assert r.status_code == 404


def test_roadmap_want_bad_token_returns_404(client):
    c, appmod = client
    r = c.post("/api/portal/not-a-real-token/oasis/roadmap/want", json={"slug": "harmony"})
    assert r.status_code == 404


def test_roadmap_want_surfaces_in_build_out_wanted(client):
    """Task 7: closing the loop -- a "want" doesn't just land silently on the
    wishlist table, it comes back resolved (name/url) in the SAME response's
    build_out.wanted, and again on the next portal-view read."""
    c, appmod = client
    token = _mint_portal(appmod, "a@b.com")
    r = c.post(f"/api/portal/{token}/oasis/roadmap/want", json={"slug": "aces-eyedrops"})
    assert r.status_code == 200
    body = r.get_json()
    wanted_slugs = {w["slug"] for w in body["build_out"]["wanted"]}
    assert "aces-eyedrops" in wanted_slugs
