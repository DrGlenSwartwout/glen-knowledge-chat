"""Tests for GET/POST /api/practitioner/settings.

Stubs:
  - _practitioner_session_pid → "p1" (or None for unauthed tests)
  - appmod.LOG_DB → tmp sqlite file (avoids touching the real DB)
"""

import sqlite3
import os
import pytest

import app as appmod


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Point LOG_DB at a fresh temp sqlite file; tear down after the test."""
    db_path = tmp_path / "chat_log.db"
    monkeypatch.setattr(appmod, "LOG_DB", db_path)
    yield db_path


@pytest.fixture()
def client(tmp_db):
    """Flask test client with a fresh tmp DB."""
    return appmod.app.test_client()


# ── 401 without a practitioner session ───────────────────────────────────────

def test_get_requires_auth(monkeypatch, client):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: None)
    r = client.get("/api/practitioner/settings")
    assert r.status_code == 401


def test_post_requires_auth(monkeypatch, client):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: None)
    r = client.post("/api/practitioner/settings", json={})
    assert r.status_code == 401


# ── GET returns defaults for a fresh practitioner ────────────────────────────

def test_get_defaults_for_fresh_practitioner(monkeypatch, client):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "p1")
    r = client.get("/api/practitioner/settings")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["branding"] == {}
    assert data["pricing"] == {"default_markup_pct": 0, "overrides": {}}
    assert data["chat_enabled"] is False


# ── POST writes branding + pricing, GET reads them back ──────────────────────

def test_post_branding_and_pricing_roundtrip(monkeypatch, client):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "p1")

    payload = {
        "branding": {
            "practice_name": "Acme Wellness",
            "contact_email": "dr@acme.com",
            "brand_color_1": "#336699",
        },
        "pricing": {
            "default_markup_pct": 15,
            "overrides": {"brain-boost": 9000},
        },
    }
    r_post = client.post("/api/practitioner/settings", json=payload)
    assert r_post.status_code == 200
    post_data = r_post.get_json()
    assert post_data["ok"] is True
    assert post_data["clamped"] == []
    assert post_data["branding"]["practice_name"] == "Acme Wellness"
    assert post_data["pricing"]["default_markup_pct"] == 15.0
    assert post_data["pricing"]["overrides"]["brain-boost"] == 9000

    # Now read back via GET
    r_get = client.get("/api/practitioner/settings")
    assert r_get.status_code == 200
    get_data = r_get.get_json()
    assert get_data["ok"] is True
    assert get_data["branding"]["practice_name"] == "Acme Wellness"
    assert get_data["branding"]["contact_email"] == "dr@acme.com"
    assert get_data["pricing"]["default_markup_pct"] == 15.0
    assert get_data["pricing"]["overrides"]["brain-boost"] == 9000


# ── POST with per-SKU override below MAP → clamped up ────────────────────────

def test_post_override_below_map_is_clamped(monkeypatch, client):
    """$60 override on a product where MAP is $67 → clamped to $67 (6700 cents)."""
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "p1")

    payload = {
        "pricing": {
            "default_markup_pct": 0,
            "overrides": {
                "retina-renew": 6000,   # $60 — below MAP of $67
                "brain-boost": 7500,    # $75 — above MAP, no clamp
            },
        },
    }
    r = client.post("/api/practitioner/settings", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True

    # retina-renew was clamped
    assert len(data["clamped"]) == 1
    clamped_entry = data["clamped"][0]
    assert clamped_entry["slug"] == "retina-renew"
    assert clamped_entry["requested_cents"] == 6000
    assert clamped_entry["clamped_to_cents"] == 6700  # MAP

    # stored prices reflect the clamp
    assert data["pricing"]["overrides"]["retina-renew"] == 6700
    assert data["pricing"]["overrides"]["brain-boost"] == 7500


# ── POST with invalid markup → 400 ───────────────────────────────────────────

def test_post_non_numeric_markup_returns_400(monkeypatch, client):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "p1")
    r = client.post("/api/practitioner/settings",
                    json={"pricing": {"default_markup_pct": "not-a-number"}})
    assert r.status_code == 400
    data = r.get_json()
    assert data["ok"] is False
    assert "markup" in (data.get("error") or "").lower()


# ── POST only branding (no pricing key) → pricing stays default ──────────────

def test_post_branding_only_sets_default_pricing(monkeypatch, client):
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "p1")
    r = client.post("/api/practitioner/settings",
                    json={"branding": {"practice_name": "Solo Clinic"}})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["branding"]["practice_name"] == "Solo Clinic"
    # pricing written with defaults (markup 0, no overrides)
    assert data["pricing"]["default_markup_pct"] == 0.0
    assert data["pricing"]["overrides"] == {}
