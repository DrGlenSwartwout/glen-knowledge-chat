"""Slice 1 console API: GET/POST /api/console/condition-programs,
POST /api/console/broad-benefit, and the /console/support-programs editor
serve route. Console-key gated (mirrors test_ff_console_publish.py)."""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import condition_programs as cp
from dashboard import broad_benefit as bb

HDRS = {"X-Console-Key": "testkey"}


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


@pytest.fixture()
def app_mod(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    return app


# ---------------------------------------------------------------------------
# GET /api/console/condition-programs
# ---------------------------------------------------------------------------

def test_get_condition_programs_requires_console_key(app_mod):
    client = app_mod.app.test_client()
    r = client.get("/api/console/condition-programs")
    assert r.status_code in (401, 403)


def test_get_condition_programs_seeds_and_returns_nine(app_mod, tmp_db):
    client = app_mod.app.test_client()
    r = client.get("/api/console/condition-programs", headers=HDRS)
    assert r.status_code == 200
    j = r.get_json()
    assert len(j["programs"]) == 9
    keys = {p["condition_key"] for p in j["programs"]}
    assert keys == {
        "glaucoma-elevated-iop", "glaucoma-normal-iop", "dry-amd", "wet-amd",
        "senile-cataract", "psc-cataract", "dry-eye", "retinitis-pigmentosa",
        "diabetic-retinopathy",
    }
    assert isinstance(j["broad_benefit"], list)
    assert "wholomega" in j["broad_benefit"]


def test_get_condition_programs_seed_is_idempotent(app_mod, tmp_db):
    client = app_mod.app.test_client()
    client.get("/api/console/condition-programs", headers=HDRS)
    r2 = client.get("/api/console/condition-programs", headers=HDRS)
    assert len(r2.get_json()["programs"]) == 9


# ---------------------------------------------------------------------------
# POST /api/console/condition-programs
# ---------------------------------------------------------------------------

def test_post_condition_programs_requires_console_key(app_mod):
    client = app_mod.app.test_client()
    r = client.post("/api/console/condition-programs", json={
        "condition_key": "dry-eye", "label": "Dry Eye", "consult_recommended": False,
        "items": [],
    })
    assert r.status_code in (401, 403)


def test_post_condition_programs_upsert_visible_via_get(app_mod, tmp_db):
    client = app_mod.app.test_client()
    # seed first
    client.get("/api/console/condition-programs", headers=HDRS)
    new_items = [{"slug": "moisturize", "name": "Moisturize", "dose": "2/day"}]
    r = client.post("/api/console/condition-programs", headers=HDRS, json={
        "condition_key": "dry-eye", "label": "Dry Eye (edited)",
        "consult_recommended": True, "items": new_items,
    })
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}
    r2 = client.get("/api/console/condition-programs", headers=HDRS)
    prog = next(p for p in r2.get_json()["programs"] if p["condition_key"] == "dry-eye")
    assert prog["label"] == "Dry Eye (edited)"
    assert prog["consult_recommended"] is True
    assert prog["items"] == new_items


# ---------------------------------------------------------------------------
# POST /api/console/broad-benefit
# ---------------------------------------------------------------------------

def test_post_broad_benefit_requires_console_key(app_mod):
    client = app_mod.app.test_client()
    r = client.post("/api/console/broad-benefit", json={"slug": "x", "action": "add"})
    assert r.status_code in (401, 403)


def test_post_broad_benefit_add_and_remove(app_mod, tmp_db):
    client = app_mod.app.test_client()
    r = client.post("/api/console/broad-benefit", headers=HDRS,
                     json={"slug": "brand-new-slug", "action": "add"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    assert "brand-new-slug" in j["broad_benefit"]

    r2 = client.post("/api/console/broad-benefit", headers=HDRS,
                      json={"slug": "brand-new-slug", "action": "remove"})
    assert r2.status_code == 200
    j2 = r2.get_json()
    assert j2["ok"] is True
    assert "brand-new-slug" not in j2["broad_benefit"]


# ---------------------------------------------------------------------------
# GET /console/support-programs (editor page)
# ---------------------------------------------------------------------------

def test_support_programs_editor_page_serves(app_mod):
    client = app_mod.app.test_client()
    r = client.get("/console/support-programs")
    assert r.status_code == 200
    assert "text/html" in r.content_type
    assert "Support Program" in r.get_data(as_text=True)
