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
    j = r.get_json()
    assert j["ok"] is True
    # "moisturize" IS a real catalog slug, so it must not be flagged unknown.
    # (See FIX 2 tests below for dedicated coverage of the unknown-slug case.)
    assert j["unknown_slugs"] == []
    r2 = client.get("/api/console/condition-programs", headers=HDRS)
    prog = next(p for p in r2.get_json()["programs"] if p["condition_key"] == "dry-eye")
    assert prog["label"] == "Dry Eye (edited)"
    assert prog["consult_recommended"] is True
    assert prog["items"] == new_items


def test_post_condition_programs_flags_unknown_slugs(app_mod, tmp_db):
    client = app_mod.app.test_client()
    client.get("/api/console/condition-programs", headers=HDRS)  # seed first
    r = client.post("/api/console/condition-programs", headers=HDRS, json={
        "condition_key": "dry-eye", "label": "Dry Eye",
        "consult_recommended": False,
        "items": [
            {"slug": "totally-bogus-slug-xyz", "name": "Bogus"},
            {"slug": "wholomega", "name": "WholOmega",
             "alts": [{"slug": "another-bogus-slug", "name": "Also Bogus"}]},
        ],
    })
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    assert set(j["unknown_slugs"]) == {"totally-bogus-slug-xyz", "another-bogus-slug"}
    # the save is NOT blocked by unknown slugs — the program is still saved
    r2 = client.get("/api/console/condition-programs", headers=HDRS)
    prog = next(p for p in r2.get_json()["programs"] if p["condition_key"] == "dry-eye")
    assert prog["label"] == "Dry Eye"


def test_post_condition_programs_valid_slugs_report_empty_unknown(app_mod, tmp_db):
    client = app_mod.app.test_client()
    client.get("/api/console/condition-programs", headers=HDRS)  # seed first
    r = client.post("/api/console/condition-programs", headers=HDRS, json={
        "condition_key": "dry-eye", "label": "Dry Eye",
        "consult_recommended": False,
        "items": [{"slug": "wholomega", "name": "WholOmega"}],
    })
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    assert j["unknown_slugs"] == []


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


def test_post_broad_benefit_add_unknown_slug_still_adds_but_flags(app_mod, tmp_db):
    client = app_mod.app.test_client()
    r = client.post("/api/console/broad-benefit", headers=HDRS,
                     json={"slug": "totally-bogus-slug-xyz", "action": "add"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    assert "totally-bogus-slug-xyz" in j["broad_benefit"]  # still added
    assert j.get("unknown_slug") is True


def test_post_broad_benefit_add_known_slug_no_unknown_flag(app_mod, tmp_db):
    client = app_mod.app.test_client()
    r = client.post("/api/console/broad-benefit", headers=HDRS,
                     json={"slug": "wholomega", "action": "add"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    assert not j.get("unknown_slug")


# ---------------------------------------------------------------------------
# GET /console/support-programs (editor page)
# ---------------------------------------------------------------------------

def test_support_programs_editor_page_serves(app_mod):
    client = app_mod.app.test_client()
    r = client.get("/console/support-programs")
    assert r.status_code == 200
    assert "text/html" in r.content_type
    assert "Support Program" in r.get_data(as_text=True)


# ---------------------------------------------------------------------------
# Real seed content (data/condition_programs_seed.json), through the seed path
# ---------------------------------------------------------------------------

EXPECTED_BROAD_BENEFIT_SLUGS = {
    "glutathione-syntropy", "vitamin-c-syntropy", "perfect-skin", "fiber-cleanse",
    "microbiome", "liver-support", "nous-energy", "chelation", "immune-modulation",
    "scar-solve", "scar-silk", "scar-soft-drink", "lymph-flow", "sleep-syntropy",
    "mitochondrial-biogenesis", "wholomega", "brain-cleanse", "free-and-easy",
    "glucose-tolerance", "reverse-age", "neuro-magnesium", "rise--shine",
    "vital-energy-be", "sustain", "stress-release", "vitality",
}


def test_real_seed_content_loads_correctly_through_seed_path(app_mod, tmp_db):
    """Loads the ACTUAL data/condition_programs_seed.json (not a test fixture)
    through the real app seed path (GET triggers _init_support_programs_tables,
    which is unmocked here) and asserts Glen-approved seed content survives
    the round trip."""
    client = app_mod.app.test_client()
    r = client.get("/api/console/condition-programs", headers=HDRS)
    assert r.status_code == 200
    j = r.get_json()

    programs = {p["condition_key"]: p for p in j["programs"]}
    assert len(programs) == 9

    assert len(EXPECTED_BROAD_BENEFIT_SLUGS) == 26
    assert set(j["broad_benefit"]) == EXPECTED_BROAD_BENEFIT_SLUGS

    assert programs["wet-amd"]["consult_recommended"] is True

    cataract_items = programs["senile-cataract"]["items"]
    lens_zyme = next(it for it in cataract_items if it["slug"] == "lens-zyme")
    assert "brunescent" in lens_zyme.get("note", "").lower()

    # dose/alts round-trip intact
    glaucoma_items = programs["glaucoma-elevated-iop"]["items"]
    iop = next(it for it in glaucoma_items if it["slug"] == "iop-syntropy")
    assert iop["dose"] == "3 times a day"
    ocuheal = next(it for it in glaucoma_items if it["slug"] == "ocuheal-eye-drops")
    assert ocuheal["alts"] == [
        {"slug": "neuro-eye-drops-aces-gl-lite-eye-drops", "name": "Neuro Eye Drops"},
    ]
