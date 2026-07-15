"""GET /api/portal/<token> — the `life_stress` payload key (Task 5) and the
saved-selection pre-check attachment (Task 4).

Mirrors tests/test_support_program_payload.py's shape:
  - flag off              -> no `life_stress` key; `life_stress_enabled` falsy
  - flag on, no reco      -> no `life_stress` key; `life_stress_enabled` present
  - flag on + reco        -> `life_stress` present, verbatim from _life_stress_for
                              plus a `selected` key (Task 4: saved slugs, or [] if none)
  - best-effort             a builder error never breaks the rest of the payload

Task 3 (curation override, portal only): a practitioner's curation for this client
replaces the auto-pool inside `_life_stress_for` itself (block gets curated=True).
Those tests exercise the REAL `_life_stress_for` / `life_stress_curation.apply` --
only `life_stress.recommend` is monkeypatched, to make the base pool deterministic.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_portal as cp
from dashboard import life_stress_curation
from dashboard import life_stress_selection

EMAIL = "lscaregiver@example.com"

RECO = {
    "label": "Life Stress",
    "patterns": [{"emotion": "Fear", "score": 1.0}],
    "items": [{"name": "Mimulus Flower Essence",
               "url": "/begin/product/mimulus-flower-essence-in-terrain-restore",
               "note": "for the fear pattern in your scan"}],
}

POOL = {
    "label": "Life Stress",
    "patterns": [{"emotion": "Fear", "score": 1.0}, {"emotion": "Grief", "score": 0.6}],
    "items": [{"slug": "mimulus-flower-essence-in-terrain-restore",
               "name": "Mimulus Flower Essence",
               "url": "/begin/product/mimulus-flower-essence-in-terrain-restore",
               "note": "for the fear pattern in your scan"},
              {"slug": "honeysuckle-flower-essence-in-terrain-restore",
               "name": "Honeysuckle Flower Essence",
               "url": "/begin/product/honeysuckle-flower-essence-in-terrain-restore",
               "note": "for the grief pattern in your scan"}],
}


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def app_env(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.delenv("LIFE_STRESS_ENABLED", raising=False)
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        token, _pid = cp.upsert_portal(cx, EMAIL, "Caregiver", {})
    client = app.app.test_client()
    return app, client, token


def test_flag_off_no_life_stress_key(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setattr(app, "_life_stress_for", lambda email: dict(RECO))
    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" not in j
    assert not j.get("life_stress_enabled")


def test_flag_on_enabled_flag_always_present(app_env, monkeypatch):
    app, client, token = app_env
    j = client.get(f"/api/portal/{token}").get_json()
    assert not j.get("life_stress_enabled")

    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    j = client.get(f"/api/portal/{token}").get_json()
    assert j["life_stress_enabled"] is True


def test_flag_on_with_recommendation_returns_block(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app, "_life_stress_for", lambda email: dict(RECO))

    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" in j
    # verbatim from _life_stress_for, plus Task 4's "selected" (empty: no saved selection)
    assert j["life_stress"] == {**RECO, "selected": []}


def test_flag_on_no_recommendation_no_key(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app, "_life_stress_for", lambda email: None)

    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" not in j
    assert j["life_stress_enabled"] is True


def test_builder_error_does_not_break_payload(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")

    def _boom(email):
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "_life_stress_for", _boom)
    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" not in j
    assert j["life_stress_enabled"] is True
    # rest of the payload still present
    assert "email" in j or "practitioner_brand" in j


def test_saved_selection_attached_for_precheck(app_env, monkeypatch):
    """Task 4: a client with a stored selection gets payload["life_stress"]["selected"]
    equal to the stored slugs, and each items[*] carries a slug (for the card to match
    against)."""
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app, "_life_stress_for", lambda email: dict(POOL))

    stored = ["mimulus-flower-essence-in-terrain-restore"]
    with sqlite3.connect(app.LOG_DB) as cx:
        life_stress_selection.set(cx, EMAIL, stored)

    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" in j
    assert j["life_stress"]["selected"] == stored
    for item in j["life_stress"]["items"]:
        assert item.get("slug")


def test_no_saved_selection_selected_is_empty(app_env, monkeypatch):
    """No stored selection -> selected is [] (present key, nothing pre-checked), and a
    builder-error path never sets it (covered by test_builder_error_does_not_break_payload
    where life_stress is absent entirely)."""
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app, "_life_stress_for", lambda email: dict(POOL))

    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" in j
    assert j["life_stress"]["selected"] == []


def test_curated_client_gets_prescription_not_pool(app_env, monkeypatch):
    """Task 3: a practitioner curation for this client overrides the auto-pool inside
    the REAL `_life_stress_for` -- the portal payload shows the curated items with
    curated=True. Only `life_stress.recommend` is monkeypatched (deterministic pool);
    `life_stress_curation.apply` runs for real against the seeded curation."""
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app.life_stress, "recommend", lambda email, today: dict(POOL))

    with sqlite3.connect(app.LOG_DB) as cx:
        life_stress_curation.set(cx, EMAIL, "pract-1",
                                  ["honeysuckle-flower-essence-in-terrain-restore"],
                                  "practitioner's note")

    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" in j
    block = j["life_stress"]
    assert block["curated"] is True
    assert [it["slug"] for it in block["items"]] == ["honeysuckle-flower-essence-in-terrain-restore"]
    assert block["items"][0]["note"] == "practitioner's note"


def test_uncurated_client_keeps_phase1_shape(app_env, monkeypatch):
    """No curation row for this client -> the real `_life_stress_for` returns the
    recommend() block unchanged: no `curated` key (Phase 1 shape preserved)."""
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app.life_stress, "recommend", lambda email, today: dict(POOL))

    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" in j
    assert "curated" not in j["life_stress"]
    assert [it["slug"] for it in j["life_stress"]["items"]] == \
        [it["slug"] for it in POOL["items"]]
