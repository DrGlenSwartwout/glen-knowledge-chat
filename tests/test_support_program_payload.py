"""GET /api/portal/<token> — the `support_program` payload key (Slice 4a).

Mirrors tests/test_ff_matches_payload.py's shape:
  - flag off               -> no `support_program` key; `support_programs_enabled` falsy
  - flag on, no condition  -> no `support_program` key
  - flag on + condition    -> `support_program` present: condition_key/label/
                              consult_recommended + items (name/url + dose/note/alts
                              where seeded), preserving item order
  - ?member=                -> the MEMBER's condition, not the caregiver's
  - best-effort              a builder error never breaks the rest of the payload
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_conditions as cc
from dashboard import client_portal as cp
from dashboard import condition_programs as prog
from dashboard import household as hh

EMAIL = "spcaregiver@example.com"
MEMBER = "spmember@example.com"

WET_AMD_ITEMS = [
    {"slug": "angiogenx", "name": "AngiogenX", "dose": "1 or more/day"},
    {"slug": "scar-solve", "name": "Scar Solve", "note": "Add for brunescent cataracts"},
    {"slug": "scar-silk", "name": "Scar Silk",
     "alts": [{"slug": "clear-the-way", "name": "Clear the Way"}]},
]

MEMBER_ITEMS = [
    {"slug": "wholomega", "name": "WholOmega", "dose": "4 capsules/day"},
]


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
    monkeypatch.delenv("SUPPORT_PROGRAMS_ENABLED", raising=False)
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        hh.init_household_tables(cx)
        prog.init_table(cx)
        token, _pid = cp.upsert_portal(cx, EMAIL, "Caregiver", {})
    client = app.app.test_client()
    return app, client, token


def _seed_program(app, key, label, items, consult_recommended=False):
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        prog.init_table(cx)
        prog.upsert(cx, key, label, consult_recommended, items)


def _seed_condition(app, email, key):
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        cc.init_table(cx)
        cc.set(cx, email, key, "test")


def test_flag_off_no_support_program_key(app_env):
    app, client, token = app_env
    _seed_program(app, "wet-amd", "Wet AMD", WET_AMD_ITEMS, consult_recommended=True)
    _seed_condition(app, EMAIL, "wet-amd")
    j = client.get(f"/api/portal/{token}").get_json()
    assert "support_program" not in j
    assert not j.get("support_programs_enabled")


def test_flag_on_enabled_flag_always_present(app_env, monkeypatch):
    app, client, token = app_env
    j = client.get(f"/api/portal/{token}").get_json()
    assert not j.get("support_programs_enabled")

    monkeypatch.setenv("SUPPORT_PROGRAMS_ENABLED", "1")
    j = client.get(f"/api/portal/{token}").get_json()
    assert j["support_programs_enabled"] is True


def test_flag_on_with_condition_returns_full_shape(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("SUPPORT_PROGRAMS_ENABLED", "1")
    _seed_program(app, "wet-amd", "Wet AMD", WET_AMD_ITEMS, consult_recommended=True)
    _seed_condition(app, EMAIL, "wet-amd")

    j = client.get(f"/api/portal/{token}").get_json()
    assert "support_program" in j
    sp = j["support_program"]
    assert sp["condition_key"] == "wet-amd"
    assert sp["label"] == "Wet AMD"
    assert sp["consult_recommended"] is True

    items = sp["items"]
    assert [it["name"] for it in items] == ["AngiogenX", "Scar Solve", "Scar Silk"]

    it0 = items[0]
    assert it0["url"] == "/begin/product/angiogenx"
    assert it0["dose"] == "1 or more/day"
    assert "note" not in it0
    assert "alts" not in it0

    it1 = items[1]
    assert it1["note"] == "Add for brunescent cataracts"
    assert "dose" not in it1

    it2 = items[2]
    assert it2["alts"] == [{"name": "Clear the Way", "url": "/begin/product/clear-the-way"}]


def test_flag_on_no_condition_no_key(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("SUPPORT_PROGRAMS_ENABLED", "1")
    _seed_program(app, "wet-amd", "Wet AMD", WET_AMD_ITEMS, consult_recommended=True)
    # No client_conditions override and no people row -> unresolved condition
    j = client.get(f"/api/portal/{token}").get_json()
    assert "support_program" not in j


def test_member_sees_their_own_condition_not_the_caregivers(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("SUPPORT_PROGRAMS_ENABLED", "1")
    monkeypatch.setenv("HOUSEHOLD_VIEW_ENABLED", "1")
    _seed_program(app, "wet-amd", "Wet AMD", WET_AMD_ITEMS, consult_recommended=True)
    _seed_program(app, "dry-eye", "Dry Eye", MEMBER_ITEMS, consult_recommended=False)
    _seed_condition(app, EMAIL, "wet-amd")
    _seed_condition(app, MEMBER, "dry-eye")
    with sqlite3.connect(app.LOG_DB) as cx:
        hh.add_member(cx, EMAIL, MEMBER, "Member", "dependent")

    j = client.get(f"/api/portal/{token}?member={MEMBER}").get_json()
    assert "support_program" in j
    sp = j["support_program"]
    assert sp["condition_key"] == "dry-eye"
    assert [it["name"] for it in sp["items"]] == ["WholOmega"]


def test_builder_error_does_not_break_payload(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("SUPPORT_PROGRAMS_ENABLED", "1")
    _seed_program(app, "wet-amd", "Wet AMD", WET_AMD_ITEMS, consult_recommended=True)
    _seed_condition(app, EMAIL, "wet-amd")

    def _boom(email):
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "_support_program_for", _boom)
    j = client.get(f"/api/portal/{token}").get_json()
    assert "support_program" not in j
    assert j["support_programs_enabled"] is True
    # rest of the payload still present
    assert "email" in j or "practitioner_brand" in j
