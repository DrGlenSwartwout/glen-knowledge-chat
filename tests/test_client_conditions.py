"""Slice 3 of Condition Support Programs.

dashboard/client_conditions.py: the operator-override store for a client's
eye-condition support-program key (email-keyed, lowercased). Also covers the
pure tag->key normalizer (_condition_key_from_tags) and the resolver
(_client_condition_for), both defined in app.py, since neither touches
Flask routing.
"""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_conditions as cc


def _app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


def _cx(tmp_db):
    cx = sqlite3.connect(tmp_db)
    cx.row_factory = sqlite3.Row
    return cx


# ---------------------------------------------------------------------------
# Store CRUD
# ---------------------------------------------------------------------------

def test_init_table_creates_schema(tmp_db):
    cx = _cx(tmp_db)
    cc.init_table(cx)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(client_conditions)")}
    assert cols == {"email", "condition_key", "set_by", "updated_at"}


def test_get_returns_none_when_unset(tmp_db):
    cx = _cx(tmp_db)
    cc.init_table(cx)
    assert cc.get(cx, "nobody@x.com") is None


def test_set_then_get_roundtrips(tmp_db):
    cx = _cx(tmp_db)
    cc.init_table(cx)
    cc.set(cx, "Jane@Example.com", "wet-amd", "glen")
    assert cc.get(cx, "jane@example.com") == "wet-amd"


def test_get_lowercases_email(tmp_db):
    cx = _cx(tmp_db)
    cc.init_table(cx)
    cc.set(cx, "jane@example.com", "dry-eye", "rae")
    assert cc.get(cx, "JANE@EXAMPLE.COM") == "dry-eye"


def test_set_overwrites_existing_override(tmp_db):
    cx = _cx(tmp_db)
    cc.init_table(cx)
    cc.set(cx, "jane@example.com", "dry-eye", "rae")
    cc.set(cx, "jane@example.com", "wet-amd", "glen")
    assert cc.get(cx, "jane@example.com") == "wet-amd"


def test_clear_removes_override(tmp_db):
    cx = _cx(tmp_db)
    cc.init_table(cx)
    cc.set(cx, "jane@example.com", "wet-amd", "glen")
    cc.clear(cx, "jane@example.com")
    assert cc.get(cx, "jane@example.com") is None


def test_clear_is_a_noop_when_nothing_set(tmp_db):
    cx = _cx(tmp_db)
    cc.init_table(cx)
    cc.clear(cx, "nobody@x.com")  # must not raise
    assert cc.get(cx, "nobody@x.com") is None


# ---------------------------------------------------------------------------
# Tag -> condition-key normalizer (_condition_key_from_tags in app.py)
# ---------------------------------------------------------------------------

UNAMBIGUOUS_CASES = [
    (["wet amd"], "wet-amd"),
    (["wet-amd"], "wet-amd"),
    (["neovascular amd"], "wet-amd"),
    (["exudative amd"], "wet-amd"),
    (["dry amd"], "dry-amd"),
    (["dry-amd"], "dry-amd"),
    (["atrophic amd"], "dry-amd"),
    (["dry eye"], "dry-eye"),
    (["dry-eye"], "dry-eye"),
    (["retinitis pigmentosa"], "retinitis-pigmentosa"),
    (["rp"], "retinitis-pigmentosa"),
    (["diabetic retinopathy"], "diabetic-retinopathy"),
    (["dr"], "diabetic-retinopathy"),
    (["psc"], "psc-cataract"),
    (["psc cataract"], "psc-cataract"),
    (["posterior subcapsular"], "psc-cataract"),
    (["senile cataract"], "senile-cataract"),
    (["age-related cataract"], "senile-cataract"),
    (["nuclear cataract"], "senile-cataract"),
    (["glaucoma elevated"], "glaucoma-elevated-iop"),
    (["ocular hypertension"], "glaucoma-elevated-iop"),
    (["high iop"], "glaucoma-elevated-iop"),
    (["elevated iop"], "glaucoma-elevated-iop"),
    (["normal tension glaucoma"], "glaucoma-normal-iop"),
    (["normal-tension"], "glaucoma-normal-iop"),
    (["low iop glaucoma"], "glaucoma-normal-iop"),
    (["normal iop"], "glaucoma-normal-iop"),
]


@pytest.mark.parametrize("tags,expected", UNAMBIGUOUS_CASES)
def test_condition_key_from_tags_unambiguous_mappings(tags, expected):
    app = _app()
    assert app._condition_key_from_tags(tags) == expected


def test_condition_key_from_tags_bare_glaucoma_is_ambiguous():
    app = _app()
    assert app._condition_key_from_tags(["glaucoma"]) is None


def test_condition_key_from_tags_bare_cataract_is_ambiguous():
    app = _app()
    assert app._condition_key_from_tags(["cataract"]) is None


def test_condition_key_from_tags_unrelated_tag_returns_none():
    app = _app()
    assert app._condition_key_from_tags(["fatty liver"]) is None


def test_condition_key_from_tags_empty_list_returns_none():
    app = _app()
    assert app._condition_key_from_tags([]) is None


def test_condition_key_from_tags_case_insensitive():
    app = _app()
    assert app._condition_key_from_tags(["WET AMD"]) == "wet-amd"
    assert app._condition_key_from_tags(["Wet-Amd"]) == "wet-amd"


def test_condition_key_from_tags_pb_prefix_stripped():
    app = _app()
    assert app._condition_key_from_tags(["pb:wet-amd"]) == "wet-amd"


def test_condition_key_from_tags_first_unambiguous_match_wins():
    app = _app()
    # "cataract" alone is ambiguous and must be skipped in favor of the
    # unambiguous "dry eye" tag that follows it.
    assert app._condition_key_from_tags(["cataract", "dry eye"]) == "dry-eye"


def test_condition_key_from_tags_extra_whitespace_variants():
    app = _app()
    assert app._condition_key_from_tags(["  wet   amd  "]) == "wet-amd"


# ---------------------------------------------------------------------------
# Resolver (_client_condition_for in app.py): override wins, else auto-detect
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_mod(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    app._init_people_table()
    return app


def _seed_person(db, email, conditions=None, tags=None):
    with sqlite3.connect(db) as cx:
        cx.execute(
            "INSERT INTO people (email, conditions, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?)",
            (email, json.dumps(conditions or []), json.dumps(tags or []), "", ""))
        cx.commit()


def test_client_condition_for_override_wins_over_auto_detected_tag(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", conditions=["Wet AMD"])
    with sqlite3.connect(tmp_db) as cx:
        from dashboard import client_conditions as _cc
        _cc.init_table(cx)
        _cc.set(cx, "jane@example.com", "dry-eye", "glen")
    assert app_mod._client_condition_for("jane@example.com") == "dry-eye"


def test_client_condition_for_auto_detects_from_conditions_when_no_override(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", conditions=["Wet AMD"])
    assert app_mod._client_condition_for("jane@example.com") == "wet-amd"


def test_client_condition_for_auto_detects_from_tags_when_no_override(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", tags=["pb:dry-eye"])
    assert app_mod._client_condition_for("jane@example.com") == "dry-eye"


def test_client_condition_for_ambiguous_tag_returns_none(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", conditions=["glaucoma"])
    assert app_mod._client_condition_for("jane@example.com") is None


def test_client_condition_for_no_tags_returns_none(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com")
    assert app_mod._client_condition_for("jane@example.com") is None


def test_client_condition_for_unknown_email_returns_none(app_mod):
    assert app_mod._client_condition_for("nobody@x.com") is None


def test_client_condition_for_empty_email_returns_none(app_mod):
    assert app_mod._client_condition_for("") is None
