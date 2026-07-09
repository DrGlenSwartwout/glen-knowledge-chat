"""The Options & Pricing card must offer the Family Plan once it's live.

Today the card ends with "No monthly subscription, and nothing you need to sign
up for." — literally the opposite of what a caregiver like Sharon is asking to
buy. The plan appears only behind FAMILY_PLAN_ENABLED, and its price comes from
family_plan.PLAN, never hardcoded in the payload or the HTML.
"""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import family_plan as fp
from dashboard import household as hh


CAREGIVER = "caregiver@example.com"
PET = "pet@example.com"


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


@pytest.fixture()
def app_db(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    with sqlite3.connect(tmp_db) as cx:
        hh.init_household_tables(cx)
        fp.init_family_plan_table(cx)
        hh.add_member(cx, CAREGIVER, PET, relationship="pet")
    return app


def test_flag_off_no_family_plan_block(app_db, monkeypatch):
    monkeypatch.delenv("FAMILY_PLAN_ENABLED", raising=False)
    opts = app_db._portal_options_for(CAREGIVER)
    assert opts is not None
    assert "family_plan" not in opts


def test_flag_on_offers_the_plan_at_147_against_a_197_anchor(app_db, monkeypatch):
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    opts = app_db._portal_options_for(CAREGIVER)
    plan = opts["family_plan"]
    assert plan["price_cents"] == fp.PLAN["amount_cents"] == 14700
    assert plan["value_cents"] == fp.PLAN["value_cents"] == 19700


def test_a_caregiver_without_the_plan_is_shown_as_inactive(app_db, monkeypatch):
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    assert app_db._portal_options_for(CAREGIVER)["family_plan"]["active"] is False


def test_a_caregiver_holding_the_plan_is_shown_as_active(app_db, tmp_db, monkeypatch):
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    with sqlite3.connect(tmp_db) as cx:
        fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    assert app_db._portal_options_for(CAREGIVER)["family_plan"]["active"] is True


def test_a_covered_member_is_shown_as_active_too(app_db, tmp_db, monkeypatch):
    """Sasha's own page should not try to sell her a plan she is already on."""
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    with sqlite3.connect(tmp_db) as cx:
        fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    assert app_db._portal_options_for(PET)["family_plan"]["active"] is True


def test_family_plan_failure_does_not_break_the_options_card(app_db, monkeypatch):
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")

    def _boom(*a, **k):
        raise RuntimeError("db gone")

    monkeypatch.setattr(fp, "covers", _boom)
    opts = app_db._portal_options_for(CAREGIVER)
    assert opts is not None
    assert opts["analysis_cents"] >= 0
    assert "family_plan" not in opts
