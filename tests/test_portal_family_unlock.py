"""_portal_biofield_unlocked must honour a caregiver's family plan.

A caregiver paid; the pet in their care (a household member with a dead email
address) never paid anything. Before this, the paywall asked whether the PET had
paid, and blurred the pet's report on the caregiver's own screen. The plan the
caregiver holds is what un-blurs the household.

Flag off => byte-identical to the old per-email gate.
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
def app_with_household(tmp_db, monkeypatch):
    """A paid-gate-on app whose DB has Karin -> Sasha linked, nobody paid."""
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setenv("PORTAL_PAID_GATE_ENABLED", "1")
    with sqlite3.connect(tmp_db) as cx:
        hh.init_household_tables(cx)
        fp.init_family_plan_table(cx)
        # The two stores the gate consults before the family branch. Prod has
        # both; without them the gate's except-clause swallows the lookup and
        # every assertion below would pass for the wrong reason.
        cx.execute("CREATE TABLE IF NOT EXISTS memberships ("
                   "id TEXT PRIMARY KEY, email TEXT NOT NULL, granted_at TEXT, "
                   "expires_at TEXT, granted_by TEXT, source TEXT)")
        cx.execute("CREATE TABLE IF NOT EXISTS biofield_readiness ("
                   "email TEXT PRIMARY KEY, paid_at TEXT)")
        hh.add_member(cx, CAREGIVER, PET, relationship="pet")
    return app


def _grant_plan(tmp_db, email):
    with sqlite3.connect(tmp_db) as cx:
        fp.activate(cx, email, next_charge_at="2026-08-09")


def test_flag_off_a_caregiver_plan_does_not_unlock_the_member(app_with_household, tmp_db, monkeypatch):
    monkeypatch.delenv("FAMILY_PLAN_ENABLED", raising=False)
    _grant_plan(tmp_db, CAREGIVER)
    assert app_with_household._portal_biofield_unlocked(PET) is False


def test_flag_on_caregiver_plan_unlocks_the_member(app_with_household, tmp_db, monkeypatch):
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    _grant_plan(tmp_db, CAREGIVER)
    assert app_with_household._portal_biofield_unlocked(PET) is True


def test_flag_on_without_a_plan_the_member_stays_blurred(app_with_household, monkeypatch):
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    assert app_with_household._portal_biofield_unlocked(PET) is False


def test_flag_on_plan_unlocks_the_caregiver_own_report(app_with_household, tmp_db, monkeypatch):
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    _grant_plan(tmp_db, CAREGIVER)
    assert app_with_household._portal_biofield_unlocked(CAREGIVER) is True


def test_flag_on_revoked_consent_still_unlocks_the_member(app_with_household, tmp_db, monkeypatch):
    """Entitlement is decoupled from report-sharing consent (Issue 3):
    _portal_biofield_unlocked delegates straight to family_plan.covers(), which no
    longer conditions on share_consent. Revoking consent still gates VIEWING
    (household.can_view) — a separate axis — but must not re-blur a household
    member's own paid-plan coverage."""
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    _grant_plan(tmp_db, CAREGIVER)
    with sqlite3.connect(tmp_db) as cx:
        hh.set_share_consent(cx, CAREGIVER, PET, 0)
    assert app_with_household._portal_biofield_unlocked(PET) is True


def test_family_lookup_failure_fails_closed(app_with_household, monkeypatch):
    """A blown household lookup must blur, never expose."""
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")

    def _boom(*a, **k):
        raise RuntimeError("db gone")

    monkeypatch.setattr(fp, "covers", _boom)
    assert app_with_household._portal_biofield_unlocked(PET) is False


def test_paid_gate_off_still_unlocks_everyone(app_with_household, monkeypatch):
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    monkeypatch.setenv("PORTAL_PAID_GATE_ENABLED", "0")
    assert app_with_household._portal_biofield_unlocked(PET) is True
