"""A dependent's own portal must not gate on a Terms agreement it can never give.

A pet, an infant, a minor: each has its own portal token, handed to the caregiver. Opening
it asks "has this dependent agreed?" — permanently false. The caregiver's agreement covers
them. Derived at render time from the household link, so it never goes stale and a standalone
adult is byte-identical. NOT stamped on the dependent's own record: they did not agree.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import household as hh

CARE = "caregiver@example.com"
PET = "pet@example.com"
ADULT = "standalone@example.com"


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def app_db(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    import begin_funnel
    with sqlite3.connect(tmp_db) as cx:
        begin_funnel.init_journey_tables(cx)         # journey_state, for is_member/record_unlock
        hh.init_household_tables(cx)
        hh.add_member(cx, CARE, PET, "Pet", "pet")   # a dependent under the caregiver
    return app


def _agree(app, email):
    """Give this email a real journey_state TOS acceptance."""
    import begin_funnel
    with sqlite3.connect(app.LOG_DB) as cx:
        begin_funnel.record_unlock(cx, session_id="s-" + email, trigger="tos",
                                   email=email, tos=True, tos_version=app.BEGIN_TOS_VERSION)


def test_flag_off_a_dependent_is_not_covered_by_the_caregiver(app_db, monkeypatch):
    monkeypatch.delenv("DEPENDENT_TOS_ENABLED", raising=False)
    _agree(app_db, CARE)
    assert app_db._portal_tos_agreed(PET) is False          # unchanged: the pet never agreed


def test_flag_on_a_dependent_is_covered_when_the_caregiver_agreed(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    _agree(app_db, CARE)
    assert app_db._portal_tos_agreed(PET) is True


def test_flag_on_but_the_caregiver_has_not_agreed(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    assert app_db._portal_tos_agreed(PET) is False


def test_a_dependent_who_revoked_consent_must_agree_themselves(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    _agree(app_db, CARE)
    with sqlite3.connect(app_db.LOG_DB) as cx:
        hh.set_share_consent(cx, CARE, PET, 0)
    assert app_db._portal_tos_agreed(PET) is False


def test_the_dependent_agreeing_directly_still_works(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    _agree(app_db, PET)
    assert app_db._portal_tos_agreed(PET) is True


def test_a_standalone_adult_is_unchanged(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    assert app_db._portal_tos_agreed(ADULT) is False
    _agree(app_db, ADULT)
    assert app_db._portal_tos_agreed(ADULT) is True


def test_coverage_does_not_stamp_the_dependents_own_record(app_db, monkeypatch):
    """The dependent must NOT appear in the compliance set — they did not agree."""
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    _agree(app_db, CARE)
    assert app_db._portal_tos_agreed(PET) is True
    with sqlite3.connect(app_db.LOG_DB) as cx:
        agreed = app_db._tos_agreed_emails(cx)
    assert CARE in agreed
    assert PET not in agreed


def test_a_blown_lookup_falls_back_to_the_dependents_own_agreement(app_db, monkeypatch):
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    monkeypatch.setattr(hh, "caregivers_for",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
    _agree(app_db, PET)
    assert app_db._portal_tos_agreed(PET) is True            # its own agreement still counts
    # and a pet that never agreed, with the lookup broken, stays gated (fail-closed)
    assert app_db._portal_tos_agreed("other-pet@example.com") is False
