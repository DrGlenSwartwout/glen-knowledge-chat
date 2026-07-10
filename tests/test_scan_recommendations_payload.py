"""The card's payload. Flag-gated, member-aware, best-effort.

Keyed on the E4L SCAN date, not the published-report date — a live client's report is
filed under a date on which she has no scan, and keying on it would show her nothing.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import scan_recommendations as sr
from dashboard import household as hh

CARE = "caregiver@example.com"
PET = "pet@example.com"

ITEMS = [
    {"item_code": "BFA", "priority_rank": 1, "protocol_days": 15,
     "section": "Infoceuticals", "category": "BFA", "label": "Big Field Aligner"},
    {"item_code": "ED6", "priority_rank": 2, "protocol_days": 15,
     "section": "Infoceuticals", "category": "ED", "label": "Heart"},
    {"item_code": "ER2", "priority_rank": 3, "protocol_days": 2,
     "section": "miHealth Functions", "category": "ER", "label": "Large Intestine"},
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
def app_db(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    with sqlite3.connect(tmp_db) as cx:
        sr.init_table(cx)
        hh.init_household_tables(cx)
        sr.replace_scan(cx, CARE, "10", "2026-07-02", ITEMS)
        sr.replace_scan(cx, CARE, "20", "2026-06-13", ITEMS[:1])
        sr.replace_scan(cx, PET, "30", "2026-07-05", ITEMS[:2])
    return app


def test_flag_off_returns_nothing(app_db, monkeypatch):
    monkeypatch.delenv("SCAN_RECOMMENDATIONS_ENABLED", raising=False)
    assert app_db._scan_recommendations_for(CARE) is None


def test_flag_on_returns_the_latest_scan(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    block = app_db._scan_recommendations_for(CARE)
    assert block["scan_date"] == "2026-07-02"
    assert block["scan_dates"] == ["2026-07-02", "2026-06-13"]


def test_infoceuticals_and_mihealth_are_separated(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    b = app_db._scan_recommendations_for(CARE)
    assert [i["code"] for i in b["infoceuticals"]] == ["BFA", "ED6"]
    assert [m["code"] for m in b["mihealth"]] == ["ER2"]


def test_mihealth_rows_carry_no_order_url(app_db, monkeypatch):
    """ER/MR are device cycles. A dead order button is worse than no button."""
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    for m in app_db._scan_recommendations_for(CARE)["mihealth"]:
        assert "order_url" not in m


def test_every_infoceutical_has_a_working_order_url(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    for i in app_db._scan_recommendations_for(CARE)["infoceuticals"]:
        assert i["order_url"].startswith("/begin/product/")
        assert "remedymatch.com" not in i["order_url"]


def test_bfa_renders_glens_label(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    bfa = app_db._scan_recommendations_for(CARE)["infoceuticals"][0]
    assert bfa["label"] == "Big Field Aligner (BFA)"
    assert bfa["rank"] == 1


def test_other_codes_render_code_then_label(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    ed6 = app_db._scan_recommendations_for(CARE)["infoceuticals"][1]
    assert ed6["label"] == "ED6 Heart"


def test_an_explicit_scan_date_wins(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    b = app_db._scan_recommendations_for(CARE, scan_date="2026-06-13")
    assert b["scan_date"] == "2026-06-13"
    assert [i["code"] for i in b["infoceuticals"]] == ["BFA"]


def test_an_unknown_scan_date_falls_back_to_the_latest(app_db, monkeypatch):
    """A published-report date can name a day on which the client has no scan."""
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    b = app_db._scan_recommendations_for(CARE, scan_date="2026-07-07")
    assert b["scan_date"] == "2026-07-02"


def test_a_client_with_no_scans_returns_none(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    assert app_db._scan_recommendations_for("stranger@example.com") is None


def test_a_member_sees_their_own_scan_not_the_caregivers(app_db, monkeypatch):
    """?member= re-points email_for_reports; the card must follow it."""
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    b = app_db._scan_recommendations_for(PET)
    assert b["scan_date"] == "2026-07-05"
    assert [i["code"] for i in b["infoceuticals"]] == ["BFA", "ED6"]
    assert b["mihealth"] == []


def test_a_broken_lookup_never_breaks_the_portal(app_db, monkeypatch):
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    monkeypatch.setattr(sr, "scan_dates_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
    assert app_db._scan_recommendations_for(CARE) is None
