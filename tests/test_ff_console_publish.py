"""Slice 3c.1: the real _ff_covered entitlement predicate + the console
review/publish endpoints for FF match drafts.

_ff_covered replaces the Slice 3b stub (always False) with the real gate:
paid Biofield Analysis on record, OR an active membership, OR (when
FAMILY_PLAN_ENABLED) a caregiver's Family Plan covering this email. Fail-closed
on any error.

The console endpoints let Glen review generated FF-match drafts and publish
them (optionally editing items first, e.g. adding a `dosing` field) — mirrors
the existing /api/console/* console-key gating (api_console_analysis_requests).
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import family_plan as fp
from dashboard import household as hh
from dashboard import ff_match_drafts as ffd

CAREGIVER = "caregiver@example.com"
MEMBER = "member@example.com"
STRANGER = "someone-else@example.com"
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
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    app._init_membership_tables()  # memberships table lives in LOG_DB too
    with sqlite3.connect(tmp_db) as cx:
        hh.init_household_tables(cx)
        fp.init_family_plan_table(cx)
        ffd.init_table(cx)
    return app


# ---------------------------------------------------------------------------
# _ff_covered
# ---------------------------------------------------------------------------

def test_ff_covered_true_via_family_plan_caregiver(app_mod, tmp_db):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        hh.add_member(cx, CAREGIVER, MEMBER, relationship="spouse")
        fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
        assert app_mod._ff_covered(cx, MEMBER) is True


def test_ff_covered_false_for_bare_unknown_email(app_mod, tmp_db):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        assert app_mod._ff_covered(cx, STRANGER) is False


def test_ff_covered_false_when_family_plan_flag_off(app_mod, tmp_db, monkeypatch):
    monkeypatch.delenv("FAMILY_PLAN_ENABLED", raising=False)
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        hh.add_member(cx, CAREGIVER, MEMBER, relationship="spouse")
        fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
        assert app_mod._ff_covered(cx, MEMBER) is False


# ---------------------------------------------------------------------------
# console endpoints
# ---------------------------------------------------------------------------

def _seed_draft(tmp_db, email="ffclient@example.com", scan_date="2026-07-02"):
    items = [{"item_code": "X1", "priority_rank": 1, "label": "Formula X"}]
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        ffd.init_table(cx)
        ffd.get_or_create(cx, email, scan_date, lambda: items)
    return email, scan_date


def test_list_drafts_requires_console_key(app_mod, tmp_db):
    _seed_draft(tmp_db)
    client = app_mod.app.test_client()
    r = client.get("/api/console/ff-match-drafts")
    assert r.status_code in (401, 403)


def test_list_drafts_returns_seeded_draft(app_mod, tmp_db):
    email, scan_date = _seed_draft(tmp_db)
    client = app_mod.app.test_client()
    r = client.get("/api/console/ff-match-drafts", headers=HDRS)
    assert r.status_code == 200
    drafts = r.get_json()["drafts"]
    assert any(d["email"] == email and d["scan_date"] == scan_date for d in drafts)


def test_publish_requires_console_key(app_mod, tmp_db):
    email, scan_date = _seed_draft(tmp_db)
    client = app_mod.app.test_client()
    r = client.post("/api/console/ff-match-drafts/publish",
                     json={"email": email, "scan_date": scan_date})
    assert r.status_code in (401, 403)


def test_publish_sets_items_then_publishes(app_mod, tmp_db):
    email, scan_date = _seed_draft(tmp_db)
    edited_items = [{"item_code": "X1", "priority_rank": 1, "label": "Formula X",
                      "dosing": "2 caps twice daily"}]
    client = app_mod.app.test_client()
    r = client.post("/api/console/ff-match-drafts/publish", headers=HDRS,
                     json={"email": email, "scan_date": scan_date, "items": edited_items})
    assert r.status_code == 200
    assert r.get_json() == {"published": True}
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        draft = ffd.get(cx, email, scan_date)
    assert draft["status"] == "published"
    assert draft["items"] == edited_items
    assert draft["items"][0]["dosing"] == "2 caps twice daily"


def test_publish_without_items_keeps_existing_items(app_mod, tmp_db):
    email, scan_date = _seed_draft(tmp_db)
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        before = ffd.get(cx, email, scan_date)["items"]
    client = app_mod.app.test_client()
    r = client.post("/api/console/ff-match-drafts/publish", headers=HDRS,
                     json={"email": email, "scan_date": scan_date})
    assert r.status_code == 200
    assert r.get_json() == {"published": True}
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        draft = ffd.get(cx, email, scan_date)
    assert draft["status"] == "published"
    assert draft["items"] == before
