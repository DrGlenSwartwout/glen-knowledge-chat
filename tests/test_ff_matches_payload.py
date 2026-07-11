"""GET /api/portal/<token> — the `ff_matches` payload key (Slice 3b, Task 5).

The GET side NEVER generates a draft — generation only happens via the POST
button (/api/portal/<token>/ff-matches). So:
  - flag off              -> no `ff_matches` key (byte-identical payload)
  - flag on, no draft yet -> no `ff_matches` key (byte-identical payload)
  - flag on + a draft     -> `ff_matches` present, dosing stripped unless
                             covered AND reviewed (default _ff_covered stub
                             is always False, so dosing is stripped even for
                             a published draft)
  - ?member=               -> the MEMBER's draft, not the caregiver's
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_portal as cp
from dashboard import ff_match_drafts as ffd
from dashboard import household as hh

EMAIL = "ffcaregiver@example.com"
MEMBER = "ffmember@example.com"

CAREGIVER_ITEMS = [
    {"name": "Caregiver FF", "slug": "caregiver-ff", "url": "/begin/product/caregiver-ff",
     "meaning": "supports the caregiver", "score": 0.9, "dosing": "2 caps daily"},
]
MEMBER_ITEMS = [
    {"name": "Member FF", "slug": "member-ff", "url": "/begin/product/member-ff",
     "meaning": "supports the member", "score": 0.8, "dosing": "1 cap daily"},
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
    monkeypatch.delenv("FF_MATCHES_ENABLED", raising=False)
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        hh.init_household_tables(cx)
        ffd.init_table(cx)
        token, _pid = cp.upsert_portal(cx, EMAIL, "Caregiver", {})
    client = app.app.test_client()
    return app, client, token


def _seed_draft(app, email, items, status="published"):
    scan_date = app._current_scan_date_for(email)  # "" when no scans — that's fine, it's the key
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        ffd.init_table(cx)
        ffd.get_or_create(cx, email, scan_date, lambda: items)
        if status == "published":
            ffd.publish(cx, email, scan_date)
    return scan_date


def test_flag_off_no_ff_matches_key(app_env):
    app, client, token = app_env
    _seed_draft(app, EMAIL, CAREGIVER_ITEMS)
    j = client.get(f"/api/portal/{token}").get_json()
    assert "ff_matches" not in j


def test_flag_on_no_draft_no_key(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    # no draft seeded
    j = client.get(f"/api/portal/{token}").get_json()
    assert "ff_matches" not in j


def test_flag_on_with_draft_strips_dosing_not_covered(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    _seed_draft(app, EMAIL, CAREGIVER_ITEMS, status="published")
    j = client.get(f"/api/portal/{token}").get_json()
    assert "ff_matches" in j
    block = j["ff_matches"]
    assert block["reviewed"] is True
    assert block["covered"] is False   # default _ff_covered stub
    assert len(block["items"]) == 1
    assert all("dosing" not in it for it in block["items"])
    assert block["items"][0]["name"] == "Caregiver FF"


def test_member_sees_their_own_draft_not_the_caregivers(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    monkeypatch.setenv("HOUSEHOLD_VIEW_ENABLED", "1")
    with sqlite3.connect(app.LOG_DB) as cx:
        hh.add_member(cx, EMAIL, MEMBER, "Member", "dependent")
    _seed_draft(app, EMAIL, CAREGIVER_ITEMS, status="published")
    _seed_draft(app, MEMBER, MEMBER_ITEMS, status="published")
    j = client.get(f"/api/portal/{token}?member={MEMBER}").get_json()
    assert "ff_matches" in j
    names = [it["name"] for it in j["ff_matches"]["items"]]
    assert names == ["Member FF"]
