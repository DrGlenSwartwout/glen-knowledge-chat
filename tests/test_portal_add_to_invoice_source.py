"""Task 3 (Phase 2a): portal add-to-invoice buttons stamp their recommendation
source on the order line, so the console can later show/filter by where a
line came from. ADDITIVE only -- one `source` key per built line dict, no
change to slug/qty/pricing/upsert_order call.

- FF-matches add-to-invoice (tests/test_ff_add_to_invoice.py harness) -> "scan"
- Support-program add-to-invoice (tests/test_support_program_add_to_invoice.py
  harness) -> "intake"
"""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_conditions as cc
from dashboard import client_portal as cp
from dashboard import client_prices as cprices
from dashboard import condition_programs as prog
from dashboard import family_plan as fp
from dashboard import ff_match_drafts as ffd
from dashboard import household as hh
from dashboard import orders as ord_mod

FF_SLUG = "mucosa-syntropy-powder"

CAREGIVER = "srccaregiver@example.com"
MEMBER = "srcmember@example.com"
TAGGED = "srctagged@example.com"

WET_AMD_ITEMS = [
    {"slug": FF_SLUG, "name": "Mucosa Syntropy Powder", "dose": "1 or more/day"},
    {"slug": "scar-solve", "name": "Scar Solve", "note": "Add for brunescent cataracts"},
]


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


def _orders_rows(tmp_db):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        return [dict(r) for r in cx.execute("SELECT * FROM orders").fetchall()]


# ---------------------------------------------------------------------------
# FF-matches add-to-invoice -> every built line carries source="scan"
# ---------------------------------------------------------------------------

@pytest.fixture()
def ff_app_mod(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    app._init_membership_tables()
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        hh.init_household_tables(cx)
        fp.init_family_plan_table(cx)
        ffd.init_table(cx)
        ord_mod.init_orders_table(cx)
        cprices.init_table(cx)
    app._migrate_orders_portal_published()
    return app


def test_ff_add_to_invoice_line_tagged_scan(ff_app_mod, tmp_db):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        hh.add_member(cx, CAREGIVER, MEMBER, relationship="")
        fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
        token, _pid = cp.upsert_portal(cx, MEMBER, "Client", {})
        ffd.get_or_create(cx, MEMBER, "", lambda: [
            {"name": "Mucosa Syntropy Powder", "slug": FF_SLUG,
             "url": "/begin/product/mucosa-syntropy-powder", "meaning": "gut lining support"},
        ])
        ffd.publish(cx, MEMBER, "")

    client = ff_app_mod.app.test_client()
    r = client.post(f"/api/portal/{token}/ff-matches/add-to-invoice")
    assert r.status_code == 200

    rows = _orders_rows(tmp_db)
    assert len(rows) == 1
    items = json.loads(rows[0]["items_json"])
    assert items, "expected at least one built line"
    assert all(li.get("source") == "scan" for li in items)
    # additive only -- existing keys still present, untouched
    assert items[0]["slug"] == FF_SLUG
    assert items[0]["qty"] == 1
    assert "unit_cents" in items[0] and "line_cents" in items[0]


# ---------------------------------------------------------------------------
# Support-program add-to-invoice -> every built line carries source="intake"
# ---------------------------------------------------------------------------

@pytest.fixture()
def sp_app_mod(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setenv("SUPPORT_PROGRAMS_ENABLED", "1")
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        hh.init_household_tables(cx)
        prog.init_table(cx)
        cc.init_table(cx)
        ord_mod.init_orders_table(cx)
        cprices.init_table(cx)
    app._migrate_orders_portal_published()
    return app


def test_support_program_add_to_invoice_line_tagged_intake(sp_app_mod, tmp_db):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        prog.init_table(cx)
        prog.upsert(cx, "wet-amd", "Wet AMD", False, WET_AMD_ITEMS, modifiers=None)
        cc.set(cx, TAGGED, "wet-amd", "test")
        token, _pid = cp.upsert_portal(cx, TAGGED, "Client", {})

    client = sp_app_mod.app.test_client()
    r = client.post(f"/api/portal/{token}/support-program/add-to-invoice")
    assert r.status_code == 200

    rows = _orders_rows(tmp_db)
    assert len(rows) == 1
    items = json.loads(rows[0]["items_json"])
    assert items, "expected at least one built line"
    assert all(li.get("source") == "intake" for li in items)
    slugs = [i["slug"] for i in items]
    assert slugs == [FF_SLUG, "scar-solve"]
