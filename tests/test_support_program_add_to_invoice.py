"""Slice 4b: POST /api/portal/<token>/support-program/add-to-invoice — the
MONEY-WRITE add-to-invoice action for the condition support-program card.

THE GATE IS DIFFERENT FROM FF (tests/test_ff_add_to_invoice.py): this is open
to ANY client who resolves to a support program (they've been tagged with an
eye condition) -- there is no _ff_covered / entitlement check. A client with
no resolved condition is rejected (409); a NOT-covered (free) but tagged
client still succeeds. _ff_covered is never monkeypatched or referenced here
-- that would hide a real gating bug in either direction.
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
from dashboard import household as hh
from dashboard import orders as ord_mod

CAREGIVER = "spcaregiver2@example.com"
MEMBER = "spmember2@example.com"
TAGGED = "sptagged@example.com"
UNTAGGED = "spuntagged@example.com"

# Real FF-eligible catalog product (qty_pricing=True, not info_only), $69.97 base --
# same fixture product used by tests/test_ff_add_to_invoice.py.
FF_SLUG = "mucosa-syntropy-powder"
FF_CATALOG_CENTS = 6997

WET_AMD_ITEMS = [
    {"slug": FF_SLUG, "name": "Mucosa Syntropy Powder", "dose": "1 or more/day"},
    {"slug": "scar-solve", "name": "Scar Solve", "note": "Add for brunescent cataracts"},
    {"slug": "scar-silk", "name": "Scar Silk",
     "alts": [{"slug": "clear-the-way", "name": "Clear the Way"}]},
]


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
    monkeypatch.setenv("SUPPORT_PROGRAMS_ENABLED", "1")
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        hh.init_household_tables(cx)
        prog.init_table(cx)
        cc.init_table(cx)
        ord_mod.init_orders_table(cx)  # must run BEFORE the portal_published migration below
        cprices.init_table(cx)
    app._migrate_orders_portal_published()
    return app


def _seed_program(tmp_db, key="wet-amd", label="Wet AMD", items=None, consult_recommended=False,
                   modifiers=None):
    items = WET_AMD_ITEMS if items is None else items
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        prog.init_table(cx)
        prog.upsert(cx, key, label, consult_recommended, items, modifiers=modifiers)


def _seed_condition(tmp_db, email, key):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cc.init_table(cx)
        cc.set(cx, email, key, "test")


def _seed_portal(tmp_db, email, name="Client"):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        token, _pid = cp.upsert_portal(cx, email, name, {})
    return token


def _orders_rows(tmp_db):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        return [dict(r) for r in cx.execute("SELECT * FROM orders").fetchall()]


def _post(client, token, member=None):
    url = f"/api/portal/{token}/support-program/add-to-invoice"
    if member:
        url += f"?member={member}"
    return client.post(url)


# ---------------------------------------------------------------------------
# Flag off -> 404
# ---------------------------------------------------------------------------

def test_flag_off_404(app_mod, tmp_db, monkeypatch):
    monkeypatch.delenv("SUPPORT_PROGRAMS_ENABLED", raising=False)
    _seed_program(tmp_db)
    _seed_condition(tmp_db, TAGGED, "wet-amd")
    token = _seed_portal(tmp_db, TAGGED)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 404
    assert _orders_rows(tmp_db) == []


# ---------------------------------------------------------------------------
# No condition tagged -> 409, no order created (the gate)
# ---------------------------------------------------------------------------

def test_no_condition_409_no_order_created(app_mod, tmp_db):
    _seed_program(tmp_db)
    # UNTAGGED has no client_conditions override and no people row -> unresolved
    token = _seed_portal(tmp_db, UNTAGGED)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 409
    assert r.get_json()["error"] == "no support program"
    assert _orders_rows(tmp_db) == []


# ---------------------------------------------------------------------------
# Tagged client -> 200, exactly one order row, primary items only, real prices
# ---------------------------------------------------------------------------

def test_tagged_client_creates_one_order_with_primary_items(app_mod, tmp_db):
    _seed_program(tmp_db)
    _seed_condition(tmp_db, TAGGED, "wet-amd")
    token = _seed_portal(tmp_db, TAGGED)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["order_ref"] == f"SPINV-{TAGGED}-wet-amd"

    rows = _orders_rows(tmp_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "proposed"
    assert row["pay_status"] == "unpaid"
    assert row["portal_published"] == 0
    assert row["source"] == "in-house"
    assert row["channel"] == "support-program"

    items = json.loads(row["items_json"])
    # PRIMARY items only -- 3 items in WET_AMD_ITEMS, alts NOT expanded
    assert len(items) == 3
    slugs = [i["slug"] for i in items]
    assert slugs == [FF_SLUG, "scar-solve", "scar-silk"]
    assert "clear-the-way" not in slugs  # the alt must not be auto-added
    # non-zero real price for the catalog-known slug
    priced = next(i for i in items if i["slug"] == FF_SLUG)
    assert priced["unit_cents"] == FF_CATALOG_CENTS
    assert priced["line_cents"] == FF_CATALOG_CENTS
    assert row["total_cents"] == sum(i["line_cents"] for i in items)


# ---------------------------------------------------------------------------
# NOT gated on _ff_covered: a free (not covered) but tagged client still
# succeeds -- the key difference from the FF endpoint.
# ---------------------------------------------------------------------------

def test_free_but_tagged_client_still_succeeds(app_mod, tmp_db):
    _seed_program(tmp_db)
    _seed_condition(tmp_db, TAGGED, "wet-amd")
    token = _seed_portal(tmp_db, TAGGED)
    # No family_plan, no household coverage seeded anywhere -- TAGGED is
    # unambiguously NOT _ff_covered, yet the support-program gate doesn't care.
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    rows = _orders_rows(tmp_db)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Double POST -> still exactly one row (idempotent / insert-once)
# ---------------------------------------------------------------------------

def test_double_post_is_idempotent_single_row(app_mod, tmp_db):
    _seed_program(tmp_db)
    _seed_condition(tmp_db, TAGGED, "wet-amd")
    token = _seed_portal(tmp_db, TAGGED)
    client = app_mod.app.test_client()
    r1 = _post(client, token)
    r2 = _post(client, token)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.get_json()["order_ref"] == r2.get_json()["order_ref"]
    rows = _orders_rows(tmp_db)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Never-downgrade: a re-click after Rae has advanced/paid the order must NOT
# rewrite it.
# ---------------------------------------------------------------------------

def test_second_post_never_overwrites_an_advanced_order(app_mod, tmp_db):
    _seed_program(tmp_db)
    _seed_condition(tmp_db, TAGGED, "wet-amd")
    token = _seed_portal(tmp_db, TAGGED)
    client = app_mod.app.test_client()

    r1 = _post(client, token)
    assert r1.status_code == 200
    ext = r1.get_json()["order_ref"]
    assert ext == f"SPINV-{TAGGED}-wet-amd"

    # Simulate Rae advancing the order in the console + it being paid, with
    # manual adjustments/shipping applied and the items re-priced/finalized.
    with sqlite3.connect(tmp_db) as cx:
        cx.execute(
            "UPDATE orders SET status='invoiced', pay_status='paid', paid_cents=9999, "
            "adjustment_cents=500, shipping_cents=700, total_cents=12345, items_json='[]' "
            "WHERE source='in-house' AND external_ref=?", (ext,))
        cx.commit()

    r2 = _post(client, token)
    assert r2.status_code == 200
    body2 = r2.get_json()
    assert body2["ok"] is True
    assert body2["already_added"] is True
    assert body2["order_ref"] == ext

    rows = _orders_rows(tmp_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "invoiced"
    assert row["pay_status"] == "paid"
    assert row["paid_cents"] == 9999
    assert row["adjustment_cents"] == 500
    assert row["shipping_cents"] == 700
    assert row["total_cents"] == 12345
    assert row["items_json"] == "[]"


# ---------------------------------------------------------------------------
# Member-aware: ?member= uses the MEMBER's condition/program, not the
# caregiver's.
# ---------------------------------------------------------------------------

def test_member_aware_uses_members_own_program(app_mod, tmp_db, monkeypatch):
    monkeypatch.setenv("HOUSEHOLD_VIEW_ENABLED", "1")
    _seed_program(tmp_db, key="wet-amd", label="Wet AMD", items=WET_AMD_ITEMS)
    _seed_program(tmp_db, key="dry-eye", label="Dry Eye",
                  items=[{"slug": FF_SLUG, "name": "Mucosa Syntropy Powder"}],
                  consult_recommended=False)
    _seed_condition(tmp_db, CAREGIVER, "wet-amd")
    _seed_condition(tmp_db, MEMBER, "dry-eye")
    with sqlite3.connect(tmp_db) as cx:
        hh.add_member(cx, CAREGIVER, MEMBER, "Member", "dependent")
    token = _seed_portal(tmp_db, CAREGIVER)

    client = app_mod.app.test_client()
    r = _post(client, token, member=MEMBER)
    assert r.status_code == 200
    body = r.get_json()
    assert body["order_ref"] == f"SPINV-{MEMBER}-dry-eye"

    rows = _orders_rows(tmp_db)
    assert len(rows) == 1
    assert rows[0]["email"] == MEMBER
    items = json.loads(rows[0]["items_json"])
    assert len(items) == 1
    assert items[0]["slug"] == FF_SLUG


def test_consult_recommended_condition_rejects_add_to_invoice(app_mod, tmp_db, monkeypatch):
    """Wet AMD (consult_recommended) is NOT one-click orderable — the endpoint
    returns 409 and writes NO order; the client is directed to a consultation."""
    monkeypatch.setenv("SUPPORT_PROGRAMS_ENABLED", "1")
    email = "consultclient@example.com"
    _seed_program(tmp_db, key="wet-amd", label="Wet AMD", consult_recommended=True)
    _seed_condition(tmp_db, email, "wet-amd")
    token = _seed_portal(tmp_db, email)
    r = app_mod.app.test_client().post(f"/api/portal/{token}/support-program/add-to-invoice",
                                       json={})
    assert r.status_code == 409
    assert "consult" in (r.get_json() or {}).get("error", "").lower()
    import sqlite3
    with sqlite3.connect(tmp_db) as cx:
        rows = cx.execute("SELECT * FROM orders WHERE external_ref LIKE 'SPINV-%'").fetchall()
    assert rows == [], "no order may be created for a consult-recommended condition"


# ---------------------------------------------------------------------------
# Modifier-resolved item reaches the persisted order: a non-consult program
# with an ACTIVE diagnosis-implied "add" modifier must have the RESOLVED item
# set (base + modifier addition) priced and written to orders.items_json --
# not just the authored base items. Guards against regressing to pricing the
# raw, unresolved program.
# ---------------------------------------------------------------------------

def test_diagnosis_implied_modifier_addition_is_priced_and_persisted(app_mod, tmp_db):
    base_items = [{"slug": "wholomega", "name": "Wholomega"}]
    modifiers = [{
        "when": "dry-eye-severe",
        "action": "add",
        "items": [{"slug": "lipid-zyme", "name": "Lipid-Zyme"}],
        "source": "diagnosis-implied",
        "client_default": True,
    }]
    _seed_program(tmp_db, key="dry-eye-modtest", label="Dry Eye (mod test)",
                  items=base_items, modifiers=modifiers, consult_recommended=False)
    _seed_condition(tmp_db, TAGGED, "dry-eye-modtest")
    token = _seed_portal(tmp_db, TAGGED)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["order_ref"] == f"SPINV-{TAGGED}-dry-eye-modtest"

    rows = _orders_rows(tmp_db)
    assert len(rows) == 1
    row = rows[0]
    items = json.loads(row["items_json"])
    slugs = [i["slug"] for i in items]
    # base item + the diagnosis-implied modifier addition, both present
    assert slugs == ["wholomega", "lipid-zyme"]

    added = next(i for i in items if i["slug"] == "lipid-zyme")
    # real catalog price, not a zero/unresolved stub -- proves the modifier
    # addition was actually priced via _ff_line_cents, not dropped on the floor
    assert added["unit_cents"] == FF_CATALOG_CENTS
    assert added["line_cents"] == FF_CATALOG_CENTS
    assert row["total_cents"] == sum(i["line_cents"] for i in items)
