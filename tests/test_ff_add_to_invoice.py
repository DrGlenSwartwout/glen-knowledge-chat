"""Slice 3c.2: POST /api/portal/<token>/ff-matches/add-to-invoice — the paid
add-to-invoice action. The MONEY-WRITE step: creates ONE unpaid, unpublished
invoice draft (`orders` row), idempotently, priced at the client's real FF
price. Gated on the real _ff_covered entitlement (never monkeypatched here —
that would hide a real gating bug); the draft must be PUBLISHED first.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_portal as cp
from dashboard import client_prices as cprices
from dashboard import family_plan as fp
from dashboard import ff_match_drafts as ffd
from dashboard import household as hh
from dashboard import orders as ord_mod

CAREGIVER = "caregiver@example.com"
MEMBER = "member@example.com"
FREE = "free@example.com"

# Real FF-eligible catalog product (qty_pricing=True, not info_only), $69.97 base.
FF_SLUG = "mucosa-syntropy-powder"
FF_CATALOG_CENTS = 6997


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
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    monkeypatch.setenv("FAMILY_PLAN_ENABLED", "1")
    app._init_membership_tables()  # memberships table lives in LOG_DB too
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        hh.init_household_tables(cx)
        fp.init_family_plan_table(cx)
        ffd.init_table(cx)
        ord_mod.init_orders_table(cx)  # must run BEFORE the portal_published migration below
        cprices.init_table(cx)
    # `portal_published` is normally added by a module-import-time migration against
    # the REAL LOG_DB (app.py:302); re-run it now that LOG_DB points at tmp_db AND the
    # orders table already exists here, so the ALTER TABLE actually lands (module-global
    # LOG_DB is read at call time, not import time — but the migration silently no-ops
    # if the table doesn't exist yet).
    app._migrate_orders_portal_published()
    return app


def _seed_covered_member(tmp_db, email=MEMBER):
    """Coverage via a caregiver's active Family Plan (real _ff_covered path —
    never monkeypatched)."""
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        hh.add_member(cx, CAREGIVER, email, relationship="")  # blank = legacy shared default
        fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")


def _seed_portal(tmp_db, email, name="Client"):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        token, _pid = cp.upsert_portal(cx, email, name, {})
    return token


def _seed_draft(tmp_db, email, scan_date="", items=None, published=True):
    """SCAN_RECOMMENDATIONS_ENABLED is left off in these tests, so
    _current_scan_date_for(email) resolves to "" (its documented no-scans
    fallback) — seed the draft at that same key so the route finds it."""
    if items is None:
        items = [{"name": "Mucosa Syntropy Powder", "slug": FF_SLUG,
                  "url": "/begin/product/mucosa-syntropy-powder",
                  "meaning": "gut lining support"}]
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        ffd.get_or_create(cx, email, scan_date, lambda: items)
        if published:
            ffd.publish(cx, email, scan_date)
    return scan_date, items


def _orders_rows(tmp_db):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        return [dict(r) for r in cx.execute("SELECT * FROM orders").fetchall()]


def _post(client, token, member=None):
    url = f"/api/portal/{token}/ff-matches/add-to-invoice"
    if member:
        url += f"?member={member}"
    return client.post(url)


# ---------------------------------------------------------------------------
# 1. Free member -> 403, no order row
# ---------------------------------------------------------------------------

def test_free_member_403_no_order_created(app_mod, tmp_db):
    token = _seed_portal(tmp_db, FREE)
    _seed_draft(tmp_db, FREE, published=True)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 403
    assert _orders_rows(tmp_db) == []


# ---------------------------------------------------------------------------
# 2. Covered member, draft NOT published -> 409
# ---------------------------------------------------------------------------

def test_covered_unpublished_draft_409(app_mod, tmp_db):
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    _seed_draft(tmp_db, MEMBER, published=False)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 409
    assert _orders_rows(tmp_db) == []


def test_covered_no_draft_at_all_409(app_mod, tmp_db):
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    # no draft seeded at all
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 409
    assert _orders_rows(tmp_db) == []


# ---------------------------------------------------------------------------
# 3. Covered member, published draft -> 200, exactly one order row
# ---------------------------------------------------------------------------

def test_covered_published_draft_creates_one_order(app_mod, tmp_db):
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    scan_date, items = _seed_draft(tmp_db, MEMBER, published=True)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["order_ref"] == f"FFINV-{MEMBER}-{scan_date}"

    rows = _orders_rows(tmp_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "proposed"
    assert row["pay_status"] == "unpaid"
    assert row["portal_published"] == 0
    assert row["source"] == "in-house"
    assert row["external_ref"] == body["order_ref"]
    assert row["channel"] == "ff-invoice"
    import json as _json
    stored_items = _json.loads(row["items_json"])
    assert len(stored_items) == len(items)
    assert stored_items[0]["slug"] == FF_SLUG
    assert stored_items[0]["name"] == "Mucosa Syntropy Powder"


# ---------------------------------------------------------------------------
# 4. Double POST -> still exactly one row (idempotent)
# ---------------------------------------------------------------------------

def test_double_post_is_idempotent_single_row(app_mod, tmp_db):
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    _seed_draft(tmp_db, MEMBER, published=True)
    client = app_mod.app.test_client()
    r1 = _post(client, token)
    r2 = _post(client, token)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.get_json()["order_ref"] == r2.get_json()["order_ref"]
    rows = _orders_rows(tmp_db)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# 4b. Never-downgrade: a re-click after Rae has advanced/paid the order must
#     NOT rewrite it (Critical finding — insert-once semantics).
# ---------------------------------------------------------------------------

def test_second_post_never_overwrites_an_advanced_order(app_mod, tmp_db):
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    scan_date, _items = _seed_draft(tmp_db, MEMBER, published=True)
    client = app_mod.app.test_client()

    r1 = _post(client, token)
    assert r1.status_code == 200
    ext = r1.get_json()["order_ref"]
    assert ext == f"FFINV-{MEMBER}-{scan_date}"

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
# 5. Covered via caregiver's family_plan.covers, not own payment -> allowed
# ---------------------------------------------------------------------------

def test_covered_via_caregiver_family_plan_allowed(app_mod, tmp_db):
    # MEMBER never paid anything themselves; coverage flows from CAREGIVER's plan.
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    _seed_draft(tmp_db, MEMBER, published=True)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    rows = _orders_rows(tmp_db)
    assert len(rows) == 1
    assert rows[0]["email"] == MEMBER


# ---------------------------------------------------------------------------
# 6. Pricing: client FF flat rate flows to each line; falls back to catalog
#    price when no client price is set.
# ---------------------------------------------------------------------------

def test_pricing_uses_client_ff_flat_when_set(app_mod, tmp_db):
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    _seed_draft(tmp_db, MEMBER, items=[
        {"name": "Mucosa Syntropy Powder", "slug": FF_SLUG, "url": "", "meaning": ""},
        {"name": "Mucosa Syntropy Powder 2", "slug": FF_SLUG, "url": "", "meaning": ""},
    ], published=True)
    FLAT = 5500  # deliberately different from the $69.97 catalog price, to prove
                 # the flat rate — not a silent catalog fallback — is what flowed through.
    with sqlite3.connect(tmp_db) as cx:
        cprices.set_ff_flat(cx, MEMBER, FLAT)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 200
    rows = _orders_rows(tmp_db)
    assert len(rows) == 1
    import json as _json
    items = _json.loads(rows[0]["items_json"])
    assert len(items) == 2
    for it in items:
        assert it["unit_cents"] == FLAT
        assert it["line_cents"] == FLAT
    assert rows[0]["total_cents"] == FLAT * 2


def test_pricing_falls_back_to_catalog_price_when_no_client_price(app_mod, tmp_db):
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    _seed_draft(tmp_db, MEMBER, items=[
        {"name": "Mucosa Syntropy Powder", "slug": FF_SLUG, "url": "", "meaning": ""},
    ], published=True)
    # no client_prices row of any kind for MEMBER
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 200
    rows = _orders_rows(tmp_db)
    import json as _json
    items = _json.loads(rows[0]["items_json"])
    assert items[0]["unit_cents"] == FF_CATALOG_CENTS
    assert items[0]["unit_cents"] != 0
    assert rows[0]["total_cents"] == FF_CATALOG_CENTS


def test_pricing_per_sku_special_wins_over_flat(app_mod, tmp_db):
    """Precedence: per-SKU client special beats the client's FF flat."""
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    _seed_draft(tmp_db, MEMBER, items=[
        {"name": "Mucosa Syntropy Powder", "slug": FF_SLUG, "url": "", "meaning": ""},
    ], published=True)
    with sqlite3.connect(tmp_db) as cx:
        cprices.set_ff_flat(cx, MEMBER, 5500)
        cprices.set_price(cx, MEMBER, FF_SLUG, 4200)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 200
    rows = _orders_rows(tmp_db)
    import json as _json
    items = _json.loads(rows[0]["items_json"])
    assert items[0]["unit_cents"] == 4200


# ---------------------------------------------------------------------------
# Flag off -> 404 (mirrors the ff-matches endpoint's own gating)
# ---------------------------------------------------------------------------

def test_flag_off_404(app_mod, tmp_db, monkeypatch):
    monkeypatch.delenv("FF_MATCHES_ENABLED", raising=False)
    _seed_covered_member(tmp_db, MEMBER)
    token = _seed_portal(tmp_db, MEMBER)
    _seed_draft(tmp_db, MEMBER, published=True)
    client = app_mod.app.test_client()
    r = _post(client, token)
    assert r.status_code == 404
    assert _orders_rows(tmp_db) == []
