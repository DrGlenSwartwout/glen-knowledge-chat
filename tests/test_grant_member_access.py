"""One-click "grant 30-day member access + reprice" on the invoice editor.

A paid $300 Causal Biofield Analysis auto-grants a 30-day care_taster membership
(_fulfill_biofield_program) that flips on member pricing. Clients invoiced by hand
never hit that path, so their FF capsules stay at full price. POST
/api/orders/<oid>/grant-member-access closes the gap in one click: grant the same
care_taster window, then reprice the order's own lines at the now-active member rate.

Rules under test:
  * The grant is a care_taster membership for the order's email.
  * Member pricing is active BY the time the reprice runs (grant committed first).
  * Override lines are re-sent WITH unit_cents (preserved); auto-priced lines are
    re-sent WITHOUT (they drop to the member rate) — same contract as the editor.
  * Idempotent: an already-active member gets no second grant, but is still repriced.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


EMAIL = "hand@invoiced.com"
# A stored invoice: a courtesy Biofield line (owner override) + an auto-priced FF line.
ITEMS = [
    {"slug": "biofield-analysis", "name": "Biofield Analysis", "qty": 1,
     "unit_cents": 20000, "override": True},
    {"slug": "fiber-cleanse", "name": "Fiber Cleanse", "qty": 1,
     "unit_cents": 6997, "override": None},
]


@pytest.fixture
def env(monkeypatch, tmp_path):
    appmod = _app()
    db = str(tmp_path / "g.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "_bos_actor",
                        lambda: type("A", (), {"role": appmod._bos_rbac.OWNER})())
    from dashboard import orders as O
    with sqlite3.connect(db) as cx:
        O.init_orders_table(cx)
        appmod.init_membership_tables(cx)
        cx.commit()
    return appmod, db, O


def _seed_order(appmod, db, O, *, email=EMAIL, items=None):
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        oid = O.upsert_order(cx, source="in-house", external_ref="INH-TEST01",
                             email=email, name="Hand Invoiced",
                             items=(items if items is not None else ITEMS),
                             total_cents=26997, channel="retail", status="proposed")
        cx.commit()
    return oid


def _spy_pricer(appmod, monkeypatch, captured):
    """Replace the pricer with a spy that records the lines it was handed and whether the
    caller is a paid member at that moment, then returns a valid priced dict so persistence
    still runs."""
    def spy(lines_in, *, email, pickup, ship, discount_cents_in=None,
            adjustment_cents_in=None, points_redeem_cents_in=None):
        captured["lines"] = lines_in
        captured["email"] = email
        captured["is_paid_member"] = appmod._is_paid_member(email)
        items_rec = [dict(l, name=l["slug"], unit_cents=l.get("unit_cents", 5000))
                     for l in lines_in]
        return {"items_rec": items_rec, "total_cents": 22000, "subtotal_cents": 22000,
                "discount_cents": 0, "adjustment_cents": 0, "shipping_cents": 0,
                "get_cents": 0}
    monkeypatch.setattr(appmod, "_price_inhouse_invoice", spy)


def _member_rows(db, email):
    with sqlite3.connect(db) as cx:
        return cx.execute(
            "SELECT source FROM memberships WHERE lower(email)=lower(?)", (email,)
        ).fetchall()


def test_grants_care_taster_then_reprices_as_member(env, monkeypatch):
    appmod, db, O = env
    oid = _seed_order(appmod, db, O)
    captured = {}
    _spy_pricer(appmod, monkeypatch, captured)

    r = appmod.app.test_client().post(f"/api/orders/{oid}/grant-member-access", json={})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"] and j["granted"] is True and j["already_member"] is False
    assert j["membership_expires"]

    # The grant is a care_taster membership for this order's email.
    rows = _member_rows(db, EMAIL)
    assert len(rows) == 1 and rows[0][0] == appmod.CARE_TASTER_SOURCE

    # Member pricing was active by the time the reprice ran.
    assert captured["is_paid_member"] is True

    # Override line preserved (sent WITH unit_cents); auto line dropped (no unit_cents).
    by_slug = {l["slug"]: l for l in captured["lines"]}
    assert by_slug["biofield-analysis"].get("unit_cents") == 20000
    assert "unit_cents" not in by_slug["fiber-cleanse"]

    # Total reflects the repriced invoice.
    assert j["new_total_cents"] == 22000 and j["old_total_cents"] == 26997


def test_idempotent_for_an_existing_member(env, monkeypatch):
    appmod, db, O = env
    oid = _seed_order(appmod, db, O)
    # Pre-existing active membership.
    with sqlite3.connect(db) as cx:
        appmod._grant_membership(cx, EMAIL, 30, "membership_month")
        cx.commit()
    captured = {}
    _spy_pricer(appmod, monkeypatch, captured)

    r = appmod.app.test_client().post(f"/api/orders/{oid}/grant-member-access", json={})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"] and j["granted"] is False and j["already_member"] is True

    # No second grant was written.
    assert len(_member_rows(db, EMAIL)) == 1
    # But the order was still repriced at the member rate.
    assert captured["is_paid_member"] is True


def test_rejects_order_with_no_email(env, monkeypatch):
    appmod, db, O = env
    oid = _seed_order(appmod, db, O, email="")
    _spy_pricer(appmod, monkeypatch, {})
    r = appmod.app.test_client().post(f"/api/orders/{oid}/grant-member-access", json={})
    assert r.status_code == 400
    assert "email" in (r.get_json().get("error") or "").lower()
    # No membership was granted to a blank email.
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM memberships").fetchone()[0] == 0
