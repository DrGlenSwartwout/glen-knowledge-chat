"""Address-on-edit + shipping-preview endpoint.

- /api/orders/<id>/edit now persists the ship-to from the editor form (address_override)
  and, absent it, leaves the stored address untouched.
- /api/orders/shipping-preview quotes shipping for the current cart + typed ship-to.

The heavy pricer (_price_inhouse_invoice / _price_cart) is monkeypatched so these
tests stay deterministic and secretless; the real rate path is verified live.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import rbac as _rbac


def _app(tmp_path, monkeypatch, *, seed_address):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard import orders as O
    with sqlite3.connect(db) as cx:
        O.init_orders_table(cx)
        O.upsert_order(cx, source="in-house", external_ref="INH-T1", status="confirmed",
                       email="b@x.com", name="Bobbi", channel="pickup",
                       items=[{"slug": "mag", "qty": 2, "name": "Mag"}],
                       address=seed_address, total_cents=1000)
        cx.commit()
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    appmod.app.config["TESTING"] = True
    monkeypatch.setattr(appmod, "_bos_actor",
                        lambda: _rbac.Actor(role="owner", name="glen"))
    # Canned priced result so the edit path doesn't need the real catalog/pricer.
    monkeypatch.setattr(appmod, "_price_inhouse_invoice", lambda *a, **k: {
        "items_rec": [{"slug": "mag", "name": "Mag", "qty": 2, "unit_cents": 500, "line_cents": 1000}],
        "cart": [{"slug": "mag", "qty": 2}], "subtotal_cents": 1000, "shipping_cents": 550,
        "get_cents": 0, "discount_cents": 0, "adjustment_cents": 0,
        "points_redeemed_cents": 0, "total_cents": 1550})
    monkeypatch.setattr(appmod, "_push_invoice_edit_to_qbo", lambda *a, **k: {"pushed": False})
    return appmod, appmod.app.test_client(), db


def _order_addr(db):
    from dashboard import orders as O
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        return O.get_order(cx, 1)


def test_edit_persists_address_and_flips_channel(tmp_path, monkeypatch):
    appmod, client, db = _app(tmp_path, monkeypatch, seed_address={})
    r = client.post("/api/orders/1/edit", json={
        "lines": [{"slug": "mag", "qty": 2}], "pickup": False,
        "address": {"address1": "12 Elm St", "city": "Newburyport", "state": "MA", "zip": "01950"}})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    o = _order_addr(db)
    a = o.get("address") or {}
    assert (a.get("street") or a.get("address1")) == "12 Elm St"
    assert (a.get("city"), a.get("state"), a.get("zip")) == ("Newburyport", "MA", "01950")
    assert o.get("channel") == "retail"          # pickup:false -> retail
    assert int(o.get("shipping_cents") or 0) == 550


def test_edit_without_address_preserves_existing(tmp_path, monkeypatch):
    seed = {"street": "9 Old Rd", "city": "Hilo", "state": "HI", "zip": "96720"}
    appmod, client, db = _app(tmp_path, monkeypatch, seed_address=seed)
    r = client.post("/api/orders/1/edit", json={"lines": [{"slug": "mag", "qty": 2}], "pickup": True})
    assert r.status_code == 200
    a = _order_addr(db).get("address") or {}
    assert (a.get("street"), a.get("city"), a.get("zip")) == ("9 Old Rd", "Hilo", "96720")


def test_shipping_preview_pickup_zero_and_gate(tmp_path, monkeypatch):
    appmod, client, _ = _app(tmp_path, monkeypatch, seed_address={})
    # Pickup short-circuits to 0 without touching the rate engine.
    r = client.post("/api/orders/shipping-preview", json={
        "lines": [{"slug": "mag", "qty": 2}], "pickup": True, "address": {}})
    assert r.status_code == 200 and r.get_json() == {"ok": True, "shipping_cents": 0, "get_cents": 0, "pickup": True}
    # Empty cart is a 400, not a bogus quote.
    r2 = client.post("/api/orders/shipping-preview", json={"lines": [], "pickup": False})
    assert r2.status_code == 400
    # Owner-gated.
    monkeypatch.setattr(appmod, "_bos_actor", lambda: None)
    assert client.post("/api/orders/shipping-preview", json={"lines": [], "pickup": True}).status_code == 401
