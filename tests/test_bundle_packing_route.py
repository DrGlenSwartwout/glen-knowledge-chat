"""A bundle ordered through the real route must ship as its CONTENTS.

Before: box_counts={'default': 1} -> quote() errors -> coarse qty rule -> 1 bottle ->
Small box. After: the bundle expands to its component bottles and gets a real box-fit
quote for all of them.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

# Two component bottles + the bundle that holds them. 30ml is a seeded standard type.
COMP_A = {"slug": "comp-a", "name": "Comp A", "bottle_type": "30ml", "price_cents": 1000}
COMP_B = {"slug": "comp-b", "name": "Comp B", "bottle_type": "30ml", "price_cents": 1000}
BUNDLE = {"slug": "bund", "name": "Bundle", "price_cents": 10000, "bundle": True,
          "bundle_components": ["Comp A", "Comp B"]}
GHOST = {"slug": "ghost-bund", "name": "Ghost Bundle", "price_cents": 5000,
         "bundle": True, "bundle_components": ["Comp A", "No Such Product"]}
_CAT = {p["slug"]: p for p in (COMP_A, COMP_B, BUNDLE, GHOST)}


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture
def env(monkeypatch, tmp_path):
    appmod = _app()
    db = str(tmp_path / "m.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "_get_product", _CAT.get)
    monkeypatch.setattr(appmod, "_catalog_products", lambda: list(_CAT.values()))
    appmod._init_people_table()
    from dashboard import orders as O
    cx = sqlite3.connect(db)
    O.init_orders_table(cx)
    cx.close()
    monkeypatch.setattr(appmod, "_bos_actor",
                        lambda: type("A", (), {"role": appmod._bos_rbac.OWNER})())
    monkeypatch.setattr(appmod._shipping, "_default_db_path", lambda: db)
    with sqlite3.connect(db) as scx:
        appmod._shipping.init_shipping_schema(scx)
    return appmod, db


def _post(appmod, lines):
    base = {"customer": {"name": "T", "email": "b@x.com",
                         "address": {"address1": "1", "city": "Hilo", "state": "HI",
                                     "zip": "96720", "country": "US"}},
            "lines": lines, "method": "Zelle", "pickup": False}
    return appmod.app.test_client().post("/api/orders/manual", json=base)


def _stored(db, order_id):
    from dashboard import orders as O
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    try:
        return O.get_order(cx, order_id)
    finally:
        cx.close()


def _quote_for(appmod, db, counts):
    return int(appmod._shipping.quote(counts, db_path=db).get("shipping_cents") or 0)


def test_bundle_ships_as_its_components(env):
    appmod, db = env
    r = _post(appmod, [{"slug": "bund", "qty": 1}])
    assert r.status_code == 200, r.get_data(as_text=True)
    o = _stored(db, r.get_json()["order_id"])
    expected = _quote_for(appmod, db, {"30ml": 2})      # the two component bottles
    assert expected > 0
    assert o["shipping_cents"] == expected


def test_bundle_matches_buying_the_components_separately(env):
    """The whole point: one bundle must cost the same to ship as its contents."""
    appmod, db = env
    a = _stored(db, _post(appmod, [{"slug": "bund", "qty": 1}]).get_json()["order_id"])
    b = _stored(db, _post(appmod, [{"slug": "comp-a", "qty": 1},
                                   {"slug": "comp-b", "qty": 1}]).get_json()["order_id"])
    assert a["shipping_cents"] == b["shipping_cents"]


def test_bundle_qty_multiplies_components(env):
    appmod, db = env
    o = _stored(db, _post(appmod, [{"slug": "bund", "qty": 2}]).get_json()["order_id"])
    assert o["shipping_cents"] == _quote_for(appmod, db, {"30ml": 4})


def test_bundle_mixed_with_a_loose_bottle(env):
    appmod, db = env
    o = _stored(db, _post(appmod, [{"slug": "bund", "qty": 1},
                                   {"slug": "comp-a", "qty": 1}]).get_json()["order_id"])
    assert o["shipping_cents"] == _quote_for(appmod, db, {"30ml": 3})


def test_unresolvable_component_is_a_loud_400_not_a_silent_undercharge(env):
    appmod, db = env
    r = _post(appmod, [{"slug": "ghost-bund", "qty": 1}])
    assert r.status_code == 400, r.get_data(as_text=True)
    err = r.get_json()["error"]
    assert "No Such Product" in err, err


def test_unresolvable_component_creates_no_order(env):
    """A loud failure must not leave a half-priced order behind."""
    appmod, db = env
    _post(appmod, [{"slug": "ghost-bund", "qty": 1}])
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT count(*) FROM orders").fetchone()[0] == 0
