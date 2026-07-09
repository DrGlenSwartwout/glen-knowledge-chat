"""The client's saved pickup default must actually reach the order.

PR #738 added `client_prefs.get_pickup_default` plus a console panel to set it, but
nothing ever read it: `/api/orders/manual` still did `pickup = bool(body.get("pickup"))`
and `biofield_invoice.default_create_order` posts NO `pickup` key. So a client marked
"pickup by default" was still charged shipping on their Biofield hand-off — the
preference was stored, displayed, and applied to nothing.

Rule: an explicit `pickup` key in the body ALWAYS wins (order entry always posts the
checkbox). Only an ABSENT key falls back to the client's saved preference. Unknown
client, blank email, or a `client_prefs` table that was never created -> False, i.e.
shipping is charged. Guessing True ships physical goods for free.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

BOTTLE = {"slug": "mix", "price_cents": 7000, "name": "Drink Mix"}
_CAT = {"mix": BOTTLE}


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
    appmod._init_people_table()
    from dashboard import orders as O
    cx = sqlite3.connect(db)
    O.init_orders_table(cx)
    cx.close()
    # Auth via the actor, NOT X-Console-Key: the console-secret globals are mutated by
    # neighbouring test files, which makes header-auth tests pass alone and 401 in a
    # full-suite run.
    monkeypatch.setattr(appmod, "_bos_actor",
                        lambda: type("A", (), {"role": appmod._bos_rbac.OWNER})())
    # dashboard.shipping resolves its db path off os.environ["DATA_DIR"], independent of
    # appmod.LOG_DB. Pin it to our db and seed the schema so shipping_cents reflects a
    # real box-fit quote instead of another test's ambient state.
    monkeypatch.setattr(appmod._shipping, "_default_db_path", lambda: db)
    with sqlite3.connect(db) as scx:
        appmod._shipping.init_shipping_schema(scx)
    return appmod, db


def _pref(db, email, on):
    from dashboard import client_prefs as CP
    with sqlite3.connect(db) as cx:
        CP.init_table(cx)
        CP.set_pickup_default(cx, email, on)


def _post(appmod, body):
    base = {"customer": {"name": "T", "email": body.pop("email", ""),
                         "address": {"address1": "1", "city": "Hilo", "state": "HI",
                                     "zip": "96720", "country": "US"}},
            "lines": [{"slug": "mix", "qty": 2}], "method": "Zelle"}
    base.update(body)
    r = appmod.app.test_client().post("/api/orders/manual", json=base)
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


def _stored(db, order_id):
    from dashboard import orders as O
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    try:
        return O.get_order(cx, order_id)
    finally:
        cx.close()


def test_absent_pickup_key_uses_saved_pref(env):
    """The Biofield hand-off path: no `pickup` key, flagged client -> pickup, $0 ship."""
    appmod, db = env
    _pref(db, "pick@x.com", True)
    o = _stored(db, _post(appmod, {"email": "pick@x.com"})["order_id"])
    assert o["channel"] == "pickup"
    assert o["shipping_cents"] == 0


def test_absent_pickup_key_unflagged_client_charges_shipping(env):
    appmod, db = env
    _pref(db, "ship@x.com", False)
    o = _stored(db, _post(appmod, {"email": "ship@x.com"})["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_absent_pickup_key_unknown_client_charges_shipping(env):
    appmod, db = env
    _pref(db, "someone-else@x.com", True)      # table exists, this client is not in it
    o = _stored(db, _post(appmod, {"email": "stranger@x.com"})["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_absent_pickup_key_with_no_client_prefs_table_charges_shipping(env):
    """`client_prefs` is created lazily by the console panel. An operator who never
    opened it must still be able to take an order — fail safe, toward charging."""
    appmod, db = env
    with sqlite3.connect(db) as cx:
        assert cx.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='client_prefs'"
        ).fetchone()[0] == 0, "fixture must not pre-create client_prefs"
    o = _stored(db, _post(appmod, {"email": "nobody@x.com"})["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_absent_pickup_key_blank_email_charges_shipping(env):
    appmod, db = env
    o = _stored(db, _post(appmod, {"email": ""})["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_explicit_false_beats_saved_pref(env):
    """Order entry always posts the checkbox; unticking it wins over the saved pref."""
    appmod, db = env
    _pref(db, "pick@x.com", True)
    o = _stored(db, _post(appmod, {"email": "pick@x.com", "pickup": False})["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_explicit_true_beats_unflagged_client(env):
    appmod, db = env
    _pref(db, "ship@x.com", False)
    o = _stored(db, _post(appmod, {"email": "ship@x.com", "pickup": True})["order_id"])
    assert o["channel"] == "pickup"
    assert o["shipping_cents"] == 0


def test_pref_is_case_and_whitespace_insensitive(env):
    """client_prefs normalizes on write; the order path must match on read."""
    appmod, db = env
    _pref(db, "  PICK@X.com ", True)
    o = _stored(db, _post(appmod, {"email": "pick@x.com"})["order_id"])
    assert o["channel"] == "pickup"


def test_edit_never_resurrects_the_pref(env):
    """PR #734 latch guard. Unticking pickup on a flagged client's order must STICK.
    If api_orders_edit consulted the saved pref, this would snap back to 'pickup'."""
    appmod, db = env
    _pref(db, "pick@x.com", True)
    oid = _post(appmod, {"email": "pick@x.com"})["order_id"]
    assert _stored(db, oid)["channel"] == "pickup"
    r = appmod.app.test_client().post(
        f"/api/orders/{oid}/edit",
        json={"lines": [{"slug": "mix", "qty": 2}], "pickup": False})
    assert r.status_code == 200, r.get_data(as_text=True)
    o = _stored(db, oid)
    assert o["channel"] == "retail", "edit route re-resolved the saved pref — latch rebuilt"
    assert o["shipping_cents"] > 0
