"""A Biofield hand-off posts NO `pickup` key -> the client's saved flag decides.
Order entry always posts the checkbox -> the checkbox decides, always."""
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
    return appmod, db


def _flag(db, email, on):
    from dashboard import customers as C
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT OR IGNORE INTO people (email) VALUES (?)", (email,))
        cx.commit()
        pid = cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]
        C.set_pickup_default(cx, pid, on)


def _post(appmod, body):
    key = appmod.dashboard.CONSOLE_SECRET or ""
    base = {"customer": {"name": "T", "email": body.pop("email", ""),
                         "address": {"address1": "1", "city": "Hilo", "state": "HI",
                                     "zip": "96720", "country": "US"}},
            "lines": [{"slug": "mix", "qty": 2}], "method": "Zelle"}
    base.update(body)
    r = appmod.app.test_client().post("/api/orders/manual", json=base,
                                      headers={"X-Console-Key": key})
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


def test_absent_pickup_key_uses_client_flag(env):
    """The Biofield hand-off path: no `pickup` key, flagged client -> pickup."""
    appmod, db = env
    _flag(db, "pick@x.com", True)
    j = _post(appmod, {"email": "pick@x.com"})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "pickup"
    assert o["shipping_cents"] == 0


def test_absent_pickup_key_unflagged_client_charges_shipping(env):
    appmod, db = env
    _flag(db, "ship@x.com", False)
    j = _post(appmod, {"email": "ship@x.com"})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_absent_pickup_key_unknown_client_charges_shipping(env):
    """Fail toward charging: never free-ship a client we've never seen."""
    appmod, db = env
    j = _post(appmod, {"email": "stranger@x.com"})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_explicit_false_beats_flagged_client(env):
    """Order entry always posts the checkbox; unticking it wins over the flag."""
    appmod, db = env
    _flag(db, "pick@x.com", True)
    j = _post(appmod, {"email": "pick@x.com", "pickup": False})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_explicit_true_beats_unflagged_client(env):
    appmod, db = env
    _flag(db, "ship@x.com", False)
    j = _post(appmod, {"email": "ship@x.com", "pickup": True})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "pickup"
    assert o["shipping_cents"] == 0


def test_edit_never_resurrects_the_flag(env):
    """PR #734 latch guard: unticking pickup on a FLAGGED client's order must stick.
    If the edit route consulted pickup_default, this would snap back to 'pickup'."""
    appmod, db = env
    _flag(db, "pick@x.com", True)
    j = _post(appmod, {"email": "pick@x.com"})          # created as pickup via flag
    assert _stored(db, j["order_id"])["channel"] == "pickup"
    key = appmod.dashboard.CONSOLE_SECRET or ""
    r = appmod.app.test_client().post(
        f"/api/orders/{j['order_id']}/edit",
        json={"lines": [{"slug": "mix", "qty": 2}], "pickup": False},
        headers={"X-Console-Key": key})
    assert r.status_code == 200, r.get_data(as_text=True)
    o = _stored(db, j["order_id"])
    assert o["channel"] == "retail", "edit route re-resolved the client flag — latch rebuilt"
    assert o["shipping_cents"] > 0
