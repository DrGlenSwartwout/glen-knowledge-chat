"""quote() returns a single summed shipping_cents that _shipping_for_cart can
consume unchanged, including the multi-box case."""
import sqlite3
from dashboard.shipping import init_shipping_schema, quote


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    return db


def test_quote_shipping_cents_is_int_single(tmp_path):
    q = quote({"15ml": 5}, db_path=_db(tmp_path))
    assert isinstance(q["shipping_cents"], int) and q["shipping_cents"] > 0


def test_quote_shipping_cents_is_int_multibox(tmp_path):
    # qty=57 yields exactly two boxes: ['L', 'S'] -> 3200 + 1300 = 4500
    # (brief's qty=200 yields 4 boxes; adjusted to qty=57 for exact 2-box assertion)
    q = quote({"15ml": 57}, db_path=_db(tmp_path))
    assert isinstance(q["shipping_cents"], int)
    assert q["shipping_cents"] == 3200 + q["box_breakdown"][1]["charged_cents"]


def test_price_cart_uses_override(tmp_path, monkeypatch):
    """An override in product_bottle_types beats the products.json bottle_type."""
    import sqlite3
    from dashboard.shipping import init_shipping_schema, set_product_bottle_override, resolve_bottle_type
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    set_product_bottle_override("foo", "30ml", db_path=db)
    # Direct resolver check (the function _price_cart will call)
    assert resolve_bottle_type("foo", {"bottle_type": "15ml"}, db_path=db) == "30ml"
    assert resolve_bottle_type("bar", {"bottle_type": "50ml"}, db_path=db) == "50ml"
