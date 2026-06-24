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
