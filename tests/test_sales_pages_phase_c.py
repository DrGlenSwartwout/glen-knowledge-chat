import sqlite3
from dashboard import sales_image_exposures as ex

def _cx(): return sqlite3.connect(":memory:")

def test_record_dedups_per_session():
    cx = _cx()
    ex.record(cx, "a", "s1")
    ex.record(cx, "a", "s1")     # same session -> no new row
    ex.record(cx, "a", "s2")     # different session
    ex.record(cx, "b", "s1")     # different product
    assert ex.per_product_counts(cx) == {"a": 2, "b": 1}

def test_record_ignores_empty_session():
    cx = _cx()
    ex.record(cx, "a", "")
    ex.record(cx, "a", None)
    assert ex.per_product_counts(cx) == {}
