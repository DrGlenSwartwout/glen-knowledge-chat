import sqlite3
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx)
    return cx


def test_product_sources_counts_first_last_and_order():
    cx = _cx()
    # neuro-magnesium: self (first, 2026-06), then biofield twice (2026-07-01, 2026-07-08)
    re.record_event(cx, "a@b.com", "neuro-magnesium", "self", occurred_at="2026-06-01", origin_ref="p1")
    re.record_event(cx, "a@b.com", "neuro-magnesium", "biofield", occurred_at="2026-07-01", origin_ref="2026-07-01")
    re.record_event(cx, "a@b.com", "neuro-magnesium", "biofield", occurred_at="2026-07-08", origin_ref="2026-07-08")
    prods = re.product_sources(cx, "a@b.com")
    p = next(x for x in prods if x["product_key"] == "neuro-magnesium")
    assert p["hidden"] is False
    # icon order by first_touch: self (June) before biofield (July)
    assert [s["source"] for s in p["sources"]] == ["self", "biofield"]
    bf = next(s for s in p["sources"] if s["source"] == "biofield")
    assert bf["count"] == 2
    assert bf["first_touch"] == "2026-07-01" and bf["last_touch"] == "2026-07-08"


def test_hidden_flag():
    cx = _cx()
    re.record_event(cx, "a@b.com", "slugx", "purchased", occurred_at="d", origin_ref="1")
    re.set_hidden(cx, "a@b.com", "slugx", True)
    assert re.product_sources(cx, "a@b.com")[0]["hidden"] is True
    re.set_hidden(cx, "a@b.com", "slugx", False)
    assert re.product_sources(cx, "a@b.com")[0]["hidden"] is False


def test_product_sources_can_limit_scan_events_to_current_scan():
    cx = _cx()
    re.record_event(cx, "a@b.com", "june-product", "scan",
                    occurred_at="2026-06-20", origin_ref="scan:old:1")
    re.record_event(cx, "a@b.com", "august-product", "scan",
                    occurred_at="2026-08-02", origin_ref="scan:new:1")
    # Non-scan history remains even when scan events are scoped.
    re.record_event(cx, "a@b.com", "june-product", "purchased",
                    occurred_at="2026-07-01", origin_ref="order:1")

    prods = {p["product_key"]: p for p in
             re.product_sources(cx, "a@b.com", scan_origin_prefix="scan:new:")}
    assert {s["source"] for s in prods["august-product"]["sources"]} == {"scan"}
    assert {s["source"] for s in prods["june-product"]["sources"]} == {"purchased"}
