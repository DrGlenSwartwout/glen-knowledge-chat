import sqlite3
from dashboard import orders, recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    orders.init_orders_table(cx)
    re.init_recommendation_events(cx)
    return cx


def test_sourced_lines_emit_events_unsourced_do_not():
    cx = _cx()
    oid = orders.upsert_order(
        cx, source="in-house", external_ref="INH-1", email="A@B.com",
        items=[{"slug": "neuro-magnesium", "qty": 1, "source": "biofield"},
               {"slug": "immune-modulation", "qty": 1}],   # no source -> no event
        status="proposed")
    ev = re.list_events(cx, "a@b.com")
    assert len(ev) == 1
    assert ev[0]["source_key"] == "biofield" and ev[0]["product_key"] == "neuro-magnesium"
    assert ev[0]["origin_ref"] == str(oid)
    # idempotent: re-upsert (edit) emits nothing new
    orders.upsert_order(cx, source="in-house", external_ref="INH-1", email="a@b.com",
                        items=[{"slug": "neuro-magnesium", "qty": 1, "source": "biofield"}],
                        status="proposed")
    assert len(re.list_events(cx, "a@b.com")) == 1


def test_record_event_failure_never_breaks_order(monkeypatch):
    cx = _cx()
    def boom(*a, **k):
        raise RuntimeError("events down")
    monkeypatch.setattr(re, "record_event", boom)
    # order creation must still succeed
    oid = orders.upsert_order(cx, source="in-house", external_ref="INH-2", email="a@b.com",
                              items=[{"slug": "x", "qty": 1, "source": "self"}], status="proposed")
    assert oid is not None
    got = orders.get_order(cx, oid)
    assert got and got["email"] == "a@b.com"
