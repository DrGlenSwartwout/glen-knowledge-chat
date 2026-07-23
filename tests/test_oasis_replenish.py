"""dashboard/oasis_replenish.py — owned-consumables projection for the "My
Healing Oasis" client-portal tile. Adapted from the task-3 brief's example to
the REAL dashboard/orders.py signatures (init_orders_table / upsert_order /
list_orders_by_email), since upsert_order always stamps created_at=now(), the
test seeds the order then back-dates created_at directly at the sqlite level
to exercise the running_low threshold."""
import sqlite3

from dashboard import oasis_replenish as rep
from dashboard import orders as _o

_CAT = {
    "neuro-mag": {"name": "Neuro-Mag", "bottle_type": "60 caps"},
    "water-ionizer": {"name": "Water Ionizer", "bottle_type": "own-box"},
    "biofield-analysis": {"name": "Biofield Analysis", "info_only": True, "service": True},
    "not-a-product": "just a string",  # malformed/mixed-type catalog entry
}


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _o.init_orders_table(cx)
    return cx


def _seed_order(cx, email, items, created_at):
    oid = _o.upsert_order(cx, source="test", external_ref="ord-1",
                          email=email, items=items, total_cents=1000)
    # upsert_order always stamps created_at=now(); back-date directly for the
    # running_low threshold test (real order creation has no created_at param).
    cx.execute("UPDATE orders SET created_at=? WHERE id=?", (created_at, oid))
    cx.commit()
    return oid


def test_only_consumables_with_running_low():
    cx = _cx()
    _seed_order(cx, "a@b.com",
               items=[{"slug": "neuro-mag", "qty": 1},
                      {"slug": "water-ionizer", "qty": 1}],
               created_at="2026-05-01")
    items = rep.replenish_items(cx, "a@b.com", catalog=_CAT, today="2026-07-01")
    slugs = {i["slug"] for i in items}
    assert slugs == {"neuro-mag"}          # device (own-box) excluded
    assert items[0]["running_low"] is True  # >30 days since last order
    assert items[0]["times_ordered"] == 1
    assert items[0]["last_ordered"] == "2026-05-01"
    assert items[0]["name"] == "Neuro-Mag"
    assert items[0]["url"] == "/begin/product/neuro-mag"


def test_recent_order_not_running_low():
    cx = _cx()
    _seed_order(cx, "a@b.com", items=[{"slug": "neuro-mag", "qty": 1}],
               created_at="2026-06-20")
    items = rep.replenish_items(cx, "a@b.com", catalog=_CAT, today="2026-07-01")
    assert len(items) == 1
    assert items[0]["running_low"] is False


def test_service_and_info_only_excluded():
    cx = _cx()
    _seed_order(cx, "a@b.com", items=[{"slug": "biofield-analysis", "qty": 1}],
               created_at="2026-06-20")
    items = rep.replenish_items(cx, "a@b.com", catalog=_CAT, today="2026-07-01")
    assert items == []


def test_malformed_catalog_entry_never_crashes():
    cx = _cx()
    _seed_order(cx, "a@b.com", items=[{"slug": "not-a-product", "qty": 1}],
               created_at="2026-06-20")
    items = rep.replenish_items(cx, "a@b.com", catalog=_CAT, today="2026-07-01")
    assert items == []  # non-dict catalog entry excluded, no crash


def test_unknown_slug_not_in_catalog_excluded():
    cx = _cx()
    _seed_order(cx, "a@b.com", items=[{"slug": "ghost-slug", "qty": 1}],
               created_at="2026-06-20")
    items = rep.replenish_items(cx, "a@b.com", catalog=_CAT, today="2026-07-01")
    assert items == []


def test_running_low_sorts_before_recent():
    cx = _cx()
    _seed_order(cx, "a@b.com", items=[{"slug": "neuro-mag", "qty": 1}],
               created_at="2026-05-01")  # running low vs 2026-07-01
    # A second consumable slug ordered recently, seeded via a second order (a
    # second external_ref -- upsert_order is idempotent on (source, external_ref)).
    oid2 = _o.upsert_order(cx, source="test", external_ref="ord-2",
                           email="a@b.com", items=[{"slug": "other-consumable", "qty": 1}],
                           total_cents=500)
    cx.execute("UPDATE orders SET created_at=? WHERE id=?", ("2026-06-28", oid2))
    cx.commit()
    cat = dict(_CAT)
    cat["other-consumable"] = {"name": "Other Consumable", "bottle_type": "30ml"}
    items = rep.replenish_items(cx, "a@b.com", catalog=cat, today="2026-07-01")
    assert [i["slug"] for i in items] == ["neuro-mag", "other-consumable"]
    assert items[0]["running_low"] is True
    assert items[1]["running_low"] is False
