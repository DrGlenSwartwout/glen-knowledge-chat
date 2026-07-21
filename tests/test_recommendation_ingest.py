import json, sqlite3
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    re.init_recommendation_events(cx)
    # orders is the only source ingested in Phase 1 (purchased). _row_to_dict pops address_json.
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT, "
               "external_ref TEXT, email TEXT, name TEXT, items_json TEXT, address_json TEXT, total_cents INTEGER, "
               "status TEXT, pay_status TEXT, paid_at TEXT)")
    return cx


def test_ingest_purchased_paid_only_by_line(monkeypatch):
    cx = _cx()
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_at, status) VALUES (?,?,?,?,?)",
               ("a@b.com", json.dumps([{"slug": "neuro-magnesium", "qty": 2},
                                       {"slug": "", "qty": 1}]), "paid", "2026-07-10", "done"))
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_at, status) VALUES (?,?,?,?,?)",
               ("a@b.com", json.dumps([{"slug": "immune-modulation"}]), "unpaid", None, "new"))
    n = re.ingest_purchased(cx, "a@b.com")
    assert n == 1                       # only the paid order's slugged line (blank slug skipped)
    ev = re.list_events(cx, "a@b.com")
    assert ev[0]["source_key"] == "purchased" and ev[0]["product_key"] == "neuro-magnesium"
    assert re.ingest_purchased(cx, "a@b.com") == 0     # idempotent


def test_ingest_purchased_reorders_increment_across_orders():
    cx = _cx()
    for oid_paid_at in [("2026-07-01",), ("2026-08-01",)]:
        cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_at, status) VALUES (?,?,?,?,?)",
                   ("a@b.com", json.dumps([{"slug": "neuro-magnesium"}]), "paid", oid_paid_at[0], "done"))
    n = re.ingest_purchased(cx, "a@b.com")
    assert n == 2                       # two distinct paid orders -> two purchased events
    prods = {p["product_key"]: p for p in re.product_sources(cx, "a@b.com")}
    purchased = next(s for s in prods["neuro-magnesium"]["sources"] if s["source"] == "purchased")
    assert purchased["count"] == 2
