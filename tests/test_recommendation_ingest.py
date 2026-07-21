import json, sqlite3
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    re.init_recommendation_events(cx)
    # minimal biofield_reveals + orders tables the readers use
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, "
               "scan_date TEXT, interpretation_json TEXT DEFAULT '{}', remedies_json TEXT DEFAULT '[]', "
               "first_approved INTEGER DEFAULT 0, token_hash TEXT, approved_at TEXT, approved_by TEXT, "
               "created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '', dropped TEXT DEFAULT '[]', "
               "layers_json TEXT DEFAULT '[]', notified_at TEXT, requested_at TEXT)")
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT, "
               "external_ref TEXT, email TEXT, name TEXT, items_json TEXT, address_json TEXT, total_cents INTEGER, "
               "status TEXT, pay_status TEXT, paid_at TEXT)")
    return cx


def test_ingest_biofield_one_event_per_remedy_per_reveal(monkeypatch):
    cx = _cx()
    cur1 = cx.execute("INSERT INTO biofield_reveals (email, scan_date, remedies_json) VALUES (?,?,?)",
               ("a@b.com", "2026-07-01",
                json.dumps([{"name": "Neuro Magnesium", "slug": "neuro-magnesium"},
                            {"name": "No Slug", "slug": ""}])))
    cur2 = cx.execute("INSERT INTO biofield_reveals (email, scan_date, remedies_json) VALUES (?,?,?)",
               ("a@b.com", "2026-07-08",
                json.dumps([{"name": "Neuro Magnesium", "slug": "neuro-magnesium"}])))
    n = re.ingest_biofield(cx, "a@b.com")
    assert n == 2                       # slug="" skipped; two distinct reveals for neuro-magnesium
    ev = [e for e in re.list_events(cx, "a@b.com")]
    assert all(e["source_key"] == "biofield" for e in ev)
    # origin_ref is the reveal id (dedup grain), NOT the scan_date
    assert {e["origin_ref"] for e in ev} == {str(cur1.lastrowid), str(cur2.lastrowid)}
    assert re.ingest_biofield(cx, "a@b.com") == 0     # idempotent


def test_ingest_biofield_distinct_reveals_same_scan_date_both_counted(monkeypatch):
    """Regression: two DISTINCT reveals sharing a scan_date must not collapse into one event.
    Dedup key is (client_email, product_key, source_key, origin_ref) and origin_ref must be the
    reveal id, not scan_date -- otherwise same-day reveals with the same remedy slug collapse."""
    cx = _cx()
    same_date = "2026-07-01"
    cx.execute("INSERT INTO biofield_reveals (email, scan_date, remedies_json) VALUES (?,?,?)",
               ("a@b.com", same_date,
                json.dumps([{"name": "Neuro Magnesium", "slug": "neuro-magnesium"}])))
    cx.execute("INSERT INTO biofield_reveals (email, scan_date, remedies_json) VALUES (?,?,?)",
               ("a@b.com", same_date,
                json.dumps([{"name": "Neuro Magnesium", "slug": "neuro-magnesium"}])))
    n = re.ingest_biofield(cx, "a@b.com")
    assert n == 2                       # two distinct reveals -> two events, despite same scan_date
    ev = [e for e in re.list_events(cx, "a@b.com") if e["product_key"] == "neuro-magnesium"]
    assert len(ev) == 2
    prods = {p["product_key"]: p for p in re.product_sources(cx, "a@b.com")}
    biofield_count = sum(s["count"] for s in prods["neuro-magnesium"]["sources"]
                          if s["source"] == "biofield")
    assert biofield_count == 2
    assert re.ingest_biofield(cx, "a@b.com") == 0     # idempotent on re-ingest


def test_ingest_purchased_paid_only_by_line(monkeypatch):
    cx = _cx()
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_at, status) VALUES (?,?,?,?,?)",
               ("a@b.com", json.dumps([{"slug": "neuro-magnesium", "qty": 2},
                                       {"slug": "", "qty": 1}]), "paid", "2026-07-10", "done"))
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_at, status) VALUES (?,?,?,?,?)",
               ("a@b.com", json.dumps([{"slug": "immune-modulation"}]), "unpaid", None, "new"))
    n = re.ingest_purchased(cx, "a@b.com")
    assert n == 1                       # only the paid order's slugged line
    ev = re.list_events(cx, "a@b.com")
    assert ev[0]["source_key"] == "purchased" and ev[0]["product_key"] == "neuro-magnesium"
    assert re.ingest_purchased(cx, "a@b.com") == 0     # idempotent
