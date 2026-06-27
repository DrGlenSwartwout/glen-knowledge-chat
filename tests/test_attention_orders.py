import sqlite3
from dashboard import orders as o


def _seed(db_path):
    cx = sqlite3.connect(str(db_path))
    o.init_orders_table(cx)
    o.init_fulfillments_table(cx)
    rows = [  # (name, email, status, pay_status, total_cents)
        ("Cart Carol",  "carol@x.com", "new",       "unpaid", 6997),
        ("Paid Pat",    "pat@x.com",   "new",       "paid",   5000),
        ("Packed Peg",  "peg@x.com",   "packed",    "paid",   3000),
        ("Shipped Sam", "sam@x.com",   "shipped",   "paid",   2000),
        ("Done Dan",    "dan@x.com",   "done",      "paid",   1000),
        ("Cancel Cal",  "cal@x.com",   "cancelled", "unpaid", 900),
        ("Prop Pria",   "pria@x.com",  "proposed",  "unpaid", 800),
        ("Deliv Dev",   "dev@x.com",   "delivered", "paid",   700),
    ]
    for i, (nm, em, st, ps, tc) in enumerate(rows):
        cx.execute("INSERT INTO orders (source,external_ref,name,email,status,pay_status,total_cents,"
                   "items_json,address_json,created_at) VALUES (?,?,?,?,?,?,?,'[]','{}',?)",
                   ("test", str(i), nm, em, st, ps, tc, "2026-06-26T00:00:00+00:00"))
    cx.commit(); cx.close()


def test_attention_orders_returns_open_only(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed(tmp_path / "chat_log.db")
    res = o.attention_orders()
    statuses = sorted(r["status"] for r in res)
    assert statuses == ["new", "new", "packed", "proposed"]   # excludes shipped/delivered/done/cancelled
    r0 = res[0]
    assert set(r0) == {"id", "name", "email", "status", "pay_status",
                       "total_cents", "created_at", "backorder_units"}
    assert r0["backorder_units"] == 0


def test_attention_orders_respects_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = tmp_path / "chat_log.db"
    cx = sqlite3.connect(str(db)); o.init_orders_table(cx); o.init_fulfillments_table(cx)
    for i in range(30):
        cx.execute("INSERT INTO orders (source,external_ref,name,status,pay_status,total_cents,"
                   "items_json,address_json,created_at) VALUES (?,?,?,?,?,?,'[]','{}',?)",
                   ("test", str(i), "c%d" % i, "new", "unpaid", 100, "2026-06-26T00:00:00+00:00"))
    cx.commit(); cx.close()
    assert len(o.attention_orders(limit=20)) == 20
