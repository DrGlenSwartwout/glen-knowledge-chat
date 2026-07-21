import sqlite3
from dashboard import client_360


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, "
               "status TEXT, pay_status TEXT, invoice_sent_at TEXT)")
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY, email TEXT, scan_date TEXT)")
    cx.execute("CREATE TABLE ff_match_drafts (email TEXT, scan_date TEXT, status TEXT)")
    cx.execute("CREATE TABLE intake_responses (email TEXT PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE inquiries (id INTEGER PRIMARY KEY, client_email TEXT)")
    return cx


def _stage(res, key):
    return next(s for s in res["stages"] if s["key"] == key)


def test_no_data_all_pending_no_source():
    res = client_360.process_strip(_cx(), "nobody@example.com")
    assert res["source"] is None
    assert res["order_id"] is None
    assert [s["key"] for s in res["stages"]] == ["recommendation", "invoice", "sent", "paid", "fulfilled"]
    assert all(s["done"] is False for s in res["stages"])
    assert _stage(res, "recommendation")["action"]["kind"] == "none"


def test_source_priority_biofield_over_scan():
    cx = _cx()
    cx.execute("INSERT INTO biofield_reveals (id, email, scan_date) VALUES (1, 'a@b.com', '2026-07-01')")
    cx.execute("INSERT INTO ff_match_drafts (email, scan_date, status) VALUES ('a@b.com', '2026-07-01', 'draft')")
    res = client_360.process_strip(cx, "A@B.com")
    assert res["source"] == "biofield"
    rec = _stage(res, "recommendation")
    assert rec["done"] is True
    assert rec["action"] == {"kind": "link", "target": "/console/biofield-portal"}


def test_scan_source_when_no_biofield():
    cx = _cx()
    cx.execute("INSERT INTO ff_match_drafts (email, scan_date, status) VALUES ('a@b.com', '2026-07-01', 'draft')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] == "scan"
    assert _stage(res, "recommendation")["action"]["target"] == "/console/ff-drafts"


def test_intake_source():
    cx = _cx()
    cx.execute("INSERT INTO intake_responses (email, status) VALUES ('a@b.com', 'submitted')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] == "intake"


def test_intake_draft_is_not_a_source():
    cx = _cx()
    cx.execute("INSERT INTO intake_responses (email, status) VALUES ('a@b.com', 'draft')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] is None


def test_chat_source():
    cx = _cx()
    cx.execute("INSERT INTO inquiries (id, client_email) VALUES (1, 'a@b.com')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] == "chat"


def test_money_stages_from_latest_order():
    cx = _cx()
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at) "
               "VALUES ('a@b.com', 'cancelled', 'unpaid', NULL)")  # ignored
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at) "
               "VALUES ('a@b.com', 'confirmed', 'unpaid', '2026-07-10')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["order_id"] == 2
    assert _stage(res, "invoice")["done"] is True
    assert _stage(res, "sent")["done"] is True          # invoice_sent_at present
    assert _stage(res, "paid")["done"] is False
    assert _stage(res, "fulfilled")["done"] is False


def test_unsent_invoice_offers_send_dispatch():
    cx = _cx()
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at) "
               "VALUES ('a@b.com', 'confirmed', 'unpaid', NULL)")
    res = client_360.process_strip(cx, "a@b.com")
    sent = _stage(res, "sent")
    assert sent["done"] is False
    assert sent["action"] == {"kind": "dispatch", "target": "orders.send_invoice", "order_id": 1}


def test_paid_and_fulfilled():
    cx = _cx()
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at) "
               "VALUES ('a@b.com', 'shipped', 'paid', '2026-07-10')")
    res = client_360.process_strip(cx, "a@b.com")
    assert _stage(res, "paid")["done"] is True
    assert _stage(res, "fulfilled")["done"] is True


def test_missing_recommendation_tables_do_not_raise():
    # Bare orders-only db (no biofield_reveals/ff_match_drafts/etc.)
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, status TEXT, "
               "pay_status TEXT, invoice_sent_at TEXT)")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["source"] is None


def test_process_strip_returns_multi_sources():
    from dashboard import client_360
    import sqlite3
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, status TEXT, pay_status TEXT, invoice_sent_at TEXT)")
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY, email TEXT, scan_date TEXT)")
    cx.execute("CREATE TABLE ff_match_drafts (email TEXT, scan_date TEXT, status TEXT)")
    cx.execute("CREATE TABLE intake_responses (email TEXT PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE inquiries (id INTEGER PRIMARY KEY, client_email TEXT)")
    cx.execute("INSERT INTO biofield_reveals VALUES (1,'a@b.com','2026-07-01')")
    cx.execute("INSERT INTO ff_match_drafts VALUES ('a@b.com','2026-07-01','draft')")
    res = client_360.process_strip(cx, "a@b.com")
    assert res["sources"] == ["biofield", "scan"]     # all present, priority order
    assert res["source"] == "biofield"                # back-compat: first
    rec = next(s for s in res["stages"] if s["key"] == "recommendation")
    assert rec["sources"] == ["biofield", "scan"] and rec["done"] is True
