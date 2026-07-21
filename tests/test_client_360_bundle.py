import sqlite3
import pytest
from dashboard import client_360


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT, name TEXT, phone TEXT, "
               "city TEXT, state TEXT, island TEXT, profession TEXT, order_count INTEGER, last_order_date TEXT)")
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, status TEXT, "
               "pay_status TEXT, invoice_sent_at TEXT, total_cents INTEGER, created_at TEXT, "
               "items_json TEXT DEFAULT '[]', address_json TEXT DEFAULT '{}')")
    cx.execute("CREATE TABLE order_payments (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, "
               "kind TEXT, amount_cents INTEGER, status TEXT DEFAULT 'active')")
    cx.execute("CREATE TABLE client_scans (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, scan_date TEXT, scan_id TEXT)")
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, scan_date TEXT, "
               "interpretation_json TEXT DEFAULT '{}', remedies_json TEXT DEFAULT '[]', first_approved INTEGER DEFAULT 0, "
               "created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '', dropped TEXT DEFAULT '[]', "
               "layers_json TEXT DEFAULT '[]', notified_at TEXT, requested_at TEXT, token_hash TEXT, "
               "approved_at TEXT, approved_by TEXT)")
    cx.execute("CREATE TABLE inquiries (id INTEGER PRIMARY KEY AUTOINCREMENT, client_email TEXT, "
               "main_challenge TEXT, main_goal TEXT, created_at TEXT)")
    return cx


def test_bundle_shape_and_invoice_math(tmp_path):
    cx = _cx()
    cx.execute("INSERT INTO people (id,email,name,phone,city,state,island,profession,order_count,last_order_date) "
               "VALUES (1,'a@b.com','Al Bee','808','Hilo','HI','Big Island','yoga',2,'2026-07-10')")
    cx.execute("INSERT INTO orders (email,status,pay_status,invoice_sent_at,total_cents,created_at) "
               "VALUES ('a@b.com','confirmed','unpaid','2026-07-10',10000,'2026-07-09')")
    cx.execute("INSERT INTO order_payments (order_id,kind,amount_cents,status) VALUES (1,'payment',4000,'active')")
    cx.execute("INSERT INTO client_scans (email,scan_date,scan_id) VALUES ('a@b.com','2026-07-01','s1')")
    cx.execute("INSERT INTO biofield_reveals (email,scan_date) VALUES ('a@b.com','2026-07-05')")
    cx.execute("INSERT INTO inquiries (client_email,main_challenge,main_goal,created_at) "
               "VALUES ('a@b.com','fatigue','energy','2026-07-08 12:00:00')")
    b = client_360.bundle(cx, "a@b.com", e4l_path=str(tmp_path / "missing.db"))

    assert b["person"]["name"] == "Al Bee"
    assert b["person"]["location"] == "Hilo, HI"
    assert b["clinical"] == {"active": [], "suggested": []}   # no e4l db -> empty
    # tests: newest first, biofield 07-05 before scan 07-01
    assert [(t["date"], t["type"]) for t in b["tests"]] == [("2026-07-05", "biofield"), ("2026-07-01", "scan")]
    # invoices: 100.00 total, 40.00 paid, 60.00 balance
    assert b["invoices"]["total_paid_cents"] == 4000
    assert b["invoices"]["open_balance_cents"] == 6000
    o = b["invoices"]["orders"][0]
    assert (o["total_cents"], o["paid_cents"], o["balance_cents"]) == (10000, 4000, 6000)
    assert o["edit_url"] == "/orders/new?edit_order=1"
    # comms include the inquiry
    assert any(c["source"] == "inquiry" and c["topic"] for c in b["comms"])
    # process present
    assert b["process"]["source"] == "biofield"


def test_bundle_empty_client(tmp_path):
    cx = _cx()
    b = client_360.bundle(cx, "nobody@x.com", e4l_path=str(tmp_path / "missing.db"))
    assert b["person"]["name"] == ""
    assert b["tests"] == []
    assert b["invoices"] == {"total_paid_cents": 0, "open_balance_cents": 0, "orders": [], "fmp": []}
    assert b["comms"] == []
    assert b["process"]["source"] is None
