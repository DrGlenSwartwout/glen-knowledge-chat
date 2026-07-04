import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import backfill_dispensary_referrals as bf
from dashboard import referrals as rf


def _seed(tmp_path):
    p = os.path.join(str(tmp_path), "chat_log.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE dispensary_orders(invoice_id TEXT PRIMARY KEY, practitioner_id TEXT, "
        "customer_email TEXT, bottles INT, credit_earned_cents INT, created_at TEXT);")
    cx.execute("INSERT INTO dispensary_orders VALUES('D1','p1','a@x.com',1,0,'t')")
    cx.execute("INSERT INTO dispensary_orders VALUES('D2','p1','a@x.com',1,0,'t')")  # dup pair
    cx.execute("INSERT INTO dispensary_orders VALUES('D3','p2','b@x.com',1,0,'t')")
    cx.execute("INSERT INTO dispensary_orders VALUES('D4','p9','c@x.com',1,0,'t')")  # unresolved
    cx.commit(); cx.close()
    return p


EMAILS = {"p1": "doc1@x.com", "p2": "doc2@x.com", "p9": ""}   # p9 has no email


def test_backfill_writes_one_row_per_pair(tmp_path):
    p = _seed(tmp_path)
    res = bf.backfill(p, EMAILS.get)
    assert res["written"] == 2 and res["unresolved"] == 1
    with sqlite3.connect(p) as cx:
        assert rf.owner_of_referee(cx, "a@x.com") == "doc1@x.com"
        assert rf.owner_of_referee(cx, "b@x.com") == "doc2@x.com"
        arow = rf.redemption_by_order_ref(cx, "D1") or rf.redemption_by_order_ref(cx, "D2")
        assert arow["kind"] == "dispensary_portal"
        assert not (arow["rewarded_at"] or "")   # attribution only, no reward stamp


def test_backfill_idempotent(tmp_path):
    p = _seed(tmp_path)
    bf.backfill(p, EMAILS.get)
    res2 = bf.backfill(p, EMAILS.get)
    assert res2["written"] == 0 and res2["skipped"] >= 2   # PK already present


def test_backfill_dry_run_writes_nothing(tmp_path):
    p = _seed(tmp_path)
    res = bf.backfill(p, EMAILS.get, dry_run=True)
    assert res["written"] == 2   # would-write count
    with sqlite3.connect(p) as cx:
        assert rf.redemption_by_order_ref(cx, "D1") is None
