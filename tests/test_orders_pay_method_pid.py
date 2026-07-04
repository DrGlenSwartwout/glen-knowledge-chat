import sqlite3
from dashboard import orders as o


def _cx():
    cx = sqlite3.connect(":memory:")
    o.init_orders_table(cx)
    return cx


def test_practitioner_id_column_exists():
    cx = _cx()
    cols = {r[1] for r in cx.execute("PRAGMA table_info(orders)")}
    assert "practitioner_id" in cols and "pay_method" in cols


def test_upsert_writes_pay_method_and_pid_on_insert():
    cx = _cx()
    o.upsert_order(cx, source="dispensary", external_ref="INV1", email="p@x.com",
                   total_cents=7000, pay_method="zelle", practitioner_id="prac-1")
    row = cx.execute("SELECT pay_method, practitioner_id FROM orders WHERE external_ref='INV1'").fetchone()
    assert row == ("zelle", "prac-1")


def test_upsert_none_does_not_clobber_existing():
    cx = _cx()
    o.upsert_order(cx, source="dispensary", external_ref="INV2", pay_method="card",
                   practitioner_id="prac-2", total_cents=100)
    # a later ingest without the fields (None) must not wipe them
    o.upsert_order(cx, source="dispensary", external_ref="INV2", total_cents=200)
    row = cx.execute("SELECT pay_method, practitioner_id, total_cents FROM orders WHERE external_ref='INV2'").fetchone()
    assert row == ("card", "prac-2", 200)
