import json, sqlite3
from dashboard import orders as O


def _fresh_db(tmp_path):
    db = str(tmp_path / "orders.db")
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    return cx


def _new_order(cx, ref="tok-abc"):
    return O.upsert_order(cx, source="biofield", external_ref=ref, email="a@b.com",
                          total_cents=30000)


def test_new_columns_exist(tmp_path):
    cx = _fresh_db(tmp_path)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(orders)")}
    assert "qbo_lines_json" in cols
    assert "qbo_sales_receipt_id" in cols


def test_set_and_read_qbo_lines_by_ref(tmp_path):
    cx = _fresh_db(tmp_path)
    _new_order(cx, "tok-1")
    payload = {"lines": [{"name": "Biofield", "amount": 300.0, "qty": 1}],
               "discount_cents": 0, "tax_cents": 0}
    assert O.set_order_qbo_lines(cx, "tok-1", payload) is True
    row = O.find_order_by_external_ref(cx, "tok-1")
    assert json.loads(row["qbo_lines_json"]) == payload


def test_set_sales_receipt_id_stamps_order(tmp_path):
    cx = _fresh_db(tmp_path)
    oid = _new_order(cx, "tok-2")
    assert O.set_order_sales_receipt_id(cx, oid, "SR123") is True
    assert O.get_order(cx, oid)["qbo_sales_receipt_id"] == "SR123"


def test_set_qbo_lines_unknown_ref_returns_false(tmp_path):
    cx = _fresh_db(tmp_path)
    assert O.set_order_qbo_lines(cx, "nope", {"lines": []}) is False


def test_claim_sales_receipt_slot_is_single_winner(tmp_path):
    cx = _fresh_db(tmp_path)
    oid = _new_order(cx, "tok-3")
    assert O.claim_sales_receipt_slot(cx, oid) is True
    assert O.get_order(cx, oid)["qbo_sales_receipt_id"] == "PENDING"
    # a second caller (refresh/race) must lose the claim
    assert O.claim_sales_receipt_slot(cx, oid) is False
