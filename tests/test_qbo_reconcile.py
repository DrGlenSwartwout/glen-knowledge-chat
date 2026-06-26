"""QBO payment reconciliation: portal-reorder/reorder orders create a QBO invoice
with a hosted pay link, but a QBO-side payment never synced back to the board, so
paid orders showed Unpaid (e.g. Mary Boyd #7 paid in QBO, Unpaid on the board).
This pass polls each open QBO-invoice order's live balance and marks the paid ones."""
import sqlite3
from dashboard.qbo_reconcile import list_open_qbo_orders, reconcile_qbo_payments


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE orders (
        id INTEGER PRIMARY KEY, source TEXT, external_ref TEXT, total_cents INTEGER,
        status TEXT, pay_status TEXT)""")
    cx.executemany(
        "INSERT INTO orders (id, source, external_ref, total_cents, status, pay_status) VALUES (?,?,?,?,?,?)",
        [
            (5, "portal-reorder", "24435", 35982, "new", "unpaid"),   # genuinely unpaid
            (7, "portal-reorder", "24437", 31985, "new", "unpaid"),   # PAID in QBO -> reconcile
            (8, "reorder",        "24500", 5000,  "new", "paid"),     # already paid -> skip
            (9, "portal-reorder", "24439", 1000,  "cancelled", "unpaid"),  # cancelled -> skip
            (10, "biofield_trial","pi_abc", 100,  "new", "unpaid"),   # not a QBO source / non-numeric -> skip
        ])
    cx.commit()
    return cx


def test_list_open_qbo_orders_filters():
    ids = {o["id"] for o in list_open_qbo_orders(_cx())}
    assert ids == {5, 7}   # paid(8), cancelled(9), non-qbo(10) excluded


def test_reconcile_marks_only_qbo_paid_orders():
    cx = _cx()
    balances = {"24435": 359.82, "24437": 0.0}    # 24437 is paid in QBO

    def fake_get_invoice(ref):
        return {"Balance": balances[ref], "TotalAmt": {"24435": 359.82, "24437": 319.85}[ref],
                "DocNumber": {"24435": "1017", "24437": "1019"}[ref]}

    marked = []

    def fake_mark_paid(cx_, oid, *, method, amount_cents):
        marked.append((oid, method, amount_cents))

    out = reconcile_qbo_payments(cx, get_invoice=fake_get_invoice, mark_paid=fake_mark_paid)
    assert [m[0] for m in marked] == [7]              # only the QBO-paid order
    assert marked[0] == (7, "qbo", 31985)
    assert [r["order_id"] for r in out] == [7]
    assert out[0]["doc_number"] == "1019"


def test_one_bad_invoice_does_not_stop_the_rest():
    cx = _cx()

    def flaky_get_invoice(ref):
        if ref == "24435":
            raise RuntimeError("QBO 404")
        return {"Balance": 0.0, "TotalAmt": 319.85, "DocNumber": "1019"}

    out = reconcile_qbo_payments(cx, get_invoice=flaky_get_invoice, mark_paid=lambda *a, **k: None)
    assert [r["order_id"] for r in out] == [7]        # 24435 error skipped, 24437 still reconciled
