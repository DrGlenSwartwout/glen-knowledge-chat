"""Reconcile QBO-side invoice payments back onto the orders board.

`portal-reorder` / `reorder` orders create a QBO invoice (external_ref = the QBO
invoice Id) with a hosted online-pay link. When the client pays via that link,
QBO records the payment but nothing flips the board order to paid — so paid
orders sit in the Cart/Unpaid lane forever (observed: Mary Boyd order #7, paid in
QBO, Unpaid on the board). This module polls each open QBO-invoice order's live
balance and marks the paid ones. Pure/injectable so it's unit-tested offline;
the route wires in the real `qbo_billing.get_invoice` + `orders.set_order_payment`.
"""
import sqlite3

# Order sources whose external_ref is a QBO invoice Id (they call create_invoice).
QBO_SOURCES = ("reorder", "portal-reorder")


def list_open_qbo_orders(cx, sources=QBO_SOURCES):
    """Orders from QBO-invoice sources that are not yet paid, not cancelled/done,
    and whose external_ref is a numeric QBO invoice id."""
    cx.row_factory = sqlite3.Row
    ph = ",".join("?" * len(sources))
    rows = cx.execute(
        f"SELECT id, source, external_ref, total_cents, status, pay_status "
        f"FROM orders WHERE source IN ({ph}) "
        "AND COALESCE(pay_status,'unpaid') != 'paid' "
        "AND status NOT IN ('cancelled', 'done') "
        "AND external_ref NOT GLOB '*[^0-9]*' "
        "AND length(external_ref) < 20 ",
        tuple(sources)).fetchall()
    return [dict(r) for r in rows]


def reconcile_qbo_payments(cx, *, get_invoice, mark_paid, sources=QBO_SOURCES):
    """For each open QBO-invoice order, read the live invoice balance; if it's paid
    (balance <= 0), mark the board order paid. Best-effort per order — a single bad
    invoice (404 / transient auth) is logged and skipped, never aborts the batch.

    get_invoice(external_ref) -> dict (QBO invoice, possibly wrapped in {"Invoice": ...})
    mark_paid(cx, order_id, *, method, amount_cents) -> None
    Returns the list of reconciled orders: [{order_id, external_ref, doc_number, amount_cents}].
    """
    reconciled = []
    for o in list_open_qbo_orders(cx, sources):
        try:
            inv = get_invoice(o["external_ref"])
            i = inv.get("Invoice", inv) if isinstance(inv, dict) else inv
            balance = float(i.get("Balance", i.get("TotalAmt", 0)) or 0)
            if balance > 0:
                continue
            amount_cents = int(o.get("total_cents")
                               or round(float(i.get("TotalAmt") or 0) * 100))
            mark_paid(cx, o["id"], method="qbo", amount_cents=amount_cents)
            reconciled.append({"order_id": o["id"], "external_ref": o["external_ref"],
                               "doc_number": i.get("DocNumber"), "amount_cents": amount_cents})
        except Exception as e:
            print(f"[qbo-reconcile] order {o.get('id')} ref {o.get('external_ref')}: {e!r}",
                  flush=True)
    return reconciled
