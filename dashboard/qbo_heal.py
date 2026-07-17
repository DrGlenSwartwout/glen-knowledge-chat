"""QBO auto-heal sweep for orders stuck at qbo_sales_receipt_id='PENDING'.

A PENDING order means claim_sales_receipt_slot() won the atomic claim but the
process died (crash, timeout, deploy) before set_order_sales_receipt_id()
overwrote 'PENDING' with a real Id or the booking failed outright. This sweep
resolves each such order to one of:

  Case B (a receipt already exists -- e.g. the booking DID complete in QBO but
  the process died before stamping the local row): find_receipt() returns the
  SalesReceipt dict carrying this order's exact token. We stamp its Id and do
  NOT call book() -- calling book() here would create a SECOND receipt for the
  same order (a double-book).

  Case A (definitely no receipt exists): find_receipt() returns None. We clear
  qbo_sales_receipt_id back to NULL and call book(), which re-claims the slot
  and books+stamps a fresh receipt.

  Inconclusive (the lookup itself failed -- a transient QBO error): find_receipt()
  raises. We must NOT treat this as "no receipt" (that would risk a double-book
  if a receipt actually exists but the query couldn't confirm it). The order is
  left PENDING and skipped; the next sweep run will retry it.

Only orders whose qbo_sales_receipt_id is exactly 'PENDING' and whose
updated_at is older than `older_than_min` are ever touched, so an in-flight
booking (PENDING for a few seconds) is never disturbed.
"""
import datetime
import sqlite3


def heal_pending_receipts(cx, *, find_receipt, book, stamp, older_than_min=10, now=None):
    """Sweep orders stuck at qbo_sales_receipt_id='PENDING' older than
    `older_than_min` minutes and resolve each to a stamped or rebooked receipt.

    find_receipt(token, email=, since_date=) -- returns the QBO SalesReceipt
        dict if one carrying this order's token exists, None if it definitely
        does not, and RAISES if the lookup could not be completed.
    book(cx, order) -- re-claims + books + stamps a fresh Sales Receipt for the
        order (qbo_sale.book_sale_on_payment). Returns the new receipt Id.
    stamp(cx, order_id, receipt_id) -- records an existing receipt's Id onto
        the order (orders.set_order_sales_receipt_id).

    Best-effort per order: any error (including find_receipt raising) is
    caught, logged, and the order is left untouched (still PENDING) for the
    next sweep. The sweep itself never raises.

    Returns a list of {"order_id", "action", "receipt_id"} dicts, one per
    order successfully resolved this run (action is "stamped" or "rebooked").
    Skipped/errored orders are not included.
    """
    cutoff = (now or datetime.datetime.utcnow()) - datetime.timedelta(minutes=older_than_min)
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT * FROM orders WHERE qbo_sales_receipt_id='PENDING' AND updated_at < ?",
        (cutoff.isoformat(),)).fetchall()
    out = []
    for r in rows:
        o = dict(r)
        try:
            token = o.get("external_ref")
            existing = find_receipt(token, email=o.get("email"),
                                    since_date=(o.get("created_at") or "")[:10])
            if existing and existing.get("Id"):
                stamp(cx, o["id"], existing["Id"])
                out.append({"order_id": o["id"], "action": "stamped",
                           "receipt_id": existing["Id"]})
            else:
                cx.execute("UPDATE orders SET qbo_sales_receipt_id=NULL WHERE id=?", (o["id"],))
                cx.commit()
                o["qbo_sales_receipt_id"] = None
                sr = book(cx, o)
                out.append({"order_id": o["id"], "action": "rebooked", "receipt_id": sr})
        except Exception as e:
            print(f"[qbo-heal] order {o.get('id')!r} skipped: {e!r}", flush=True)
    return out
