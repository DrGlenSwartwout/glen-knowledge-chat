"""Stripe payments ledger — a money-first read over the orders table.

The console has order boards but no transaction ledger. This model exposes the
subset of orders that captured card money via Stripe, plus the recent
`stripe_failures` (declined/failed charges) so failed subscription renewals can't
go unseen. Read-only: nothing here moves money or mutates orders.

A "payment" is an order where ANY of:
  - `stripe_payment_intent` is set (one-time checkout: retail funnel / in-house
    card — set via orders.set_order_stripe_pi), OR
  - `external_ref` is a Stripe PaymentIntent id (`pi_...`) — every recurring charge
    (subscription renewals AND the $99/mo coaching `membership` charge) is ingested
    only when the PaymentIntent SUCCEEDS, storing the PI id in `external_ref`, NOT
    the column, OR
  - `source` is a known recurring-charge source — belt-and-suspenders for the rare
    case where the QBO invoice id was stored instead of the PI id.
A $0 `founding` card-vault reservation (external_ref is a `seti_` SetupIntent) is
NOT a captured charge and is excluded. Caller sets cx.row_factory = sqlite3.Row.
"""

# Sources whose orders are, by construction, succeeded recurring card charges.
_RECURRING_SOURCES = ("subscription", "membership")

# An order counts as a captured Stripe payment when this predicate holds.
_CAPTURED = ("((stripe_payment_intent IS NOT NULL AND stripe_payment_intent != '') "
             "OR external_ref LIKE 'pi\\_%' ESCAPE '\\' "
             "OR source IN ('subscription', 'membership'))")


def _pi(row):
    """The PaymentIntent id to display: the column when set, else the
    external_ref for subscription rows (which carry `pi_...` there)."""
    col = (row["stripe_payment_intent"] or "").strip() if row["stripe_payment_intent"] else ""
    if col:
        return col
    ref = (row["external_ref"] or "").strip()
    return ref if ref.startswith("pi_") else ""


def _row_to_payment(row):
    paid = int(row["paid_cents"] or 0)
    total = int(row["total_cents"] or 0)
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "paid_at": row["paid_at"],
        "email": row["email"] or "",
        "name": row["name"] or "",
        "source": row["source"],
        "channel": row["channel"] or "",
        "amount_cents": paid if paid else total,
        "pay_status": row["pay_status"] or "",
        "stripe_payment_intent": _pi(row),
        "external_ref": row["external_ref"] or "",
    }


def list_payments(cx, *, source=None, limit=200):
    """Captured Stripe payments, newest first. Optional exact `source` filter
    (e.g. 'subscription', 'funnel'). Newest = most recent paid_at, falling back
    to created_at, then id."""
    sql = f"SELECT * FROM orders WHERE {_CAPTURED}"
    params = []
    if source:
        sql += " AND source = ?"
        params.append(source)
    sql += " ORDER BY COALESCE(paid_at, created_at) DESC, id DESC LIMIT ?"
    params.append(int(limit))
    return [_row_to_payment(r) for r in cx.execute(sql, params).fetchall()]


def payments_summary(cx):
    """Count + summed amount over the captured set (paid_cents, else total_cents)."""
    row = cx.execute(
        f"SELECT COUNT(*) AS n, "
        f"COALESCE(SUM(CASE WHEN paid_cents > 0 THEN paid_cents ELSE total_cents END), 0) AS cents "
        f"FROM orders WHERE {_CAPTURED}").fetchone()
    return {"count": int(row["n"] or 0), "total_cents": int(row["cents"] or 0)}


def recent_failures(cx, *, limit=20):
    """Recent Stripe failures (declined/failed charges), newest first. Empty list
    if the table is missing or empty."""
    try:
        rows = cx.execute(
            "SELECT id, context, error, created_at, emailed_at FROM stripe_failures "
            "ORDER BY created_at DESC, id DESC LIMIT ?", (int(limit),)).fetchall()
    except Exception:
        return []
    return [{"id": r["id"], "context": r["context"] or "", "error": r["error"] or "",
             "created_at": r["created_at"], "emailed_at": r["emailed_at"]} for r in rows]
