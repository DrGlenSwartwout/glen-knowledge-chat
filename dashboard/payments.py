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

# The $1 biofield-trial charge (app.py creates the checkout session at 100 cents).
# Used as the amount fallback when a Stripe session/PI omits the amount.
TRIAL_AMOUNT_CENTS = 100

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


def _status(row):
    """Display status. Every ledger row is a SUCCEEDED capture (failed charges
    never create orders, and recurring orders are ingested only on success), so
    the orders-table default 'unpaid' is meaningless here — report 'paid'. Any
    other explicit value (e.g. a future 'refunded') is preserved."""
    s = (row["pay_status"] or "").strip()
    return s if s and s != "unpaid" else "paid"


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
        "pay_status": _status(row),
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


def backfill_trial_orders(cx, fetch_session, *, dry_run=False, now=None):
    """One-time backfill: ensure every historical $1 biofield trial has a
    captured-charge order so it shows in the ledger (going-forward trials get one
    at fulfillment, but trials completed before that shipped have none).

    Reads `biofield_trial_grants` (the per-trial idempotency markers), and for
    each re-fetches the Stripe checkout session via `fetch_session(session_id)`
    (shape: stripe_pay.get_session — payment_intent, amount_total). Idempotent on
    (source='biofield_trial', external_ref=PaymentIntent). Never raises; a per-row
    fetch error is counted, not fatal. Returns {created, skipped, unpaid, failed}.

    dry_run=True does every read + Stripe fetch but writes nothing; 'created' then
    counts what WOULD be created."""
    from dashboard import orders as O
    if not dry_run:
        O.init_orders_table(cx)
    try:
        grants = cx.execute(
            "SELECT session_id, email FROM biofield_trial_grants").fetchall()
    except Exception:
        return {"created": 0, "skipped": 0, "unpaid": 0, "failed": 0}
    out = {"created": 0, "skipped": 0, "unpaid": 0, "failed": 0, "reconciled": 0}
    # Reconcile trial orders created before they were marked paid: a biofield_trial
    # order only exists because its $1 was captured, so an 'unpaid' one is a stale
    # flag (it showed as "Unpaid" on the Done board). Flip it to paid WITHOUT
    # moving it off 'done'. Idempotent — a re-run touches only still-unpaid rows.
    if not dry_run:
        try:
            for r in cx.execute(
                    "SELECT id, total_cents FROM orders WHERE source='biofield_trial' "
                    "AND COALESCE(pay_status,'unpaid')!='paid'").fetchall():
                oid = r["id"] if hasattr(r, "keys") else r[0]
                amt = int((r["total_cents"] if hasattr(r, "keys") else r[1]) or 0) or TRIAL_AMOUNT_CENTS
                O.mark_order_paid_keep_status(cx, oid, method="card", amount_cents=amt)
                out["reconciled"] += 1
        except Exception as e:
            print(f"[trial-backfill] reconcile pass skipped: {e!r}", flush=True)
    for g in grants:
        sid = g["session_id"] if hasattr(g, "keys") else g[0]
        email = g["email"] if hasattr(g, "keys") else g[1]
        try:
            sess = fetch_session(sid) or {}
            pi_id = (sess.get("payment_intent") or "").strip()
            if not pi_id:
                out["unpaid"] += 1
                continue
            exists = cx.execute(
                "SELECT 1 FROM orders WHERE source='biofield_trial' AND external_ref=?",
                (pi_id,)).fetchone()
            if exists:
                out["skipped"] += 1
                continue
            amount = int(sess.get("amount_total") or 0) or TRIAL_AMOUNT_CENTS
            if not dry_run:
                oid = O.upsert_order(cx, source="biofield_trial", external_ref=pi_id,
                                     email=email or "", items=[], total_cents=amount,
                                     address={}, channel="retail", status="done")
                # It's a captured $1 charge — record it paid without leaving 'done'.
                O.mark_order_paid_keep_status(cx, oid, method="card", amount_cents=amount)
            out["created"] += 1
        except Exception as e:
            print(f"[trial-backfill] {sid}: {e!r}", flush=True)
            out["failed"] += 1
    if not dry_run:
        cx.commit()
    return out


def reconcile_captured_charges(cx, get_payment_intent, *, dry_run=False):
    """Mark orders paid when they carry a genuinely-captured Stripe charge but were
    left pay_status='unpaid'. Some digital/checkout flows (biofield unlock, prepay
    term membership) create the order with the PaymentIntent as its ref but never
    record the payment, so it shows "Unpaid" on the Done board.

    For each unpaid order that has a PaymentIntent (the stripe_payment_intent column,
    or an external_ref that looks like one), verify status=='succeeded' via
    get_payment_intent, then record payment WITHOUT moving the order off its current
    status (keeps 'done'/etc). Only Stripe-verified 'succeeded' charges are touched.
    Idempotent (already-paid rows are excluded); never raises per row. Returns
    {reconciled, skipped, unverified, failed, orders:[{id,name,amount_cents}]}."""
    from dashboard import orders as O
    out = {"reconciled": 0, "skipped": 0, "unverified": 0, "failed": 0, "orders": []}
    try:
        rows = cx.execute(
            "SELECT id, external_ref, stripe_payment_intent, total_cents, name FROM orders "
            "WHERE COALESCE(pay_status,'unpaid')!='paid' AND ("
            "  (stripe_payment_intent IS NOT NULL AND TRIM(stripe_payment_intent)!='') "
            "  OR external_ref LIKE 'pi\\_%' ESCAPE '\\')").fetchall()
    except Exception as e:
        print(f"[reconcile-captured] query skipped: {e!r}", flush=True)
        return out
    for r in rows:
        g = (lambda k, i: r[k] if hasattr(r, "keys") else r[i])
        oid = g("id", 0)
        ext = (g("external_ref", 1) or "").strip()
        pi_col = (g("stripe_payment_intent", 2) or "").strip()
        total = int(g("total_cents", 3) or 0)
        name = g("name", 4) or ""
        pi_id = pi_col or (ext if ext.startswith("pi_") else "")
        if not pi_id:
            out["skipped"] += 1
            continue
        try:
            pi = get_payment_intent(pi_id) or {}
        except Exception as e:
            print(f"[reconcile-captured] #{oid} PI fetch failed: {e!r}", flush=True)
            out["failed"] += 1
            continue
        if (pi.get("status") or "") != "succeeded":
            out["unverified"] += 1
            continue
        if not dry_run:
            O.mark_order_paid_keep_status(cx, oid, method="card", amount_cents=total)
        out["reconciled"] += 1
        out["orders"].append({"id": oid, "name": name, "amount_cents": total})
    return out


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
