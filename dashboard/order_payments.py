"""Multi-payment + refund ledger per order. One row per payment or refund;
balances are always derived, never stored. Functions take a sqlite connection
for testability. QBO/Stripe sync lives in this module (Tasks 2-3) and is called
as module functions so tests can monkeypatch them."""
import logging
from datetime import datetime, timezone

from dashboard import orders, qbo_billing, stripe_pay

log = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc).isoformat()


def ensure_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS order_payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            kind TEXT NOT NULL DEFAULT 'payment',
            amount_cents INTEGER NOT NULL,
            method TEXT,
            source TEXT NOT NULL DEFAULT 'manual',
            external_ref TEXT,
            refunds_payment_id INTEGER,
            paid_at TEXT,
            note TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            void_reason TEXT,
            voided_at TEXT,
            qbo_txn_id TEXT,
            qbo_sync TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT,
            created_by TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_order_payments_order "
               "ON order_payments(order_id)")


def _row(cx, pid):
    r = cx.execute("SELECT * FROM order_payments WHERE id=?", (pid,)).fetchone()
    return dict(r) if r else None


def list_payments(cx, order_id):
    rows = cx.execute(
        "SELECT * FROM order_payments WHERE order_id=? ORDER BY id DESC",
        (order_id,)).fetchall()
    return [dict(r) for r in rows]


def ledger_rows_for_payments_view(cx, *, limit=200):
    """Active non-Stripe order_payments entries (Zelle/check/cash/etc.), mapped
    into the same dict shape as dashboard.payments._row_to_payment so the
    /api/payments money view can render Stripe charges and manual payments in
    one uniform list. Refunds are represented as a negative amount_cents.
    Newest first (paid_at, falling back to created_at). Voided/pending rows
    (status != 'active') are excluded. Stripe-sourced PAYMENTS are excluded —
    they already appear via dashboard.payments' charge ledger — but Stripe-sourced
    REFUNDS ARE included (a card refund has no row in the charge ledger, so this is
    its only appearance), tagged 'card:refund'. Read-only; caller sets
    cx.row_factory = sqlite3.Row."""
    rows = cx.execute(
        "SELECT op.id AS op_id, op.kind, op.amount_cents, op.method, op.source AS op_source, "
        "op.external_ref AS op_ref, op.paid_at, op.created_at, "
        "o.email, o.name, o.channel "
        "FROM order_payments op JOIN orders o ON o.id = op.order_id "
        # Non-Stripe rows (manual payments + refunds) PLUS Stripe REFUNDS. A card
        # refund is source='stripe' but has NO row in the Stripe charge ledger
        # (dashboard.payments tracks charges, not refunds), so include it here.
        # Stripe PAYMENTS (card charges) stay excluded — they're already there.
        "WHERE op.status='active' AND (op.source != 'stripe' OR op.kind='refund') "
        "AND op.kind IN ('payment','refund') "
        "ORDER BY COALESCE(op.paid_at, op.created_at) DESC, op.id DESC "
        "LIMIT ?", (int(limit),)).fetchall()
    out = []
    for r in rows:
        amt = int(r["amount_cents"] or 0)
        if r["kind"] == "refund":
            amt = -amt
        method = (r["method"] or "other").strip()
        if (r["op_source"] or "") == "stripe":
            tag = "card:refund"   # a Stripe-issued card refund (money out, shown negative)
        else:
            tag = "manual:" + (method.lower().replace(" ", "-") or "other")
        out.append({
            "id": f"op-{r['op_id']}",
            "created_at": r["created_at"],
            "paid_at": r["paid_at"],
            "email": r["email"] or "",
            "name": r["name"] or "",
            "source": tag,
            "channel": r["channel"] or "",
            "amount_cents": amt,
            "pay_status": "refunded" if r["kind"] == "refund" else "paid",
            "stripe_payment_intent": "",
            "external_ref": r["op_ref"] or "",
        })
    return out


def _sum(cx, order_id, kind):
    v = cx.execute(
        "SELECT COALESCE(SUM(amount_cents),0) FROM order_payments "
        "WHERE order_id=? AND kind=? AND status='active'",
        (order_id, kind)).fetchone()[0]
    return int(v or 0)


def balance(cx, order_id):
    o = orders.get_order(cx, order_id) or {}
    invoice = int(o.get("total_cents") or 0)
    paid = _sum(cx, order_id, "payment")
    refunded = _sum(cx, order_id, "refund")
    return {"invoice_cents": invoice, "paid_cents": paid,
            "refunded_cents": refunded,
            "balance_cents": invoice - (paid - refunded)}


def _insert(cx, order_id, *, kind, amount_cents, method, source, external_ref,
            refunds_payment_id, paid_at, note, actor):
    now = _now()
    cur = cx.execute(
        "INSERT INTO order_payments (order_id, kind, amount_cents, method, "
        "source, external_ref, refunds_payment_id, paid_at, note, status, "
        "qbo_sync, created_at, updated_at, created_by) "
        "VALUES (?,?,?,?,?,?,?,?,?,'active','pending',?,?,?)",
        (order_id, kind, int(amount_cents), method, source, external_ref,
         refunds_payment_id, paid_at or now, note, now, now, actor))
    cx.commit()
    return _row(cx, cur.lastrowid)


def _qbo_ctx(cx, order_id):
    """Resolve (customer_id, qbo_invoice_id) for an order via its stored QBO
    invoice ref (orders.external_ref). Returns (None, None) if unresolvable."""
    o = orders.get_order(cx, order_id) or {}
    inv_id = o.get("external_ref")
    if not inv_id:
        return None, None
    inv = qbo_billing.get_invoice(inv_id)
    if not inv:
        return None, inv_id
    return (inv.get("CustomerRef") or {}).get("value"), inv_id


def _mark_sync(cx, pid, *, qbo_txn_id=None, state):
    cx.execute("UPDATE order_payments SET qbo_txn_id=COALESCE(?, qbo_txn_id), "
               "qbo_sync=?, updated_at=? WHERE id=?",
               (qbo_txn_id, state, _now(), pid))
    cx.commit()


def _push_payment(cx, pid):
    row = _row(cx, pid)
    if row.get("qbo_txn_id"):
        return  # already synced — idempotent
    try:
        cid, inv_id = _qbo_ctx(cx, row["order_id"])
        if not cid or not inv_id:
            raise RuntimeError("no QBO invoice/customer for order")
        res = qbo_billing.record_payment(cid, row["amount_cents"], inv_id,
                                         method=row["method"])
        _mark_sync(cx, pid, qbo_txn_id=(res or {}).get("Id"), state="synced")
    except Exception:
        _mark_sync(cx, pid, state="error")


def add_payment(cx, order_id, amount_cents, method, *, source="manual",
                external_ref=None, paid_at=None, note=None, actor=None):
    if int(amount_cents) <= 0:
        raise ValueError("amount_cents must be positive")
    if external_ref:
        dup = cx.execute(
            "SELECT id FROM order_payments WHERE order_id=? AND kind='payment' "
            "AND external_ref=? AND status='active'",
            (order_id, external_ref)).fetchone()
        if dup:
            return _row(cx, dup[0])
    row = _insert(cx, order_id, kind="payment", amount_cents=amount_cents,
                  method=method, source=source, external_ref=external_ref,
                  refunds_payment_id=None, paid_at=paid_at, note=note,
                  actor=actor)
    _push_payment(cx, row["id"])
    return _row(cx, row["id"])


def refundable_cents(cx, order_id, refunds_payment_id=None):
    """How much can still be refunded. Against a specific payment: the lesser of
    that payment's un-refunded remainder and the order's net paid (a standalone
    refund already reduces net paid, so a payment-tied refund must not be able to
    jointly drive it negative). Otherwise (standalone): order net paid."""
    b = balance(cx, order_id)
    order_net_paid = max(0, b["paid_cents"] - b["refunded_cents"])
    if refunds_payment_id is not None:
        pay = _row(cx, refunds_payment_id)
        if not pay or pay["kind"] != "payment" or pay["status"] != "active":
            return 0
        used = cx.execute(
            "SELECT COALESCE(SUM(amount_cents),0) FROM order_payments "
            "WHERE refunds_payment_id=? AND kind='refund' AND status='active'",
            (refunds_payment_id,)).fetchone()[0]
        remainder = int(pay["amount_cents"]) - int(used or 0)
        return min(remainder, order_net_paid)
    return order_net_paid


def _push_refund(cx, pid):
    row = _row(cx, pid)
    if row.get("qbo_txn_id"):
        return
    try:
        cid, inv_id = _qbo_ctx(cx, row["order_id"])
        if not cid or not inv_id:
            raise RuntimeError("no QBO invoice/customer for order")
        res = qbo_billing.record_refund(cid, row["amount_cents"], inv_id,
                                        method=row["method"])
        _mark_sync(cx, pid, qbo_txn_id=(res or {}).get("Id"), state="synced")
    except Exception:
        _mark_sync(cx, pid, state="error")


def add_refund(cx, order_id, amount_cents, method, *, refunds_payment_id=None,
               note=None, actor=None):
    amt = int(amount_cents)
    if amt <= 0:
        raise ValueError("amount_cents must be positive")
    if amt > refundable_cents(cx, order_id, refunds_payment_id):
        raise ValueError("refund exceeds refundable amount")
    external_ref = None
    src_pay = _row(cx, refunds_payment_id) if refunds_payment_id else None
    if src_pay and src_pay.get("source") == "stripe" and src_pay.get("external_ref"):
        sr = stripe_pay.refund(src_pay["external_ref"], amount_cents=amt)
        external_ref = sr.get("id")
    try:
        row = _insert(cx, order_id, kind="refund", amount_cents=amt, method=method,
                      source=("stripe" if external_ref else "manual"),
                      external_ref=external_ref, refunds_payment_id=refunds_payment_id,
                      paid_at=None, note=note, actor=actor)
    except Exception:
        if external_ref:
            # Stripe already moved the money — this row is the only trace of it.
            # Losing the insert here means an unrecorded refund; log a breadcrumb
            # for manual reconciliation and re-raise (never swallow).
            log.error("Stripe refund %s succeeded for order %s but the ledger "
                      "insert failed — refund is unrecorded, needs manual "
                      "reconciliation", external_ref, order_id)
        raise
    _push_refund(cx, row["id"])
    return _row(cx, row["id"])


def void(cx, payment_id, reason, *, actor=None):
    # actor: reserved for a future voided_by column; not persisted yet
    row = _row(cx, payment_id)
    if not row or row["status"] == "void":
        return row
    if row.get("qbo_txn_id"):
        try:
            if row["kind"] == "refund":
                qbo_billing.void_refund(row["qbo_txn_id"])
            else:
                qbo_billing.void_payment(row["qbo_txn_id"])
            sync_state = "void_synced"
        except Exception:
            # keep the app void; row stays flagged for resync/repair
            sync_state = "void_error"
    else:
        sync_state = "void_synced"   # nothing to reverse in QBO
    cx.execute("UPDATE order_payments SET status='void', void_reason=?, "
               "voided_at=?, updated_at=?, qbo_sync=? WHERE id=?",
               (reason, _now(), _now(), sync_state, payment_id))
    cx.commit()
    return _row(cx, payment_id)


def resync(cx, payment_id):
    row = _row(cx, payment_id)
    if not row:
        return row
    if row["status"] == "void":
        # a void whose QBO reversal previously failed is repairable here —
        # never permanently stranded as 'void_error'.
        if row.get("qbo_sync") == "void_error" and row.get("qbo_txn_id"):
            try:
                if row["kind"] == "refund":
                    qbo_billing.void_refund(row["qbo_txn_id"])
                else:
                    qbo_billing.void_payment(row["qbo_txn_id"])
                _mark_sync(cx, payment_id, state="void_synced")
            except Exception:
                pass  # still void_error; another resync can retry later
        return _row(cx, payment_id)
    if row["kind"] == "payment":
        _push_payment(cx, payment_id)
    else:
        _push_refund(cx, payment_id)   # defined in Task 3
    return _row(cx, payment_id)
