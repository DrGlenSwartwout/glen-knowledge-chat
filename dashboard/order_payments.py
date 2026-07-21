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


def _has_active_rows(cx, order_id):
    return cx.execute(
        "SELECT 1 FROM order_payments WHERE order_id=? AND status='active' LIMIT 1",
        (order_id,)).fetchone() is not None


def balance(cx, order_id):
    """Derived paid / refunded / balance for an order. Ledger rows are authoritative.

    Legacy fallback: a pre-ledger order can carry its payment on the ORDER row
    (orders.paid_cents, set by mark_order_paid_* before the ledger existed) with NO
    order_payments rows -- e.g. #49, marked Paid/Zelle by hand. When there are zero
    active ledger rows and the order is marked paid, surface orders.paid_cents as the
    paid amount so every balance() consumer (Edit Invoice panel, customer invoice,
    reconciliation) shows the real paid + balance-due rather than a misleading
    'balance = full invoice total'. A single active ledger row (payment OR refund)
    turns the fallback off -- the ledger then wins. `ledger_paid_cents` is ALWAYS the
    real ledger-sourced paid (never the legacy fallback), for callers (refunds) that
    must not treat legacy paid_cents as refundable."""
    o = orders.get_order(cx, order_id) or {}
    invoice = int(o.get("total_cents") or 0)
    ledger_paid = _sum(cx, order_id, "payment")
    refunded = _sum(cx, order_id, "refund")
    paid, legacy = ledger_paid, False
    if not _has_active_rows(cx, order_id):
        lp = int(o.get("paid_cents") or 0)
        if lp > 0 and o.get("pay_status") == "paid":
            paid, legacy = lp, True
    return {"invoice_cents": invoice, "paid_cents": paid,
            "refunded_cents": refunded, "ledger_paid_cents": ledger_paid,
            "legacy_fallback": legacy,
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
    """Mark a payment row synced. Creating a QBO Payment here is DISABLED by design.

    Every payment reaches QuickBooks on its own as a BANK DEPOSIT — cards via
    eProcessing/PayPal/Authorize.net, Zelle via the Bank of America feed. So a
    payment pushed from the ledger is a straight duplicate of a deposit QBO is
    already going to receive, and QBO cannot tell them apart (it happily holds
    both, inflating income). On 2026-07-19 that had produced 6 duplicate QBO
    Payments which had to be deleted by hand.

    Note the scope: QBO is retired for INVOICING, not as a system. QuickBooks is
    still the accounting system of record and still gets the money — from the bank
    feed rather than from us. So the fix is not to sever QBO, it is to stop being a
    second source of the same payment.

    Rows are marked synced with a NULL txn id (the same shape skip_qbo_push has
    always produced), which the guard below then treats as terminal, so nothing
    re-pushes later. Linking an EXISTING QBO txn via qbo_txn_id still works and is
    still the preferred move — it short-circuits on the guard before reaching here.

    _push_refund is now disabled for the same reason — see its docstring for why the
    settlement shape (separate debit vs netted into a batch deposit) does not change
    the answer.
    """
    row = _row(cx, pid)
    if row.get("qbo_txn_id") or row.get("qbo_sync") == "synced":
        return  # already synced — idempotent. Honor qbo_sync too, not just txn_id:
                # a legacy-backfill row is synced with a NULL txn_id and must NEVER
                # be pushed (its payment already exists in QBO — pushing = double-count).
    _mark_sync(cx, pid, state="synced")


def add_payment(cx, order_id, amount_cents, method, *, source="manual",
                external_ref=None, paid_at=None, note=None, actor=None,
                qbo_txn_id=None, skip_qbo_push=False):
    """Record a payment in the ledger. This NEVER creates a QBO payment any more.

    It used to push one by default. It no longer does: every payment reaches
    QuickBooks on its own as a bank deposit, so pushing made a duplicate of money
    QBO was already getting. See _push_payment for the full reasoning.

    Still worth passing:
      - qbo_txn_id=<id>: link an EXISTING QBO txn. Preferred when the id is known —
        the ledger then mirrors QBO exactly, and a later void() acts on the real txn.
      - skip_qbo_push=True: now VESTIGIAL. It was the way to opt out of the push;
        the push is gone, so this changes nothing. Accepted so existing callers keep
        working. Nothing new should pass it.

    Rows end up 'synced' either way, so nothing re-pushes later."""
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
    if qbo_txn_id or skip_qbo_push:
        _mark_sync(cx, row["id"], qbo_txn_id=qbo_txn_id, state="synced")
    _push_payment(cx, row["id"])
    return _row(cx, row["id"])


def refundable_cents(cx, order_id, refunds_payment_id=None):
    """How much can still be refunded. Against a specific payment: the lesser of
    that payment's un-refunded remainder and the order's net paid (a standalone
    refund already reduces net paid, so a payment-tied refund must not be able to
    jointly drive it negative). Otherwise (standalone): order net paid."""
    # Refunds key off REAL ledger payments only, never balance()'s legacy fallback:
    # a legacy paid_cents has no payment row to refund against (backfill it first).
    order_net_paid = max(0, _sum(cx, order_id, "payment") - _sum(cx, order_id, "refund"))
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
    """Mark a refund row synced. Creating a QBO RefundReceipt here is DISABLED, for the
    same reason as _push_payment: the bank feed already carries the money movement.

    A refund settles in one of two shapes, and BOTH make pushing wrong:
      - a separate bank DEBIT (Zelle/Wise, sometimes card): the feed downloads it, so a
        pushed RefundReceipt is a second copy;
      - NETTED into the next processor batch deposit (typical for cards): the deposit
        simply arrives smaller, so the refund is ALREADY reflected, and pushing subtracts
        it a second time.
    The only case where pushing would be right is refund money that never touches the
    connected bank account, which cannot happen for a real refund. So the channel detail
    does not change the answer.

    Timing note: QBO held ZERO RefundReceipts and ZERO CreditMemos for all of 2026 when
    this was disabled, so nothing had to be migrated or unwound — unlike payments, where
    six duplicates had already been created and had to be deleted by hand (#1037/#1039).

    Refunds are the worse direction to get wrong: an over-stated refund UNDERSTATES income.
    """
    row = _row(cx, pid)
    if row.get("qbo_txn_id") or row.get("qbo_sync") == "synced":
        return  # already synced (incl. legacy backfill: synced, NULL txn_id) — never re-push
    _mark_sync(cx, pid, state="synced")


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


def backfill_legacy_payments(cx, *, dry_run=True, skip_order_ids=None):
    """Create one source='legacy' payment row per PAID pre-ledger order so it shows
    correctly in the ledger/panel/board. Candidates: orders with paid_cents>0, source
    != 'biofield_trial' (trials skipped), status != 'cancelled' (a cancelled order
    must never get a "paid" ledger row), that have NO ledger rows yet, minus any
    skip_order_ids. Amount = the order's legacy paid_cents; method = its pay_method.

    NO QBO push — these payments already exist in QBO from the pre-ledger flow, so we
    insert directly (never via add_payment/record_payment); qbo_sync='synced' and
    qbo_txn_id stays NULL, so a later void() won't touch QBO. Idempotent: an order
    that already has ANY ledger row is excluded (re-runs write nothing, and orders
    under active reconciliation aren't disturbed). Provenance-neutral timestamps: the
    row's paid_at/created_at come from the order's own paid_at (never now()).

    dry_run=True returns the plan without writing. Returns
    {"candidates": [...], "count": N, "written": M}."""
    skip = {int(x) for x in (skip_order_ids or [])}
    rows = cx.execute(
        "SELECT o.id, o.email, o.name, COALESCE(o.paid_cents,0) AS paid, "
        "COALESCE(o.pay_method,'') AS method, o.paid_at, o.created_at, "
        "COALESCE(o.total_cents,0) AS total "
        "FROM orders o "
        "WHERE COALESCE(o.paid_cents,0) > 0 "
        "AND COALESCE(o.source,'') != 'biofield_trial' "
        "AND COALESCE(o.status,'') != 'cancelled' "
        "AND o.id NOT IN (SELECT DISTINCT order_id FROM order_payments) "
        "ORDER BY o.id").fetchall()
    plan = []
    for r in rows:
        if int(r["id"]) in skip:
            continue
        plan.append({
            "order_id": int(r["id"]), "email": r["email"] or "", "name": r["name"] or "",
            "amount_cents": int(r["paid"]), "method": (r["method"] or "legacy"),
            "total_cents": int(r["total"]), "paid_at": r["paid_at"] or r["created_at"],
        })
    written = 0
    if not dry_run:
        for p in plan:
            ts = p["paid_at"]
            cx.execute(
                "INSERT INTO order_payments (order_id, kind, amount_cents, method, source, "
                "status, qbo_sync, note, paid_at, created_at, updated_at, created_by) "
                "VALUES (?, 'payment', ?, ?, 'legacy', 'active', 'synced', "
                "'legacy pay_status backfill', ?, ?, ?, 'backfill')",
                (p["order_id"], p["amount_cents"], p["method"], ts, ts, ts))
            written += 1
        cx.commit()
    return {"candidates": plan, "count": len(plan), "written": written}
