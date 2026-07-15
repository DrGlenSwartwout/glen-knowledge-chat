"""Multi-payment + refund ledger per order. One row per payment or refund;
balances are always derived, never stored. Functions take a sqlite connection
for testability. QBO/Stripe sync lives in this module (Tasks 2-3) and is called
as module functions so tests can monkeypatch them."""
import sqlite3
from datetime import datetime, timezone

from dashboard import orders

_METHODS = ("Credit card (Stripe)", "eProcessing", "Check", "Cash",
            "Venmo", "PayPal", "Zelle", "Wise")


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
    return _insert(cx, order_id, kind="payment", amount_cents=amount_cents,
                   method=method, source=source, external_ref=external_ref,
                   refunds_payment_id=None, paid_at=paid_at, note=note,
                   actor=actor)
