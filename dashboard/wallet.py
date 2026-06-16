"""Wellness Credit wallet for practitioners (Phase 2).

One credit balance per practitioner (``practitioners.wallet_balance_cents``)
with an audit log (``wallet_ledger``). Credit is EARNED from wholesale orders
(higher rate) and drop-ship dispensary sales (lower rate, credit-only), and
SPENT on the next order (up to 100%) or on a $297 certification module (capped
at 50%, monthly-gated). No-cash-value: there is no path from credit to cash.

The decision math (caps/gates/amounts) is pure and unit-tested. The Supabase
reads/writes are thin wrappers around a single row-locked transaction. DB access
goes through the ``_cursor()`` seam, which lazily imports ``db_supabase`` so this
module imports without psycopg2 and is trivially patchable in tests.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional

# ── Config (tunable; sets the wallet liability — confirm vs margin model) ──────
# Stocked wholesale orders do NOT earn credit: they already get the best per-bottle
# price (the pricing engine) plus cash margin when the practitioner dispenses. Credit
# is earned only on drop-ship sales, where we do the fulfilment — the practitioner
# earns the $20/bottle spread between the ~$50 small-order cost and the $70 retail.
EARN_RATE_ORDER = 0.0                     # wholesale orders earn nothing (dial kept)
DROPSHIP_CREDIT_PER_BOTTLE_CENTS = 2000   # $20/bottle credit on drop-ship sales
EARN_FEE_FREE_PCT = 0.03                   # 3% credit when an order is paid fee-free (Zelle/Wise)
PERSONAL_EARN_FEE_FREE_PCT = 0.035        # 3.5% credit on PERSONAL orders paid fee-free (Zelle/Wise)
MODULE_TUITION_CENTS = 29700              # $297
# Products carry high fixed cost, so credit covers at most half a product order.
# Training has minimal marginal cost, so credit can cover a whole module. Credit is
# applied training-first, then products (the caps already steer it that way).
ORDER_REDEEM_PCT = 0.50                   # credit covers <= 50% of a product order
MODULE_CREDIT_CAP_PCT = 1.00              # credit covers <= 100% of a module
MODULES_MAX = 12


# ── DB seam (lazy import keeps this module psycopg2-free at import) ────────────

def _cursor():
    from db_supabase import supabase_cursor
    return supabase_cursor()


# ── Pure decision logic ───────────────────────────────────────────────────────

def earn_amount_order_cents(order_total_cents: int) -> int:
    return math.floor(EARN_RATE_ORDER * max(0, int(order_total_cents)))


def earn_amount_dropship_cents(bottles: int) -> int:
    """$20 of credit per drop-shipped bottle (the $70 retail minus the ~$50 cost)."""
    return DROPSHIP_CREDIT_PER_BOTTLE_CENTS * max(0, int(bottles))


def redeem_amount_for_order_cents(balance_cents: int, order_total_cents: int) -> int:
    cap = math.floor(ORDER_REDEEM_PCT * max(0, int(order_total_cents)))
    return max(0, min(int(balance_cents), cap))


def redeem_amount_for_module_cents(
    balance_cents: int, tuition_cents: int = MODULE_TUITION_CENTS
) -> int:
    cap = math.floor(MODULE_CREDIT_CAP_PCT * max(0, int(tuition_cents)))
    return max(0, min(int(balance_cents), cap))


def period_key(dt) -> str:
    """Monthly gating key 'YYYY-MM'. dt is passed in for testability."""
    return dt.strftime("%Y-%m")


# ── Internal DB helpers ───────────────────────────────────────────────────────

def _already_posted(cur, qbo_invoice_id: str, entry_type: str) -> bool:
    cur.execute(
        "SELECT 1 AS exists FROM wallet_ledger "
        "WHERE qbo_invoice_id = %s AND entry_type = %s LIMIT 1",
        (qbo_invoice_id, entry_type),
    )
    return cur.fetchone() is not None


def _module_used_this_period(cur, pid: str, period: str) -> bool:
    cur.execute(
        "SELECT 1 AS exists FROM wallet_ledger "
        "WHERE practitioner_id = %s AND entry_type = 'spend_module' "
        "AND earn_period = %s LIMIT 1",
        (pid, period),
    )
    return cur.fetchone() is not None


def _locked_balance(cur, pid: str) -> int:
    cur.execute(
        "SELECT wallet_balance_cents FROM practitioners WHERE id = %s FOR UPDATE",
        (pid,),
    )
    row = cur.fetchone()
    return int(row["wallet_balance_cents"]) if row else 0


def _insert_ledger(cur, pid, entry_type, amount, balance_after,
                   qbo_invoice_id, module_slug, earn_period, note) -> None:
    cur.execute(
        "INSERT INTO wallet_ledger (practitioner_id, entry_type, amount_cents, "
        "balance_after_cents, qbo_invoice_id, module_slug, earn_period, note) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (pid, entry_type, amount, balance_after,
         qbo_invoice_id, module_slug, earn_period, note),
    )


def _apply(
    practitioner_id: str,
    entry_type: str,
    compute_delta: Callable[[int], int],
    *,
    qbo_invoice_id: Optional[str] = None,
    module_slug: Optional[str] = None,
    earn_period: Optional[str] = None,
    note: Optional[str] = None,
    precheck: Optional[Callable[[object], bool]] = None,
) -> int:
    """Row-locked read-modify-write in one transaction. Returns the signed delta
    actually applied (0 on no-op: idempotent invoice, failed precheck, or a
    zero amount)."""
    pid = str(practitioner_id)
    with _cursor() as cur:
        if qbo_invoice_id is not None and _already_posted(cur, qbo_invoice_id, entry_type):
            return 0
        if precheck is not None and not precheck(cur):
            return 0
        bal = _locked_balance(cur, pid)
        delta = compute_delta(bal)
        if delta == 0:
            return 0
        new_bal = bal + delta
        cur.execute(
            "UPDATE practitioners SET wallet_balance_cents = %s WHERE id = %s",
            (new_bal, pid),
        )
        _insert_ledger(cur, pid, entry_type, delta, new_bal,
                       qbo_invoice_id, module_slug, earn_period, note)
        return delta


# ── Public API ────────────────────────────────────────────────────────────────

def get_balance_cents(practitioner_id: str) -> int:
    with _cursor() as cur:
        cur.execute(
            "SELECT wallet_balance_cents FROM practitioners WHERE id = %s",
            (str(practitioner_id),),
        )
        row = cur.fetchone()
        return int(row["wallet_balance_cents"]) if row else 0


def set_modules_completed(practitioner_id: str, n: int) -> int:
    """Admin setter for certification progress (clamped 0..12). The source that
    drives this (cert-platform/admin) is wired in a later phase."""
    n = max(0, min(MODULES_MAX, int(n)))
    with _cursor() as cur:
        cur.execute(
            "UPDATE practitioners SET modules_completed = %s WHERE id = %s",
            (n, str(practitioner_id)),
        )
    return n


def earn_order(practitioner_id, order_total_cents, qbo_invoice_id, *, note=None) -> int:
    """Credit a confirmed wholesale order. With EARN_RATE_ORDER at 0 this is a
    no-op (returns 0, writes no row); the dial is kept for future use. Idempotent
    per invoice."""
    amt = earn_amount_order_cents(order_total_cents)
    return _apply(practitioner_id, "earn_order", lambda _bal: amt,
                  qbo_invoice_id=qbo_invoice_id, note=note)


def earn_dropship(practitioner_id, bottles, *, qbo_invoice_id=None, ref=None) -> int:
    """Credit a drop-ship dispensary sale at $20/bottle (credit-only, never cash).
    qbo_invoice_id makes it idempotent per client invoice (retry-safe)."""
    amt = earn_amount_dropship_cents(bottles)
    return _apply(practitioner_id, "earn_dropship", lambda _bal: amt,
                  qbo_invoice_id=qbo_invoice_id, note=ref)


def earn_dropship_margin(practitioner_id, margin_cents, *, qbo_invoice_id, ref=None) -> int:
    """Credit the actual margin earned on a drop-ship dispensary sale.

    margin_cents is the per-line margin (selling − base − fee) computed by
    ``practitioner_pricing.quote_line``.  Idempotent per ``qbo_invoice_id``
    (a second call with the same invoice is a silent no-op, so webhook retries
    are safe).  Credit-only — there is no path from credit to cash.

    Replaces the flat ``earn_dropship`` ($20/bottle) once Plan 3 cuts over
    the dispensary hook; the old function is kept for backward compatibility
    until that cutover.
    """
    amt = max(0, int(margin_cents))
    return _apply(practitioner_id, "earn_dropship_margin", lambda _bal: amt,
                  qbo_invoice_id=qbo_invoice_id, note=ref)


def earn_amount_fee_free_cents(order_total_cents: int) -> int:
    return math.floor(EARN_FEE_FREE_PCT * max(0, int(order_total_cents)))


def personal_earn_cents(charged_cents, method):
    """Wallet credit a cert participant earns on a PERSONAL order: 3.5% of the
    charged amount when paid fee-free (Zelle/Wise), else 0."""
    if (method or "").strip().lower() in ("zelle", "wise"):
        return math.floor(max(0, int(charged_cents or 0)) * PERSONAL_EARN_FEE_FREE_PCT)
    return 0


def earn_fee_free(practitioner_id, order_total_cents, qbo_invoice_id, *, note=None) -> int:
    """Credit 3% of a wholesale order paid by a fee-free method (Zelle/Wise).
    Idempotent per invoice."""
    amt = earn_amount_fee_free_cents(order_total_cents)
    return _apply(practitioner_id, "earn_fee_free", lambda _bal: amt,
                  qbo_invoice_id=qbo_invoice_id, note=note)


def redeem_for_order(practitioner_id, order_total_cents, qbo_invoice_id) -> int:
    """Spend credit against an order (up to 100% of the order, up to balance).
    Amount is computed against the locked balance. Returns redeemed cents."""
    delta = _apply(
        practitioner_id, "spend_order",
        lambda bal: -redeem_amount_for_order_cents(bal, order_total_cents),
        qbo_invoice_id=qbo_invoice_id,
    )
    return -delta


def redeem_for_module(practitioner_id, module_slug, *, today,
                      tuition_cents: int = MODULE_TUITION_CENTS) -> int:
    """Spend credit toward a module (<=50% of tuition, up to balance), gated to
    one module-redemption per calendar month. Returns redeemed cents (0 if
    gated or no balance)."""
    pid = str(practitioner_id)
    period = period_key(today)
    delta = _apply(
        pid, "spend_module",
        lambda bal: -redeem_amount_for_module_cents(bal, tuition_cents),
        module_slug=module_slug, earn_period=period,
        precheck=lambda cur: not _module_used_this_period(cur, pid, period),
    )
    return -delta


def get_ledger(practitioner_id, limit: int = 50) -> List[dict]:
    with _cursor() as cur:
        cur.execute(
            "SELECT entry_type, amount_cents, balance_after_cents, qbo_invoice_id, "
            "module_slug, earn_period, note, created_at FROM wallet_ledger "
            "WHERE practitioner_id = %s ORDER BY created_at DESC LIMIT %s",
            (str(practitioner_id), int(limit)),
        )
        return [dict(r) for r in cur.fetchall()]
