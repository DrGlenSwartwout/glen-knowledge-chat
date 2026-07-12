# dashboard/ship_credit.py
"""Shipping-overpayment credit: an email-keyed, auto-applying store credit.

When a combined-shipment recalc finds an already-paid member overpaid on shipping
(recorded per-order as dashboard.orders.overpay_credit_cents), that overpayment is
GRANTED here as a spendable balance in the points_ledger 'ship_credit' scope —
isolated from 'rm' loyalty points so it is NOT subject to their price floor and
applies in full.

At the customer's next checkout the outstanding balance auto-applies as its own
credit line (see _apply_ship_credit in app.py), reducing what is charged; or an
operator refunds it one-click (finance.refund_ship_credit). All of that is gated by
SHIP_CREDIT_AUTOAPPLY_ENABLED at the call sites — this module is the ledger, pure of
the flag. Email is normalized to lowercase on every side so grant (from the order's
email) and apply (from the checkout email) hit the same balance.
"""
from dashboard import points as _points

SCOPE = "ship_credit"
GRANT_REASON = "ship_overpay"        # +credit granted at recalc, keyed to source order
APPLY_REASON = "ship_credit_applied"  # -debit when applied to a new order
REFUND_REASON = "ship_credit_refunded"  # -debit when refunded instead of applied


def _norm(email):
    return (email or "").strip().lower()


def grant(cx, email, cents, *, source_ref):
    """Idempotent grant of a shipping-overpayment credit, keyed to the SOURCE order
    (the paid order that overpaid). Re-running recalc never double-grants."""
    email = _norm(email)
    if not email or int(cents or 0) <= 0 or not source_ref:
        return
    _points.credit(cx, email, value_cents=int(cents), reason=GRANT_REASON,
                   order_ref=str(source_ref), scope=SCOPE)


def balance(cx, email):
    """Spendable shipping credit for this customer, in cents."""
    email = _norm(email)
    if not email:
        return 0
    return _points.balance(cx, email, scope=SCOPE)


def plan_application(balance_cents, chargeable_cents):
    """Pure: how much credit to apply to an order — bounded by the balance AND by the
    order's own chargeable total, so a credit can never make a total negative. The
    remainder stays as balance for the next order. Flag-agnostic."""
    return max(0, min(int(balance_cents or 0), int(chargeable_cents or 0)))


def consume(cx, email, cents, *, applied_ref):
    """Guarded debit when credit is applied to a NEW order. Idempotent on the applying
    order_ref, so re-pricing / resubmitting the same order never double-spends, and
    clamped to the live balance. Returns cents actually consumed."""
    email = _norm(email)
    if not email or int(cents or 0) <= 0 or not applied_ref:
        return 0
    return _points.spend(cx, email, value_cents=int(cents), reason=APPLY_REASON,
                         order_ref=str(applied_ref), scope=SCOPE)


def already_refunded(cx, *, source_ref):
    """True if this source order's shipping credit was already refunded (double-refund
    guard for the one-click refund action)."""
    if not source_ref:
        return False
    return _points.has_entry(cx, order_ref=str(source_ref), reason=REFUND_REASON, scope=SCOPE)


def mark_refunded(cx, email, cents, *, source_ref):
    """Guarded debit removing a credit that was REFUNDED to the customer instead of
    applied, so it can't also auto-apply. Idempotent on the source order (this row is
    the already-refunded marker). Returns cents actually removed."""
    email = _norm(email)
    if not email or int(cents or 0) <= 0 or not source_ref:
        return 0
    return _points.spend(cx, email, value_cents=int(cents), reason=REFUND_REASON,
                         order_ref=str(source_ref), scope=SCOPE)
