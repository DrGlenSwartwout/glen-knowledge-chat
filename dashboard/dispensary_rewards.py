"""L2-only settlement for dispensary sales.

A dispensary/drop-ship sale is paid at wholesale by the practitioner (their pay is the
markup), so it pays NO L1 — only the L2 override (points) to whoever referred the
PRACTITIONER into the system. Fires on every paid dispensary order (first + reorders),
across all pay methods, called from both the card path (app._settle_order_points) and
the alt-pay manual-confirm path (dashboard.orders._record_payment_exec).

Config is read from env, mirroring app.py's parsing exactly, so the two callers behave
identically without importing app.
"""
import os


def _referrals_on():
    return os.environ.get("REFERRALS", "").strip().lower() in ("1", "true", "yes")


def _tier2_on():
    return os.environ.get("REFERRAL_TIER2_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _reward_pct():
    try:
        return max(0, int(os.environ.get("REFERRER_REWARD_PCT", "0")))
    except (TypeError, ValueError):
        return 0


def settle_dispensary_l2(cx, order, order_ref):
    """Credit L2 points to the practitioner's upline on a paid dispensary order.
    Resolves the practitioner from order['practitioner_id'] (stamped at checkout), NOT
    from the patient's referral row. Idempotent per order_ref. Returns cents credited.
    Never raises into the caller."""
    try:
        if not _referrals_on() or not _tier2_on():
            return 0
        pct = _reward_pct()
        if pct <= 0:
            return 0
        pid = (str(order.get("practitioner_id") or "")).strip()
        if not pid:
            return 0
        from dashboard.practitioner_portal import practitioner_email_by_id
        from dashboard import referrals as _rf, points as _points
        practitioner = (practitioner_email_by_id(pid) or "").strip().lower()
        if not practitioner:
            return 0
        l2 = (_rf.owner_of_referee(cx, practitioner) or "").strip().lower()
        patient = (str(order.get("email") or "")).strip().lower()
        if not l2 or l2 == practitioner or l2 == patient:
            return 0
        product_cents = max(0, int(order.get("total_cents") or 0)
                            - int(order.get("shipping_cents") or 0)
                            - int(order.get("get_cents") or 0))
        reward_l2 = product_cents * pct // 200
        if reward_l2 <= 0:
            return 0
        _points.init_points_table(cx)
        key = f"disp_l2:{order_ref}"
        if _points.has_entry(cx, order_ref=key, reason="referral_reward_l2"):
            return 0
        _points.credit(cx, l2, value_cents=reward_l2, reason="referral_reward_l2", order_ref=key)
        return reward_l2
    except Exception as _e:
        print(f"[dispensary-l2] settle skipped ref={order_ref!r}: {_e!r}", flush=True)
        return 0


def settle_dispensary_margin(order, order_ref):
    """Credit the practitioner's drop-ship Wellness Credit (wallet margin) and record the
    dispensary sale for a paid dispensary order. Reads practitioner_id + margin_cents off
    the order (both stamped at checkout). Idempotent per invoice — both
    wallet.earn_dropship_margin (keyed by qbo_invoice_id) and record_dispensary_order
    (ON CONFLICT invoice_id DO NOTHING) no-op on repeat. Best-effort; never raises.

    Used by the ALT-PAY settlement path (_record_payment_exec); the CARD path credits the
    same margin inline in begin_checkout_return's kind=='client' block. An order is only
    ever one pay method, and the earn is invoice-idempotent, so the two paths never
    double-credit. Returns margin cents credited (0 if none)."""
    try:
        pid = (str(order.get("practitioner_id") or "")).strip()
        inv = str(order_ref or "")
        if not pid or not inv:
            return 0
        margin = max(0, int(order.get("margin_cents") or 0))
        from dashboard import wallet as _wallet, practitioner_portal as _pp
        _wallet.earn_dropship_margin(pid, margin, qbo_invoice_id=inv)
        if hasattr(_pp, "record_dispensary_order"):
            _pp.record_dispensary_order(pid, invoice_id=inv, credit_earned_cents=margin)
        return margin
    except Exception as _e:
        print(f"[dispensary-margin] settle skipped ref={order_ref!r}: {_e!r}", flush=True)
        return 0
