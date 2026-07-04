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
