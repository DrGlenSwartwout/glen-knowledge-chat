"""Rewards / affiliate payout action.

Registered as a MONEY_SEND action so owner approval is required before
any payout is processed. Cash mode: marks all pending earnings as paid
(actual money transfer is handled via the owner's existing finance tools).
Points mode: redeems the full points balance at cash_out_face_pct value
and records the conversion.
"""
from datetime import datetime, timezone

from dashboard.actions import action, MONEY_SEND
from dashboard.rbac import OWNER, OPS, VA


@action(
    key="rewards.process_payout",
    module="money",
    title="Process affiliate cash-out",
    description="Approve and record an affiliate cash-out (cash: mark paid; points: redeem at face value).",
    risk_tier=MONEY_SEND,
    permission=(OWNER, OPS, VA),
)
def process_payout(params, ctx):
    cx = (ctx or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection provided")

    slug = str(params.get("slug") or "").strip()
    mode = str(params.get("mode") or "cash").strip()
    if not slug:
        raise ValueError("slug is required")

    from dashboard import rewards as _rewards
    from dashboard import points as _points

    settings = _rewards.load_settings({})
    face_pct = float(settings["cash_out_face_pct"])
    now = datetime.now(timezone.utc).isoformat()

    if mode == "cash":
        total = _rewards.pending_cash_total(cx, slug)
        _rewards.mark_paid(cx, slug)
        return {
            "slug": slug,
            "mode": "cash",
            "amount_cents": total,
            "status": "paid",
            "paid_at": now,
        }
    else:
        # Points mode: redeem the full balance at the face-value rate
        referrer_email = _rewards.referrer_email_for_slug(cx, slug)
        if not referrer_email:
            raise ValueError(f"no approved affiliate found for slug={slug!r}")
        bal = _points.balance(cx, referrer_email)
        if bal <= 0:
            return {"slug": slug, "mode": "points", "points_redeemed": 0, "cash_value_cents": 0}
        cash_value = round(bal * face_pct)
        order_ref = f"cashout:{slug}:{int(datetime.now(timezone.utc).timestamp())}"
        _points.redeem(cx, referrer_email, value_cents=bal, order_ref=order_ref)
        return {
            "slug": slug,
            "mode": "points",
            "points_redeemed": bal,
            "cash_value_cents": cash_value,
            "order_ref": order_ref,
        }
