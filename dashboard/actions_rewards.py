"""Rewards / affiliate payout action.

Registered as a MONEY_SEND action so owner approval is required before
any payout is processed. Cash mode: marks all pending earnings as paid
(actual money transfer is handled via the owner's existing finance tools).
Points mode: refused — points are store-credit/gift only and not cashable.
"""
from datetime import datetime, timezone

from dashboard.actions import action, MONEY_SEND
from dashboard.rbac import OWNER, OPS, VA


@action(
    key="rewards.process_payout",
    module="money",
    title="Process affiliate cash-out",
    description="Approve and record an affiliate cash-out (cash: mark paid; points are not cashable).",
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
        # Points are store-credit / gift-power only and are NOT cashable.
        # Cash flows exclusively through the pro-influencer affiliate_earnings rail.
        raise ValueError(
            "points are store-credit/gift only and not cashable; "
            "only pro-affiliate (tier:pro-influencer) cash earnings are payable"
        )
