"""Phase 2a-1 console actions for product-review moderation: approve / reject / feature."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import product_reviews as _pr


def _name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_approve(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    _pr.set_status(ctx["cx"], rid, "approved", by=_name(ctx.get("actor")))
    return {"id": rid, "status": "approved"}


def _exec_reject(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    _pr.set_status(ctx["cx"], rid, "rejected", by=_name(ctx.get("actor")))
    return {"id": rid, "status": "rejected"}


def _exec_feature(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    _pr.set_featured(ctx["cx"], rid, bool(params.get("on")))
    return {"id": rid, "featured": bool(params.get("on"))}


def _exec_gift_approve(params, ctx):
    from dashboard import review_gifts as _rg
    rid = int(params.get("review_id") or 0)
    if not rid:
        raise ValueError("review_id required")
    g = _rg.get_for_review(ctx["cx"], rid)
    if not g:
        raise ValueError("no gift for review")
    sku = (params.get("sku") or "").strip()
    if sku and _rg.valid_sku(sku):
        _rg.swap_sku(ctx["cx"], g["id"], sku, _rg.catalog_by_sku().get(sku, {}).get("label", sku))
    _rg.set_status(ctx["cx"], g["id"], "approved", by=_name(ctx.get("actor")))
    return {"review_id": rid, "status": "approved"}


def _exec_gift_reject(params, ctx):
    from dashboard import review_gifts as _rg
    rid = int(params.get("review_id") or 0)
    if not rid:
        raise ValueError("review_id required")
    g = _rg.get_for_review(ctx["cx"], rid)
    if g:
        _rg.set_status(ctx["cx"], g["id"], "rejected", by=_name(ctx.get("actor")))
    return {"review_id": rid, "status": "rejected"}


def register():
    if get_action("reviews.approve"):
        return
    register_action(Action(key="reviews.approve", module="reviews", title="Approve review",
        description="Publish a product review on its sales page.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(key="reviews.reject", module="reviews", title="Reject review",
        description="Hide a product review from the sales page.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_reject))
    register_action(Action(key="reviews.feature", module="reviews", title="Feature review",
        description="Pin/unpin a product review at the top of its section.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_feature))
    register_action(Action(key="reviews.gift_approve", module="reviews", title="Approve review gift",
        description="Approve the AI-suggested gift (optionally swap the item) for a 5-point review.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_gift_approve))
    register_action(Action(key="reviews.gift_reject", module="reviews", title="Reject review gift",
        description="Reject the AI-suggested gift for a review.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_gift_reject))
