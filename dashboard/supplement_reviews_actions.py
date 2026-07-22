"""Console actions for the free product review feature: confirm / reject a draft.
Confirm promotes ai_draft -> confirmed (the review becomes visible on the client's
portal); reject moves it to the rejected side-state. Mirrors reviews_actions.py."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import supplement_reviews as _sr


def _name(actor):
    a = actor or {}
    return a.get("name") or a.get("email") or "console"


def _exec_confirm(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    cx = ctx["cx"]
    _sr.init_table(cx)
    res = _sr.set_status(cx, rid, "confirmed", by=_name(ctx.get("actor")))
    return {"id": rid, "status": res["status"]}


def _exec_reject(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    cx = ctx["cx"]
    _sr.init_table(cx)
    res = _sr.set_status(cx, rid, "rejected", by=_name(ctx.get("actor")))
    return {"id": rid, "status": res["status"]}


def register():
    if get_action("product_review.confirm"):
        return
    register_action(Action(key="product_review.confirm", module="product_review",
        title="Confirm product review",
        description="Publish a free product review to the client's portal.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_confirm))
    register_action(Action(key="product_review.reject", module="product_review",
        title="Reject product review",
        description="Reject a submitted product review so it never shows on the portal.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_reject))
