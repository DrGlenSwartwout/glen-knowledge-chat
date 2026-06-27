"""Phase 4 console actions: approve & send / dismiss a suggested testimonial invite."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import testimonial_invites as _ti


def _name(actor):
    return getattr(actor, "name", "") or getattr(actor, "role", "") or "console"


def _exec_approve(params, ctx):
    cid = int(params.get("id") or 0)
    if not cid:
        raise ValueError("id required")
    cx = ctx["cx"]
    cand = _ti.get(cx, cid)
    if not cand:
        raise ValueError("candidate not found")
    if _ti.send_invite_email(cand["email"], cand.get("name"), quote=cand.get("quote")):
        _ti.set_status(cx, cid, "sent", by=_name(ctx.get("actor")), sent=True)
        return {"id": cid, "status": "sent"}
    # send failed -> leave 'approved' so it can be retried, never lose the candidate
    _ti.set_status(cx, cid, "approved", by=_name(ctx.get("actor")))
    return {"id": cid, "status": "approved", "sent": False}


def _exec_dismiss(params, ctx):
    cid = int(params.get("id") or 0)
    if not cid:
        raise ValueError("id required")
    _ti.set_status(ctx["cx"], cid, "dismissed", by=_name(ctx.get("actor")))
    return {"id": cid, "status": "dismissed"}


def register():
    if get_action("testimonial_invite.approve"):
        return
    register_action(Action(
        key="testimonial_invite.approve", module="testimonial_invite",
        title="Approve & send testimonial invite",
        description="Email this client a testimonial invite (-> /results) and mark the candidate sent.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(
        key="testimonial_invite.dismiss", module="testimonial_invite",
        title="Dismiss testimonial invite",
        description="Dismiss a suggested testimonial invite (no email sent).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_dismiss))
