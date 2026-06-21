"""Studio-credit console actions on the Business-OS dispatch spine: log a claim,
approve (grants the 30-day comp + magic-link email via the injected grant_fn),
reject. OWNER/OPS, LOW_WRITE."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import studio_credit as _sc

_DEPS = {"grant_fn": None}


def configure(grant_fn=None):
    if grant_fn is not None:
        _DEPS["grant_fn"] = grant_fn


def _actor_name(ctx):
    a = ctx.get("actor")
    return (getattr(a, "name", "") or getattr(a, "role", "") or "console")


def _exec_add(params, ctx):
    email = (params.get("email") or "").strip()
    if not email:
        raise ValueError("email required")
    claim = _sc.add_claim(
        ctx["cx"], email=email,
        invoice_ref=(params.get("invoice_ref") or "").strip(),
        proof_note=(params.get("proof_note") or "").strip(),
        source="console", created_by=_actor_name(ctx))
    return {"ok": True, "id": claim["id"], "status": claim["status"]}


def _exec_approve(params, ctx):
    cid = (params.get("id") or "").strip()
    if not cid:
        raise ValueError("id required")
    grant_fn = _DEPS["grant_fn"]
    if grant_fn is None:
        raise RuntimeError("studio_credit_actions not configured with grant_fn")
    return _sc.approve_claim(
        ctx["cx"], cid, decided_by=_actor_name(ctx),
        grant_fn=grant_fn, force=bool(params.get("force")))


def _exec_reject(params, ctx):
    cid = (params.get("id") or "").strip()
    if not cid:
        raise ValueError("id required")
    return _sc.reject_claim(
        ctx["cx"], cid, decided_by=_actor_name(ctx),
        reason=(params.get("reason") or "").strip())


def register():
    if get_action("studio_credit.add"):
        return
    register_action(Action(
        key="studio_credit.add", module="studio_credit", title="Log studio-credit claim",
        description="Record a studio-app purchaser's claim for a free month (stays pending).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_add))
    register_action(Action(
        key="studio_credit.approve", module="studio_credit", title="Approve studio credit",
        description="Grant the 30-day membership comp + send the magic link (one per email per year).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(
        key="studio_credit.reject", module="studio_credit", title="Reject studio-credit claim",
        description="Reject the claim with a reason; grants nothing.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_reject))
