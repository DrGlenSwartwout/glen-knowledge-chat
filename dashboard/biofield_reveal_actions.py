"""Begin #4a console actions: edit interpretation/remedies; approve = un-blur the
top remedy (first_approved=1). The ready email already went out at ingest, so
approve sends nothing. Registered on the Business-OS dispatch spine."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import biofield_reveals as _br

_DEPS = {}


def configure(**kw):
    _DEPS.update(kw)


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_edit(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    cur = _br.get(ctx["cx"], rid)
    if not cur:
        raise ValueError("not found")
    interp = dict(cur["interpretation"])
    if "greeting" in params:
        interp["greeting"] = (params.get("greeting") or "").strip()
    if "body" in params:
        interp["body"] = (params.get("body") or "").strip()
    _br.set_interpretation(ctx["cx"], rid, interp)
    if isinstance(params.get("remedies"), list):
        _br.set_remedies(ctx["cx"], rid, params["remedies"])
    return {"ok": True}


def _exec_approve(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    ok = _br.approve_first(ctx["cx"], rid, _actor_name(ctx.get("actor")))
    return {"ok": bool(ok)}


def register():
    if get_action("biofield_reveal.approve"):
        return
    register_action(Action(
        key="biofield_reveal.edit", module="biofield_reveal", title="Edit Biofield reveal",
        description="Edit the interpretation and/or ranked remedies (stays pending).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="biofield_reveal.approve", module="biofield_reveal", title="Approve top remedy",
        description="Un-blur the top remedy for the visitor (the rest unlock via the $1 trial).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
