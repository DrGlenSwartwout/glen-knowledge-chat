"""Business-OS single dispatch path. Panels and Justus both call dispatch_action.
Every dispatch resolves policy and writes an event."""
from dashboard.actions import get_action
from dashboard.rbac import policy_for, AUTO, CONFIRM, QUEUE, DENY
from dashboard import events as _events


def _amount_of(params):
    try:
        if params and params.get("amount") is not None:
            return float(params.get("amount"))
    except (TypeError, ValueError):
        return None
    return None


def _execute(cx, action, params, actor, source):
    actor_name = (actor.name or actor.role) if actor else "system"
    try:
        result = action.executor(params or {}, {"actor": actor, "cx": cx})
        eid = _events.append_event(
            cx, actor=actor_name, source=source, action_key=action.key,
            module=action.module, risk_tier=action.risk_tier, params=params,
            result=result, status="done", reversible=action.reversible)
        return {"status": "done", "result": result, "event_id": eid}
    except Exception as e:  # noqa: BLE001 - we log every failure as an event
        eid = _events.append_event(
            cx, actor=actor_name, source=source, action_key=action.key,
            module=action.module, risk_tier=action.risk_tier, params=params,
            result={"error": str(e)}, status="failed", reversible=action.reversible)
        return {"status": "failed", "error": str(e), "event_id": eid}


def dispatch_action(cx, key, params, actor, *, source="panel",
                    attended=True,  # attended: reserved for Phase 1c (Justus unattended mode)
                    confirmed=False):
    action = get_action(key)
    if action is None:
        return {"status": "error", "error": f"unknown action: {key}"}
    if actor is None or actor.role not in action.permission:
        return {"status": "denied", "reason": "permission"}

    mode = policy_for(actor.role, action.risk_tier, amount=_amount_of(params))
    if mode == DENY:
        return {"status": "denied", "reason": "policy"}
    if mode == QUEUE:
        eid = _events.append_event(
            cx, actor=actor.name or actor.role, source=source, action_key=key,
            module=action.module, risk_tier=action.risk_tier, params=params,
            result=None, status="pending_approval", reversible=action.reversible)
        return {"status": "queued", "event_id": eid}
    if mode == CONFIRM and not confirmed:
        summary = (action.confirm_summary(params) if action.confirm_summary
                   else f"Confirm: {action.title}")
        return {"status": "needs_confirmation", "summary": summary,
                "key": key, "params": params}
    return _execute(cx, action, params, actor, source)


def approve_event(cx, event_id, actor):
    ev = _events.get_event(cx, event_id)
    if not ev or ev["status"] != "pending_approval":
        return {"status": "error", "error": "not a pending approval"}
    action = get_action(ev["action_key"])
    if action is None:
        return {"status": "error", "error": "unknown action"}
    res = _execute(cx, action, ev["params"], actor, source="approval")
    _events.set_event_status(cx, event_id, "confirmed")
    return res


def cancel_event(cx, event_id):
    ev = _events.get_event(cx, event_id)
    if not ev or ev["status"] != "pending_approval":
        return {"status": "error", "error": "not a pending approval"}
    _events.set_event_status(cx, event_id, "cancelled")
    return {"status": "cancelled", "event_id": event_id}
