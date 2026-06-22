"""Console actions to manage the email suppression list. Fed by the local bounce
scanner (email_suppression.add, source=bounce-scan) and readable for the report.
Registered on the Business-OS dispatch spine (POST /api/action/<key>)."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import email_suppression as _es


def _exec_add(params, ctx):
    cx = ctx["cx"]
    _es.init_table(cx)
    src = (params.get("source") or "console").strip()
    n = 0
    for e in (params.get("entries") or []):
        email = (e.get("email") or "").strip()
        if not email:
            continue
        _es.add(cx, email, (e.get("bounce_type") or "hard"),
                (e.get("reason") or ""), src)
        n += 1
    return {"added": n}


def _exec_list(params, ctx):
    cx = ctx["cx"]
    _es.init_table(cx)
    return {"rows": _es.list_recent(cx, int(params.get("limit") or 200))}


def register():
    if get_action("email_suppression.add"):
        return
    register_action(Action(
        key="email_suppression.add", module="email_suppression",
        title="Add email suppressions",
        description="Suppress hard-bounced addresses (the app stops emailing them).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_add))
    register_action(Action(
        key="email_suppression.list", module="email_suppression",
        title="List email suppressions",
        description="List suppressed addresses.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_list))
