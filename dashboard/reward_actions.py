"""Operator fulfillment of data-sharing rewards on the Business-OS dispatch spine:
approve (fulfill) or dismiss a pending reward grant. OWNER/OPS, LOW_WRITE. Manual —
no automated store-credit/coupon/product; the operator hands out the gift by hand."""
import datetime
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS

_TERMINAL = ("fulfilled", "dismissed")


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _actor_name(ctx):
    a = ctx.get("actor")
    return (getattr(a, "name", "") or getattr(a, "role", "") or "console")


def init_fulfilled_column(cx):
    cols = [r[1] for r in cx.execute("PRAGMA table_info(member_reward_grants)").fetchall()]
    if "fulfilled_at" not in cols:
        cx.execute("ALTER TABLE member_reward_grants ADD COLUMN fulfilled_at TEXT")
        cx.commit()


def set_reward_status(cx, grant_id, new_status, actor):
    """Flip a PENDING grant to terminal, stamping operator + time. Idempotent +
    never-downgrade: only a currently-pending row is affected."""
    if new_status not in _TERMINAL:
        raise ValueError(f"bad reward status: {new_status}")
    init_fulfilled_column(cx)
    cur = cx.execute(
        "UPDATE member_reward_grants SET status=?, granted_by=?, fulfilled_at=? "
        "WHERE id=? AND status='pending'",
        (new_status, actor or "console", _now(), grant_id))
    cx.commit()
    return cur.rowcount > 0


def _exec_fulfill(params, ctx):
    gid = params.get("grant_id")
    if not gid:
        raise ValueError("grant_id required")
    ok = set_reward_status(ctx["cx"], gid, "fulfilled", _actor_name(ctx))
    return {"ok": ok}


def _exec_dismiss(params, ctx):
    gid = params.get("grant_id")
    if not gid:
        raise ValueError("grant_id required")
    ok = set_reward_status(ctx["cx"], gid, "dismissed", _actor_name(ctx))
    return {"ok": ok}


def register():
    if get_action("reward.fulfill"):
        return
    register_action(Action(
        key="reward.fulfill", module="reward", title="Approve data-sharing reward",
        description="Mark a pending data-sharing reward fulfilled (operator gifts manually).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_fulfill))
    register_action(Action(
        key="reward.dismiss", module="reward", title="Dismiss data-sharing reward",
        description="Dismiss a pending data-sharing reward without granting.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_dismiss))
