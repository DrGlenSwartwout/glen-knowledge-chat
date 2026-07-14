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


def select_gift(cx, grant_id, sku, actor):
    """Attach a catalog gift (from data/reward-gifts.json, matched to the grant's tier) to
    a PENDING data-sharing reward grant and fulfill it. Bad grant/sku: no gift row created,
    grant left pending."""
    from dashboard import review_gifts as _rg
    row = cx.execute("SELECT email, tier FROM member_reward_grants WHERE id=? AND status='pending'",
                     (grant_id,)).fetchone()
    if not row:
        return {"ok": False, "error": "no pending grant"}
    email, tier = row[0], row[1]
    opt = next((o for o in _rg.reward_options_for_level(tier) if o["sku"] == sku), None)
    if not opt:
        return {"ok": False, "error": "sku not in level catalog"}
    _rg.add_reward_gift(cx, email, sku, opt["label"], grant_id)
    set_reward_status(cx, grant_id, "fulfilled", actor)
    return {"ok": True, "sku": sku}


def _exec_select_gift(params, ctx):
    return select_gift(ctx["cx"], params.get("grant_id"), params.get("sku"), _actor_name(ctx))


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
    register_action(Action(
        key="reward.select_gift", module="reward", title="Select reward gift",
        description="Attach a catalog gift to a pending data-sharing reward and fulfill it.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_select_gift))
