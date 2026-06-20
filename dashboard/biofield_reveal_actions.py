"""Begin #4a console actions: edit / approve a Biofield reveal. On approve,
mint a magic-link token, stamp it, write an auth_tokens row, email the owner.
Registered on the Business-OS dispatch spine. app.py injects deps via configure()."""
from datetime import datetime, timezone, timedelta

from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import biofield_reveals as _br

_DEPS = {}  # base_url, send, hash_token, mint_token - set by app.py


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
    top = dict(cur["top"])
    if "name" in params:
        top["name"] = (params.get("name") or "").strip()
    if "meaning" in params:
        top["meaning"] = (params.get("meaning") or "").strip()
    if "slug" in params:
        top["slug"] = (params.get("slug") or "").strip()
    _br.set_top(ctx["cx"], rid, top)
    return {"ok": True}


def _exec_approve(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    row = _br.get(ctx["cx"], rid)
    if not row:
        raise ValueError("not found")
    mint = _DEPS.get("mint_token") or (lambda: "tok")
    hash_token = _DEPS.get("hash_token") or (lambda t: t)
    token = mint()
    th = hash_token(token)
    ok = _br.approve(ctx["cx"], rid, _actor_name(ctx.get("actor")), th)
    if not ok:
        return {"ok": False, "note": "already approved"}
    now = datetime.now(timezone.utc)
    exp = (now + timedelta(days=30)).isoformat()
    ctx["cx"].execute(
        "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
        (th, row["email"], "biofield_reveal", now.isoformat(), exp))
    ctx["cx"].commit()
    # Best-effort notify; approval must never fail if the email fails.
    try:
        send = _DEPS.get("send")
        base = _DEPS.get("base_url", "")
        if send:
            url = f"{base}/begin/biofield/{token}"
            body = ("Aloha,\n\nYour Biofield Analysis is ready. View your top remedy match here:\n"
                    f"{url}\n\nIn wellness,\nDr. Glen and Rae\n")
            send(row["email"], "Your Biofield Analysis is ready", body)
    except Exception as e:  # noqa: BLE001 - notify must never fail the approve
        print(f"[biofield-reveal-approve] notify failed: {e!r}", flush=True)
    return {"ok": True}


def register():
    if get_action("biofield_reveal.approve"):
        return
    register_action(Action(
        key="biofield_reveal.edit", module="biofield_reveal", title="Edit Biofield reveal",
        description="Edit the top-match name/meaning (stays draft).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="biofield_reveal.approve", module="biofield_reveal", title="Approve Biofield reveal",
        description="Approve the top reveal, mint the magic link, and email the owner.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
