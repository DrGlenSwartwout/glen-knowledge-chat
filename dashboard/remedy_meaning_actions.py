"""Console actions for the canonical remedy meanings store: save/delete one slug,
AI-propose one slug, and propose-all-missing. Registered on the dispatch spine."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import biofield_meanings as _bm

_DEPS = {}


def configure(**kw):
    _DEPS.update(kw)


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_save(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    _bm.init_table(ctx["cx"])
    _bm.upsert(ctx["cx"], slug, (params.get("meaning") or "").strip(),
               _actor_name(ctx.get("actor")), "glen")
    return {"ok": True}


def _exec_delete(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    _bm.init_table(ctx["cx"])
    _bm.delete(ctx["cx"], slug)
    return {"ok": True}


def _exec_propose(params, ctx):
    slug = (params.get("slug") or "").strip()
    prod = (_DEPS.get("products") or {}).get(slug)
    if not prod:
        raise ValueError("unknown product")
    _bm.init_table(ctx["cx"])
    text = _bm.propose_meaning(dict(prod, slug=slug), _DEPS.get("client"))
    if text:
        _bm.upsert(ctx["cx"], slug, text, "ai", "ai")
    return {"ok": bool(text), "meaning": text}


def _exec_propose_all(params, ctx):
    products = _DEPS.get("products") or {}
    _bm.init_table(ctx["cx"])
    existing = _bm.get_map(ctx["cx"])
    cap = int(params.get("cap") or 200)
    proposed, failed = 0, 0
    for slug, prod in list(products.items()):
        if proposed + failed >= cap:
            break
        if existing.get(slug):
            continue
        text = _bm.propose_meaning(dict(prod, slug=slug), _DEPS.get("client"))
        if text:
            _bm.upsert(ctx["cx"], slug, text, "ai", "ai")
            proposed += 1
        else:
            failed += 1
    print(f"[remedy-meaning] propose_all proposed={proposed} failed={failed}", flush=True)
    return {"ok": True, "proposed": proposed, "failed": failed}


def register():
    if get_action("remedy_meaning.save"):
        return
    register_action(Action(key="remedy_meaning.save", module="remedy_meaning", title="Save remedy meaning",
        description="Set the canonical meaning for a product.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS), executor=_exec_save))
    register_action(Action(key="remedy_meaning.delete", module="remedy_meaning", title="Delete remedy meaning",
        description="Remove the canonical meaning for a product.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS), executor=_exec_delete))
    register_action(Action(key="remedy_meaning.propose", module="remedy_meaning", title="Propose remedy meaning (AI)",
        description="AI-propose a function-covering meaning for one product.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS), executor=_exec_propose))
    register_action(Action(key="remedy_meaning.propose_all", module="remedy_meaning", title="Propose all missing (AI)",
        description="AI-propose meanings for every product without one.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS), executor=_exec_propose_all))
