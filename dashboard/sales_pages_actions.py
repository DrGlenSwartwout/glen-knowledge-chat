"""Phase-5 console actions for sales-page copy review: approve / edit / regenerate.
Registered on the Business-OS dispatch spine. The regenerate executor needs the
Anthropic client + product lookups, which app.py injects via configure()."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import sales_pages as _sp
from dashboard import sales_copy as _sc

_MODEL = "claude-haiku-4-5-20251001"
_DEPS = {}  # client, get_product, product_card, strip_dash — set by app.py at startup


def configure(**kw):
    _DEPS.update(kw)


def regenerate_copy(slug):
    """Generate all narrative sections synchronously; return {section: text} or None."""
    client = _DEPS.get("client")
    get_product = _DEPS.get("get_product")
    product_card = _DEPS.get("product_card")
    strip = _DEPS.get("strip_dash") or (lambda s: s)
    if client is None or get_product is None:
        return None
    p = get_product(slug)
    if not p:
        return None
    prod = dict(p)
    if not prod.get("ingredients") and product_card is not None:
        prod["ingredients"] = (product_card(p) or {}).get("ingredients", [])
    out = {}
    for section in _sc.NARRATIVE_SECTIONS:
        system, user = _sc.build_section_prompt(section, prod)
        msg = client.messages.create(
            model=_MODEL, max_tokens=600, system=system,
            messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in msg.content
                       if getattr(b, "type", "") == "text")
        out[section] = strip(text).strip()
    return out


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_approve(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    _sp.set_state(ctx["cx"], slug, "approved", by=_actor_name(ctx.get("actor")))
    try:
        from dashboard import sales_page_viewers as _spv
        from dashboard import inbox as _inbox
        get_product = _DEPS.get("get_product")
        strip = _DEPS.get("strip_dash") or (lambda s: s)
        pname = ((get_product(slug) or {}).get("name", slug) if get_product else slug)
        base = _DEPS.get("base_url", "")
        _spv.notify_on_approve(ctx["cx"], slug, pname, base, send=_inbox.send_email, strip=strip)
    except Exception as e:  # noqa: BLE001 - notifying viewers must never fail the approve
        print(f"[sales-pages] viewer notify skipped: {e}", flush=True)
    return {"slug": slug, "state": "approved"}


def _exec_edit(params, ctx):
    slug = (params.get("slug") or "").strip()
    section = (params.get("section") or "").strip()
    if not slug or section not in _sc.NARRATIVE_SECTIONS:
        raise ValueError("slug and valid section required")
    cx = ctx["cx"]
    _sp.upsert_section(cx, slug, section, params.get("text") or "")
    # any edit returns the page to draft; edited copy must be re-approved (never auto-approves)
    _sp.set_state(cx, slug, "draft", by=_actor_name(ctx.get("actor")))
    return {"slug": slug, "section": section, "saved": True}


def _exec_regenerate(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    content = regenerate_copy(slug)
    if content is None:
        raise RuntimeError("regeneration unavailable")
    cx = ctx["cx"]
    for section, text in content.items():
        if text:
            _sp.upsert_section(cx, slug, section, text, model=_MODEL)
    _sp.set_state(cx, slug, "draft")
    return {"slug": slug, "state": "draft", "content": content}


def register():
    if get_action("sales_pages.approve"):
        return
    register_action(Action(
        key="sales_pages.approve", module="sales_pages", title="Approve sales page",
        description="Mark a product's AI sales copy approved (drops the draft banner on the live page).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(
        key="sales_pages.edit", module="sales_pages", title="Edit sales-page section",
        description="Save edited copy for one narrative section (stays draft).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="sales_pages.regenerate", module="sales_pages", title="Regenerate sales copy",
        description="Regenerate all narrative sections in-process for review (stays draft).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_regenerate))
