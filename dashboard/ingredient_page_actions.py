"""Console actions for ingredient-page review: edit / approve / regenerate.

Registered on the Business-OS dispatch spine. Mirrors dashboard/sales_pages_actions.py.
app.py injects dependencies via configure() at startup.
"""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import ingredient_pages as _ip
from dashboard import ingredient_copy as _ic

_MODEL = "claude-haiku-4-5-20251001"
_DEPS = {}  # client, send, strip, base_url -- set by app.py at startup


def configure(**kw):
    _DEPS.update(kw)


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_edit(params, ctx):
    """Update any subset of: narrative sections, scores, traditional-use, related-forms.
    Always stays draft after an edit.
    """
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    cx = ctx["cx"]

    section = (params.get("section") or "").strip()
    if section and section in _ic.NARRATIVE_SECTIONS:
        _ip.upsert_section(cx, slug, section, params.get("text") or "")

    if "research_score" in params or "traditional_score" in params:
        page = _ip.get_page(cx, slug) or {}
        research = params.get("research_score", page.get("research_score"))
        traditional = params.get("traditional_score", page.get("traditional_score"))
        _ip.set_scores(cx, slug, research, traditional)

    if "traditional_use" in params:
        _ip.set_traditional_use(cx, slug, params["traditional_use"] or [])

    if "related_forms" in params:
        _ip.set_related_forms(cx, slug, params["related_forms"] or [])

    # always stays draft after an edit
    _ip.set_state(cx, slug, "draft", by=_actor_name(ctx.get("actor")))
    return {"slug": slug, "saved": True, "state": "draft"}


def _exec_approve(params, ctx):
    """Set state approved, then notify requesters. Notify never fails the approve."""
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    cx = ctx["cx"]
    _ip.set_state(cx, slug, "approved", by=_actor_name(ctx.get("actor")))
    try:
        send = _DEPS.get("send")
        strip = _DEPS.get("strip") or (lambda s: s)
        base = _DEPS.get("base_url", "")
        page = _ip.get_page(cx, slug) or {}
        name = page.get("name") or slug
        if send is not None:
            _ip.notify_on_approve(cx, slug, name, base, send=send, strip=strip)
    except Exception as exc:  # noqa: BLE001 - notify must never fail the approve
        print(f"[ingredient-pages] notify_on_approve skipped: {exc}", flush=True)
    return {"slug": slug, "state": "approved"}


def _exec_regenerate(params, ctx):
    """Re-run propose_curation + narrative sections; stays draft for review."""
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    client = _DEPS.get("client")
    if client is None:
        raise RuntimeError("regeneration unavailable: no client configured")
    cx = ctx["cx"]
    page = _ip.get_page(cx, slug) or {}
    name = page.get("name") or slug

    from dashboard import ingredients as _ing
    info = _ing.resolve(slug)
    fmp = (info or {}).get("fmp") or {}
    studies = _ing.research_studies(name)
    ingredient = {"name": name, "fmp": fmp, "studies": studies}

    curation = _ic.propose_curation(ingredient, client)

    strip = _DEPS.get("strip") or (lambda s: s)
    sections_text = {}
    for section in _ic.NARRATIVE_SECTIONS:
        try:
            system, user = _ic.build_section_prompt(section, ingredient)
            msg = client.messages.create(
                model=_MODEL, max_tokens=800, system=system,
                messages=[{"role": "user", "content": user}])
            text = "".join(getattr(b, "text", "") for b in msg.content
                           if getattr(b, "type", "") == "text")
            sections_text[section] = strip(text).strip()
        except Exception as _sec_err:
            print(f"[ingredient-pages] regenerate section {section} failed: {_sec_err}", flush=True)

    for section, text in sections_text.items():
        if text:
            _ip.upsert_section(cx, slug, section, text, model=_MODEL)

    _ip.set_scores(cx, slug, curation.get("research_score"), curation.get("traditional_score"))
    if curation.get("traditional_use"):
        _ip.set_traditional_use(cx, slug, curation["traditional_use"])
    if curation.get("related_forms"):
        _ip.set_related_forms(cx, slug, curation["related_forms"])

    _ip.set_state(cx, slug, "draft")
    return {"slug": slug, "state": "draft", "content": sections_text, "curation": curation}


def register():
    if get_action("ingredient_page.approve"):
        return
    register_action(Action(
        key="ingredient_page.approve", module="ingredient_pages", title="Approve ingredient page",
        description="Mark an ingredient page approved; emails every requester the ready link.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(
        key="ingredient_page.edit", module="ingredient_pages", title="Edit ingredient page",
        description="Save edited narrative/scores/traditional-use/related-forms (stays draft).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="ingredient_page.regenerate", module="ingredient_pages", title="Regenerate ingredient page",
        description="Re-run AI narrative + propose_curation for review (stays draft).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_regenerate))
