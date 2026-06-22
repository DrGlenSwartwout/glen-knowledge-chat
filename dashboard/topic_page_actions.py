"""Console actions for topic-page review: edit / approve / regenerate.

approve is COMPLIANCE-GATED: it refuses to publish unless the stored compliance
scan passed. Mirrors dashboard/ingredient_page_actions.py otherwise.
"""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import topic_pages as _tp
from dashboard import topic_copy as _tc

_MODEL = "claude-haiku-4-5-20251001"
_DEPS = {}  # client, send, strip, base_url, ingredient_slugs, product_slugs, topic_slugs


def configure(**kw):
    _DEPS.update(kw)


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_edit(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    cx = ctx["cx"]
    section = (params.get("section") or "").strip()
    if section and section in _tc.NARRATIVE_SECTIONS:
        _tp.upsert_section(cx, slug, section, params.get("text") or "")
    if "name" in params:
        _tp.set_name(cx, slug, params.get("name") or "")
    if "kind" in params:
        _tp.set_kind(cx, slug, params.get("kind") or "")
    # editing invalidates any prior compliance verdict
    _tp.set_compliance(cx, slug, {})
    _tp.set_state(cx, slug, "draft", by=_actor_name(ctx.get("actor")))
    return {"slug": slug, "ok": True, "state": "draft"}


def _exec_approve(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    cx = ctx["cx"]
    page = _tp.get_page(cx, slug) or {}
    comp = page.get("compliance") or {}
    if not comp.get("passed"):
        return {"slug": slug, "ok": False, "error": "compliance_failed",
                "flags": comp.get("flags") or []}
    _tp.set_state(cx, slug, "approved", by=_actor_name(ctx.get("actor")))
    try:
        send = _DEPS.get("send")
        strip = _DEPS.get("strip") or (lambda s: s)
        base = _DEPS.get("base_url", "")
        name = page.get("name") or slug
        if send is not None:
            _tp.notify_on_approve(cx, slug, name, base, send=send, strip=strip)
    except Exception as exc:  # noqa: BLE001 - notify must never fail the approve
        print(f"[topic-pages] notify skipped: {exc}", flush=True)
    return {"slug": slug, "ok": True, "state": "approved"}


def _exec_regenerate(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    client = _DEPS.get("client")
    if client is None:
        raise RuntimeError("regeneration unavailable: no client configured")
    cx = ctx["cx"]
    page = _tp.get_page(cx, slug) or {}
    name = page.get("name") or slug
    kind = page.get("kind") or "symptom"
    topic = {"name": name, "kind": kind}
    strip = _DEPS.get("strip") or (lambda s: s)

    content = {}
    for section in _tc.NARRATIVE_SECTIONS:
        try:
            system, user = _tc.build_section_prompt(section, topic)
            msg = client.messages.create(model=_MODEL, max_tokens=600, system=system,
                                         messages=[{"role": "user", "content": user}])
            text = "".join(getattr(b, "text", "") for b in msg.content
                           if getattr(b, "type", "") == "text")
            content[section] = strip(text).strip()
        except Exception as _e:
            print(f"[topic-pages] regenerate section {section} failed: {_e}", flush=True)
    for section, text in content.items():
        if text:
            _tp.upsert_section(cx, slug, section, text, model=_MODEL)

    curation = _tc.propose_curation(topic, client)
    links = _tc.validate_links(
        curation.get("links") or {},
        ingredient_slugs=_DEPS.get("ingredient_slugs") or {},
        product_slugs=_DEPS.get("product_slugs") or {},
        topic_slugs=_DEPS.get("topic_slugs") or {},
    )
    _tp.set_links(cx, slug, links)
    _tp.set_seo(cx, slug, {"title": curation.get("title") or name,
                           "meta_description": curation.get("meta_description") or ""})

    full = _tp.get_page(cx, slug) or {}
    result = _tc.compliance_scan(full.get("content") or {}, client)
    _tp.set_compliance(cx, slug, result)
    _tp.set_state(cx, slug, "draft" if result.get("passed") else "gated")
    return {"slug": slug, "state": "draft" if result.get("passed") else "gated",
            "compliance": result}


def _exec_dismiss(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    _tp.set_state(ctx["cx"], slug, "dismissed", by=_actor_name(ctx.get("actor")))
    return {"slug": slug, "ok": True, "state": "dismissed"}


def register():
    if get_action("topic_page.approve"):
        return
    register_action(Action(
        key="topic_page.approve", module="topic_pages", title="Approve topic page",
        description="Publish a topic page (refused unless the compliance scan passed); emails requesters.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(
        key="topic_page.edit", module="topic_pages", title="Edit topic page",
        description="Edit a section/name/kind (resets to draft and clears the compliance verdict).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="topic_page.regenerate", module="topic_pages", title="Regenerate topic page",
        description="Re-draft sections, re-validate links, and re-run the compliance scan.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_regenerate))
    register_action(Action(
        key="topic_page.dismiss", module="topic_pages", title="Dismiss topic suggestion",
        description="Drop a create-a-page suggestion (sets it dismissed; never public).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_dismiss))
