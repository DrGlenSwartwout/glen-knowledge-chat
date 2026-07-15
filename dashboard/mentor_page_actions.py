"""Console actions for mentor-page review: edit / approve / rebuild / dismiss.

Mirrors dashboard/topic_page_actions.py. Difference: mentor pages are about real
people, so there is no automated compliance gate. Approve is OWNER-only human
review, and publishing a page about a living person is a deliberate owner choice.
"""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import mentor_pages as _mp
from dashboard import mentor_copy as _mc

_DEPS = {}  # client, send, strip, base_url, retriever


def configure(**kw):
    _DEPS.update(kw)


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_edit(params, ctx):
    slug = (params.get("slug") or "").strip().lower()
    if not slug:
        raise ValueError("slug required")
    cx = ctx["cx"]
    section = (params.get("section") or "").strip()
    if section and section in _mc.NARRATIVE_SECTIONS:
        _mp.upsert_section(cx, slug, section, params.get("text") or "")
    if "name" in params:
        _mp.set_name(cx, slug, params.get("name") or "")
    if "field" in params:
        _mp.set_field(cx, slug, params.get("field") or "")
    if "lifespan" in params:
        _mp.set_lifespan(cx, slug, params.get("lifespan") or "")
    if "vital_status" in params:
        _mp.set_vital_status(cx, slug, params.get("vital_status") or "")
    _mp.set_state(cx, slug, "draft", by=_actor_name(ctx.get("actor")))
    return {"slug": slug, "ok": True, "state": "draft"}


def _exec_approve(params, ctx):
    slug = (params.get("slug") or "").strip().lower()
    if not slug:
        raise ValueError("slug required")
    cx = ctx["cx"]
    page = _mp.get_page(cx, slug) or {}
    if not (page.get("content") or {}):
        return {"slug": slug, "ok": False, "error": "empty_page"}
    _mp.set_state(cx, slug, "approved", by=_actor_name(ctx.get("actor")))
    try:
        send = _DEPS.get("send")
        strip = _DEPS.get("strip") or (lambda s: s)
        base = _DEPS.get("base_url", "")
        name = page.get("name") or slug
        if send is not None:
            _mp.notify_on_approve(cx, slug, name, base, send=send, strip=strip)
    except Exception as exc:  # noqa: BLE001 - notify must never fail the approve
        print(f"[mentor-pages] notify skipped: {exc}", flush=True)
    try:
        from dashboard import indexnow as _in
        base = _DEPS.get("base_url", "")
        page_url = base.rstrip("/") + "/mentors/" + slug if base else ""
        _in.submit(page_url, base_url=base, http=_DEPS.get("indexnow_http"))
    except Exception as exc:  # noqa: BLE001 - indexnow must never fail the approve
        print(f"[mentor-pages] indexnow skipped: {exc}", flush=True)
    return {"slug": slug, "ok": True, "state": "approved"}


def _exec_rebuild(params, ctx):
    slug = (params.get("slug") or "").strip().lower()
    if not slug:
        raise ValueError("slug required")
    cx = ctx["cx"]
    page = _mp.get_page(cx, slug) or {}
    name = page.get("name") or params.get("name") or slug.replace("-", " ").title()
    return _mc.build_page(cx, slug, name,
                          client=_DEPS.get("client"),
                          retriever=_DEPS.get("retriever"),
                          strip=_DEPS.get("strip"))


def _exec_dismiss(params, ctx):
    slug = (params.get("slug") or "").strip().lower()
    if not slug:
        raise ValueError("slug required")
    _mp.set_state(ctx["cx"], slug, "dismissed", by=_actor_name(ctx.get("actor")))
    return {"slug": slug, "ok": True, "state": "dismissed"}


def register():
    if get_action("mentor_page.approve"):
        return
    register_action(Action(
        key="mentor_page.approve", module="mentor_pages", title="Approve mentor page",
        description="Publish a mentor page (owner review); emails anyone who requested it.",
        risk_tier=LOW_WRITE, permission=(OWNER,), executor=_exec_approve))
    register_action(Action(
        key="mentor_page.edit", module="mentor_pages", title="Edit mentor page",
        description="Edit a section or the name/field/lifespan/vital-status (resets to draft).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="mentor_page.rebuild", module="mentor_pages", title="Rebuild mentor page",
        description="Re-draft the page from its seed, or from Pinecone `mentors` grounding.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_rebuild))
    register_action(Action(
        key="mentor_page.dismiss", module="mentor_pages", title="Dismiss mentor page",
        description="Set a mentor page to dismissed (never public).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_dismiss))
