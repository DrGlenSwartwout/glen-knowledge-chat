"""Phase 2a-1 console actions for product-review moderation: approve / reject / feature."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import product_reviews as _pr

# Testimonial cohort tags that grant a certification level when the testimonial is APPROVED.
# Scoped to kind='testimonial' rows carrying one of these source_tags (e.g. the ASH cert-L1
# video homework). The grant is non-downgrading + idempotent; the human approval is the gate.
_CERT_TAG_LEVELS = {"ash-cert-l1": 1}


def _name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _grant_cert(email, level):
    """Best-effort idempotent cert-level grant (Supabase). Never raises into the approval."""
    try:
        from dashboard import practitioner_portal as _pp
        _pp.grant_cert_level_at_least(email, int(level))
        return True
    except Exception as e:  # noqa: BLE001 - cert grant must never block moderation
        print(f"[reviews] cert grant failed for {email!r} L{level}: {e!r}", flush=True)
        return False


def _notify_feedback(rv, outcome):
    """Auto-send the 'your feedback is ready' portal email for a reviewed cohort testimonial.
    Best-effort; only cert-tagged testimonials, never blocks moderation."""
    try:
        if (rv.get("kind") == "testimonial" and rv.get("email")
                and (rv.get("source_tag") or "").strip() in _CERT_TAG_LEVELS):
            from dashboard import cert_notify
            cert_notify.send_feedback_ready(rv["email"], rv.get("name"), outcome,
                                            practitioner_id=rv.get("practitioner_id") or 0)
    except Exception as e:  # noqa: BLE001
        print(f"[reviews] feedback-ready notify failed: {e!r}", flush=True)


def _exec_approve(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    cx = ctx["cx"]
    _pr.set_status(cx, rid, "approved", by=_name(ctx.get("actor")))
    out = {"id": rid, "status": "approved"}
    # Cohort homework: approving a cert-tagged testimonial grants that student their level.
    rv = _pr.get_review(cx, rid) or {}
    if rv.get("kind") == "testimonial" and rv.get("email"):
        lvl = _CERT_TAG_LEVELS.get((rv.get("source_tag") or "").strip())
        if lvl and _grant_cert(rv["email"], lvl):
            out["cert_granted"] = {"email": rv["email"], "level": lvl}
    _notify_feedback(rv, "approved")
    return out


def _exec_reject(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    cx = ctx["cx"]
    _pr.set_status(cx, rid, "rejected", by=_name(ctx.get("actor")))
    _notify_feedback(_pr.get_review(cx, rid) or {}, "refine")
    return {"id": rid, "status": "rejected"}


def _exec_feature(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    _pr.set_featured(ctx["cx"], rid, bool(params.get("on")))
    return {"id": rid, "featured": bool(params.get("on"))}


def _exec_set_quality(params, ctx):
    """Reviewer-entered Audio & Visual quality scores (1-10) — judged by eye/ear, not AI."""
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    _pr.set_manual_quality(ctx["cx"], rid, audio=params.get("audio") or 0,
                           visual=params.get("visual") or 0)
    r = _pr.get_review(ctx["cx"], rid) or {}
    return {"id": rid, "audio_quality": r.get("audio_quality"), "visual_quality": r.get("visual_quality")}


def _exec_gift_approve(params, ctx):
    from dashboard import review_gifts as _rg
    rid = int(params.get("review_id") or 0)
    if not rid:
        raise ValueError("review_id required")
    g = _rg.get_for_review(ctx["cx"], rid)
    if not g:
        raise ValueError("no gift for review")
    sku = (params.get("sku") or "").strip()
    if sku and _rg.valid_sku(sku):
        _rg.swap_sku(ctx["cx"], g["id"], sku, _rg.catalog_by_sku().get(sku, {}).get("label", sku))
    _rg.set_status(ctx["cx"], g["id"], "approved", by=_name(ctx.get("actor")))
    return {"review_id": rid, "status": "approved"}


def _exec_gift_reject(params, ctx):
    from dashboard import review_gifts as _rg
    rid = int(params.get("review_id") or 0)
    if not rid:
        raise ValueError("review_id required")
    g = _rg.get_for_review(ctx["cx"], rid)
    if g:
        _rg.set_status(ctx["cx"], g["id"], "rejected", by=_name(ctx.get("actor")))
    return {"review_id": rid, "status": "rejected"}


def register():
    if get_action("reviews.approve"):
        return
    register_action(Action(key="reviews.approve", module="reviews", title="Approve review",
        description="Publish a product review on its sales page.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(key="reviews.reject", module="reviews", title="Reject review",
        description="Hide a product review from the sales page.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_reject))
    register_action(Action(key="reviews.feature", module="reviews", title="Feature review",
        description="Pin/unpin a product review at the top of its section.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_feature))
    register_action(Action(key="reviews.set_quality", module="reviews", title="Set A/V quality",
        description="Save reviewer-entered Audio & Visual quality scores (1-10) on a review.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_set_quality))
    register_action(Action(key="reviews.gift_approve", module="reviews", title="Approve review gift",
        description="Approve the AI-suggested gift (optionally swap the item) for a 5-point review.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_gift_approve))
    register_action(Action(key="reviews.gift_reject", module="reviews", title="Reject review gift",
        description="Reject the AI-suggested gift for a review.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_gift_reject))
