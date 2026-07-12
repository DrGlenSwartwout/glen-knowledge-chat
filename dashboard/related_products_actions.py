"""Console action: save Glen's manual "Dr. Glen recommends" picks for a product."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import related_store as _rs


def _exec_set(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    # Each related pick is either a bare slug string or {"slug","reason"}; store the
    # normalized {"slug","reason"} form (reason is Glen's optional explanation).
    related = []
    for item in (params.get("related") or []):
        if isinstance(item, dict):
            s = (item.get("slug") or "").strip()
            reason = (item.get("reason") or "").strip()
        elif isinstance(item, str):
            s, reason = item.strip(), ""
        else:
            continue
        if s:
            related.append({"slug": s, "reason": reason})
    _rs.save_manual(slug, related)
    return {"slug": slug, "related": related, "saved": True}


def register():
    if get_action("related_products.set"):
        return
    register_action(Action(
        key="related_products.set", module="related_products",
        title="Set related products",
        description="Save Dr. Glen's manual related-product picks for a product.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_set))
