"""C1 console action: create a draft purchase order from a reorder-report supplier group."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import purchase_orders as _po

_LINE_KEYS = ("ingredient_id", "ingredient", "suggested_qty", "unit",
              "price_per_unit", "unit_size", "packs", "est_cost")


def _sanitize(lines):
    out = []
    for ln in (lines or []):
        if not isinstance(ln, dict):
            continue
        if ln.get("ingredient_id") is None or ln.get("suggested_qty") is None:
            continue
        out.append({k: ln.get(k) for k in _LINE_KEYS})
    return out


def _exec_create_po(params, ctx):
    sid = params.get("supplier_id")
    if sid is None:
        return {"ok": False, "error": "no supplier — assign a preferred source first"}
    lines = _sanitize(params.get("lines"))
    if not lines:
        return {"ok": False, "error": "no orderable lines"}
    res = _po.create_draft_po(ctx["cx"], int(sid), params.get("supplier_name") or "", lines)
    return {"ok": True, **res}


def register():
    if get_action("reorder.create_po"):
        return
    register_action(Action(
        key="reorder.create_po", module="reorder", title="Create draft PO",
        description="Create a draft purchase order from a reorder-report supplier group.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_create_po))
