"""Business-OS Home signal board: per-module priority signals + aggregation.

Each module may register a signal function:
    signal(cx, actor) -> {level, summary, top_actions, count}
The board aggregates all nine modules (gray default for unwired ones) and
overlays pending-approval events from the Phase 1a event log so queued actions
light up the owning module. Priority rules start as seed heuristics and are
refined from real data over time."""

# Priority colors (worst floats up on the board).
RED = "red"      # urgent, act now
AMBER = "amber"  # needs attention soon
GREEN = "green"  # healthy, nothing required
GRAY = "gray"    # idle / not wired yet

_ORDER = {GRAY: 0, GREEN: 1, AMBER: 2, RED: 3}

# The nine functional modules, in display order. Keys match Action.module.
MODULES = ("money", "crm", "orders", "marketing", "products",
           "content", "comms", "tasks", "b2b")
MODULE_TITLES = {
    "money": "Money & Finance",
    "crm": "Sales & CRM",
    "orders": "Orders & Fulfillment",
    "marketing": "Marketing & Growth",
    "products": "Products & Inventory",
    "content": "Content & Knowledge",
    "comms": "Comms & Calendar",
    "tasks": "Team & Tasks",
    "b2b": "Practitioner & B2B",
}

SIGNAL_REGISTRY = {}


def signal(module_key):
    """Decorator: register a module's signal function."""
    def deco(fn):
        SIGNAL_REGISTRY[module_key] = fn
        return fn
    return deco


def worst_level(levels):
    worst = GRAY
    for lv in levels:
        if _ORDER.get(lv, 0) > _ORDER.get(worst, 0):
            worst = lv
    return worst


def _bump(level, floor):
    """Raise `level` to at least `floor`."""
    return level if _ORDER.get(level, 0) >= _ORDER.get(floor, 0) else floor


def _pending_by_module(cx):
    cur = cx.execute(
        "SELECT module, COUNT(*) AS n FROM events "
        "WHERE status='pending_approval' GROUP BY module")
    return {row[0]: row[1] for row in cur.fetchall()}


def _default_cell():
    return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}


def aggregate_signals(cx, actor=None):
    """Return the ordered list of nine board cells."""
    pending = _pending_by_module(cx)
    cells = []
    for m in MODULES:
        fn = SIGNAL_REGISTRY.get(m)
        sig = fn(cx, actor) if fn else _default_cell()
        sig = {"level": sig.get("level", GRAY),
               "summary": sig.get("summary", ""),
               "top_actions": list(sig.get("top_actions", [])),
               "count": int(sig.get("count", 0) or 0)}
        pc = pending.get(m, 0)
        if pc:
            sig["level"] = _bump(sig["level"], AMBER)
            sig["count"] += pc
            sig["top_actions"] = (
                [{"label": f"Review {pc} pending", "href": "/console/home#pending"}]
                + sig["top_actions"])
        sig["module"] = m
        sig["title"] = MODULE_TITLES[m]
        cells.append(sig)
    return cells


@signal("tasks")
def tasks_signal(cx, actor=None):
    rows = cx.execute(
        "SELECT priority FROM todos "
        "WHERE status NOT IN ('done','dismissed','delegated')").fetchall()
    n = len(rows)
    if n == 0:
        return {"level": GREEN, "summary": "No open tasks", "top_actions": [], "count": 0}
    high = sum(1 for r in rows if (r[0] or "").lower() == "high")
    level = RED if high else AMBER
    summary = f"{n} open" + (f", {high} high priority" if high else "")
    return {"level": level, "summary": summary,
            "top_actions": [{"label": "Open task inbox", "href": "/console"}],
            "count": n}
