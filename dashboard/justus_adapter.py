"""Static migration metadata mapping Justus agent tools onto BOS actions, plus
the formatter that turns a dispatch_action result back into the human-readable
string the agent loop expects. Pure (no app.py / DB dependency) for testability.

READ tools and draft_todo_reply (content generation, no mutation) bypass the
audit path. complete_todo is intentionally absent from JUSTUS_WRITE_ACTIONS: it
reuses the Phase 1a action `tasks.complete_todo`."""
from dashboard.actions import LOW_WRITE, IRREVERSIBLE

JUSTUS_READ_TOOLS = {
    "list_todos", "list_household_candidates", "list_pending_merges",
    "draft_todo_reply",
}

# tool name -> (action_key, module, risk_tier). permission is (owner, ops, va).
JUSTUS_WRITE_ACTIONS = {
    "delegate_todo":               ("tasks.delegate_todo", "tasks", LOW_WRITE),
    "dismiss_todo":                ("tasks.dismiss_todo", "tasks", LOW_WRITE),
    "add_todo":                    ("tasks.add_todo", "tasks", LOW_WRITE),
    "split_capture":               ("tasks.split_capture", "tasks", LOW_WRITE),
    "add_idea":                    ("tasks.add_idea", "tasks", LOW_WRITE),
    "move_project":                ("tasks.move_project", "tasks", LOW_WRITE),
    "set_project_field":           ("tasks.set_project_field", "tasks", LOW_WRITE),
    "drop_project":                ("tasks.drop_project", "tasks", LOW_WRITE),
    "confirm_household_candidate": ("crm.confirm_household_candidate", "crm", LOW_WRITE),
    "dismiss_household_candidate": ("crm.dismiss_household_candidate", "crm", LOW_WRITE),
    "queue_household_merge":       ("crm.queue_household_merge", "crm", LOW_WRITE),
    "apply_pending_merge":         ("crm.apply_pending_merge", "crm", IRREVERSIBLE),
    "cancel_pending_merge":        ("crm.cancel_pending_merge", "crm", LOW_WRITE),
}


def is_read(name):
    return name in JUSTUS_READ_TOOLS


def action_key_for(name):
    if name == "complete_todo":
        return "tasks.complete_todo"
    meta = JUSTUS_WRITE_ACTIONS.get(name)
    return meta[0] if meta else None


def format_justus_result(name, res):
    """Turn a dispatch_action result dict into the agent-facing string."""
    st = res.get("status")
    if st == "done":
        r = res.get("result") or {}
        if isinstance(r, dict) and r.get("message"):
            return r["message"]
        if name == "complete_todo":
            return f"Completed todo #{(r or {}).get('todo_id', '?')}."
        return f"Done: {name}."
    if st == "queued":
        return (f"Queued for approval (event #{res.get('event_id')}). It will show on "
                f"the Home board for an owner to approve before it runs.")
    if st == "needs_confirmation":
        return (f"This needs confirmation: {res.get('summary', '')}. "
                f"Confirm with the user before proceeding.")
    if st == "denied":
        return "Not permitted: you do not have approval to run this action."
    if st in ("failed", "error"):
        return f"Error: {res.get('error', 'failed')}."
    return f"Unexpected result: {res}"
