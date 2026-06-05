"""First real BOS action: complete a todo. Proves the registry/dispatch pattern.
The executor receives the live sqlite connection via ctx['cx'] (the dispatch
layer injects it; tests dispatch with an in-memory connection)."""
from datetime import datetime, timezone

from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS, VA


@action(key="tasks.complete_todo", module="tasks", title="Complete todo",
        description="Mark a todo as done by id.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS, VA))
def complete_todo(params, ctx):
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection provided")
    todo_id = int(params["todo_id"])
    now = datetime.now(timezone.utc).isoformat()
    cx.execute("UPDATE todos SET status='done', done_at=? WHERE id=?", (now, todo_id))
    cx.commit()
    return {"todo_id": todo_id, "status": "done"}
