# BOS Phase 1c: Justus -> dispatch_action Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Route the Justus agent's WRITE tools through the audited `dispatch_action` path so every Justus write is logged to the event spine (visible on the Home board) and governed by the autonomy matrix, while preserving the agent's tools, prompts, UX, and existing executor logic.

**Architecture:** An adapter approach. The agent's tool definitions, system prompt, and streaming loop are unchanged. Only the tool *dispatcher* changes: READ tools (`list_*`, `draft_todo_reply`) run the existing executor directly (no audit noise); WRITE tools dispatch through `dispatch_action` (policy + event log), reusing the existing executor as the action's executor. Pure helpers (role mapping, tool->action map, result formatting) live in `dashboard/` and are unit-tested locally; the app.py wiring is verified under doppler because `app.py` validates Pinecone at import.

**Builds on:** Phase 1a (registry/dispatch/events/rbac) + 1b (signals/Home). Same branch `sess/ec0e1f15`, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Decisions (confirmed with Glen):**
- Rae is `owner` (full). Role map: admin key -> owner; `workspace:rae` -> owner; `workspace:shaira` -> va; any other scoped owner -> va (safe default).
- Owner/ops dispatch passes `confirmed=True` (preserves today's prompt-based confirm UX). The autonomy matrix still enforces queue/deny for `va` and the unattended agent.
- Module mapping: projects + todos -> `tasks`; household/merge -> `crm`. So Justus actions light up those Home cells.
- `complete_todo` unifies with the existing 1a action `tasks.complete_todo`. The other write tools register new actions that wrap the existing executors.
- READ tools and `draft_todo_reply` (content generation, no mutation) are not audited.

---

## File Structure

- `dashboard/rbac.py` (modify): add `SCOPE_ROLES`, `role_for_owner`, `actor_for_scope`.
- `dashboard/justus_adapter.py` (new): the pure tool->action map, READ set, `action_key_for`, `is_read`, `format_justus_result`. One responsibility: the static migration metadata + result formatting.
- `tests/test_bos_justus_adapter.py` (new): unit tests for the rbac additions + the adapter helpers.
- `app.py` (modify): register the Justus write actions (wrapping existing executors), add the `_justus_tool_dispatch(actor)` adapter, and wire `console_ask` to build the actor + pass the adapter to `_ask_justus_stream_tools`.

---

## Task 1: Role mapping in `dashboard/rbac.py`

**Files:**
- Modify: `dashboard/rbac.py`
- Test: `tests/test_bos_justus_adapter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bos_justus_adapter.py`:

```python
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_role_for_owner():
    from dashboard import rbac as R
    assert R.role_for_owner("rae") == R.OWNER
    assert R.role_for_owner("Rae") == R.OWNER
    assert R.role_for_owner("shaira") == R.VA
    assert R.role_for_owner("someone-else") == R.VA
    assert R.role_for_owner("") == R.VA


def test_actor_for_scope():
    from dashboard import rbac as R
    a = R.actor_for_scope("admin")
    assert a.role == R.OWNER
    assert R.actor_for_scope("").role == R.OWNER
    assert R.actor_for_scope("workspace:rae").role == R.OWNER
    assert R.actor_for_scope("workspace:shaira").role == R.VA
    assert R.actor_for_scope("workspace:shaira").name == "shaira"
    assert R.actor_for_scope("workspace:unknown").role == R.VA
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_justus_adapter.py -k "role_for_owner or actor_for_scope" -q`
Expected: FAIL with `AttributeError: module 'dashboard.rbac' has no attribute 'role_for_owner'`.

- [ ] **Step 3: Append to `dashboard/rbac.py`** (after `resolve_actor`)

```python
# Justus / console actor mapping. Rae is owner; Shaira is the scoped VA.
SCOPE_ROLES = {"rae": OWNER, "shaira": VA}


def role_for_owner(owner):
    """Map a console owner string (glen/rae/shaira/...) to a role. Glen and Rae
    are owners; Shaira is the VA; any unknown scoped owner defaults to VA."""
    o = (owner or "").lower()
    if o in ("glen", "owner", ""):
        return OWNER
    return SCOPE_ROLES.get(o, VA)


def actor_for_scope(scope, owner_hint=""):
    """Build an Actor from an auth scope. 'admin' (or empty) -> owner. A scoped
    token 'workspace:<owner>' -> the owner's role (rae owner, shaira va)."""
    if not scope or scope == "admin":
        return Actor(role=OWNER, name="owner")
    if scope.startswith("workspace:"):
        o = scope.split(":", 1)[1]
    else:
        o = owner_hint or ""
    return Actor(role=role_for_owner(o), name=o or "scoped")
```

Note: `role_for_owner("")` must return OWNER per the empty-string branch above; but the test asserts `role_for_owner("") == VA`. Reconcile by removing `""` from the owner branch. Final `role_for_owner`:

```python
def role_for_owner(owner):
    o = (owner or "").lower()
    if o in ("glen", "owner"):
        return OWNER
    return SCOPE_ROLES.get(o, VA)
```

(With this, `role_for_owner("")` -> VA, matching the test. `actor_for_scope("admin")` still returns OWNER via its own `admin`/empty branch, which is what matters for auth.)

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_justus_adapter.py -k "role_for_owner or actor_for_scope" -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/rbac.py tests/test_bos_justus_adapter.py
git commit -m "feat(bos): console actor/role mapping (rae owner, shaira va)"
```

---

## Task 2: The adapter metadata + formatter (`dashboard/justus_adapter.py`)

**Files:**
- Create: `dashboard/justus_adapter.py`
- Test: `tests/test_bos_justus_adapter.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_read_and_key_mapping():
    from dashboard import justus_adapter as J
    assert J.is_read("list_todos")
    assert J.is_read("draft_todo_reply")
    assert not J.is_read("apply_pending_merge")
    assert J.action_key_for("complete_todo") == "tasks.complete_todo"
    assert J.action_key_for("apply_pending_merge") == "crm.apply_pending_merge"
    assert J.action_key_for("add_idea") == "tasks.add_idea"
    assert J.action_key_for("not_a_tool") is None


def test_write_actions_risk_tiers():
    from dashboard import justus_adapter as J
    from dashboard import actions as A
    # apply_pending_merge is the only irreversible one
    assert J.JUSTUS_WRITE_ACTIONS["apply_pending_merge"][2] == A.IRREVERSIBLE
    assert J.JUSTUS_WRITE_ACTIONS["delegate_todo"][2] == A.LOW_WRITE
    assert J.JUSTUS_WRITE_ACTIONS["confirm_household_candidate"][1] == "crm"
    assert J.JUSTUS_WRITE_ACTIONS["add_idea"][1] == "tasks"
    # complete_todo is NOT here (it reuses the 1a action)
    assert "complete_todo" not in J.JUSTUS_WRITE_ACTIONS


def test_format_justus_result():
    from dashboard import justus_adapter as J
    assert J.format_justus_result("delegate_todo",
        {"status": "done", "result": {"message": "Delegated #5"}}) == "Delegated #5"
    assert J.format_justus_result("complete_todo",
        {"status": "done", "result": {"todo_id": 9, "status": "done"}}) == "Completed todo #9."
    q = J.format_justus_result("apply_pending_merge", {"status": "queued", "event_id": 12})
    assert "queued" in q.lower() and "12" in q
    assert "not permitted" in J.format_justus_result("apply_pending_merge",
        {"status": "denied", "reason": "policy"}).lower()
    assert J.format_justus_result("x", {"status": "failed", "error": "boom"}) == "Error: boom."
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_justus_adapter.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.justus_adapter'`.

- [ ] **Step 3: Write the implementation**

Create `dashboard/justus_adapter.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_justus_adapter.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/justus_adapter.py tests/test_bos_justus_adapter.py
git commit -m "feat(bos): justus tool->action map + result formatter (pure)"
```

---

## Task 3: Wire the adapter into `app.py` (verified under doppler)

**Files:**
- Modify: `app.py`

This task touches code that imports Pinecone at load, so it cannot run under local pytest. Verify with the doppler commands in Step 5.

- [ ] **Step 1: Register the Justus write actions** (after `_execute_console_tool` is defined, search for `def _execute_console_tool`)

Add below it:

```python
import dashboard.justus_adapter as _ja


def _register_justus_actions():
    """Register each Justus WRITE tool as a BOS action whose executor wraps the
    existing _execute_console_tool, so dispatch_action audits + governs it."""
    from dashboard.actions import action as _act, get_action as _get
    for _name, (_key, _module, _tier) in _ja.JUSTUS_WRITE_ACTIONS.items():
        if _get(_key):
            continue

        def _make(nm):
            def _exec(params, ctx):
                return {"message": _execute_console_tool(nm, params or {})}
            return _exec

        _act(key=_key, module=_module, title=_key,
             description=f"Justus action: {_name}", risk_tier=_tier,
             permission=(_bos_rbac.OWNER, _bos_rbac.OPS, _bos_rbac.VA))(_make(_name))


_register_justus_actions()
```

- [ ] **Step 2: Add the adapter** (near the registration above)

```python
def _justus_tool_dispatch(actor):
    """Return a tool_dispatch(name, input)->str for _ask_justus_stream_tools.
    READ tools run direct; WRITE tools go through dispatch_action (audit + policy)."""
    def dispatch(name, inp):
        inp = inp or {}
        if _ja.is_read(name):
            return _execute_console_tool(name, inp)
        key = _ja.action_key_for(name)
        if not key:
            return _execute_console_tool(name, inp)
        params = dict(inp)
        if name == "complete_todo" and "id" in params:
            params["todo_id"] = params.pop("id")
        cx = _sqlite3.connect(LOG_DB)
        cx.row_factory = _sqlite3.Row
        try:
            res = _bos_dispatch.dispatch_action(
                cx, key, params, actor, source="justus",
                confirmed=(actor.role in (_bos_rbac.OWNER, _bos_rbac.OPS)))
        finally:
            cx.close()
        return _ja.format_justus_result(name, res)
    return dispatch
```

- [ ] **Step 3: Wire `console_ask` to build the actor and use the adapter**

Find the `console_ask` route (search `def console_ask`) and the call to `_ask_justus_stream_tools(...)`. It currently passes `_execute_console_tool` as the `tool_dispatch` argument. Replace that argument with `_justus_tool_dispatch(_actor)`, where `_actor` is built right before the call from the already-resolved `ctx` and `owner`:

```python
    _actor = _bos_rbac.actor_for_scope((ctx or {}).get("scope", "admin"), owner)
```

So the call becomes (keep all other arguments exactly as they are):

```python
    return Response(_ask_justus_stream_tools(
        query, system, history, tools, _justus_tool_dispatch(_actor), ...),
        mimetype="text/event-stream")
```

(Only the `tool_dispatch` positional/keyword argument changes from `_execute_console_tool` to `_justus_tool_dispatch(_actor)`. Do not change `query`, `system`, `history`, `tools`, or any others.)

- [ ] **Step 4: Compile check**

Run: `python3 -m py_compile app.py`
Expected: OK.

- [ ] **Step 5: Verify under doppler (real env)**

Run the registry-correctness + import check (app imports Pinecone, so use real creds + a local DATA_DIR):

```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app
from dashboard import actions as A, rbac as R, justus_adapter as J
# every write tool registered with the right module + tier
for name,(key,module,tier) in J.JUSTUS_WRITE_ACTIONS.items():
    a = A.get_action(key)
    assert a is not None, "missing "+key
    assert a.module==module and a.risk_tier==tier, key
# complete_todo still the 1a action
assert A.get_action("tasks.complete_todo") is not None
# the irreversible one is governed: owner confirm, va/agent deny
assert R.policy_for(R.OWNER, A.IRREVERSIBLE) == R.CONFIRM
assert R.policy_for(R.VA, A.IRREVERSIBLE) == R.DENY
print("JUSTUS_MIGRATION_OK", len(J.JUSTUS_WRITE_ACTIONS), "actions")
PY'
rm -rf /tmp/bostest
```
Expected: prints `JUSTUS_MIGRATION_OK 13 actions` with no assertion error.

Also confirm the local suite still passes:
```bash
python3 -m pytest tests/test_bos_spine.py tests/test_bos_signals.py tests/test_bos_justus_adapter.py -q
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(bos): route Justus write tools through dispatch_action (audit + policy)"
```

---

## Self-Review

**Spec coverage** (blueprint section 5.10 "Justus, the cross-cutting operator"):
- Justus write actions execute through `dispatch_action` -> Task 3 adapter.
- Every write audited (event log) + on the Home board -> falls out of dispatch writing events; module mapping (tasks/crm) -> Task 2 map; Home overlay already built in 1b.
- Policy enforced: va/agent risky actions queue/deny; owner unchanged -> Task 1 role map + Task 3 `confirmed=(owner/ops)`.
- Tools/prompts/loop unchanged -> only the dispatcher swapped.

**Not in this phase (by design):** generating the agent tool list from the registry; new cross-module Justus actions (money/orders/etc.); enforced owner-confirm round-trip in the agent UI (owner keeps the current prompt-based confirm). These come with later modules.

**Placeholder scan:** none. The Task 1 Step 3 note explicitly gives the final `role_for_owner` body to use.

**Type consistency:** `Actor(role=...)`, `action_key_for`/`is_read`/`format_justus_result`, the `JUSTUS_WRITE_ACTIONS` tuple shape `(key, module, tier)`, and the dispatch result `status` values match Tasks 1-3 and the Phase 1a `dispatch_action` contract.
