# BOS Phase 1a: Action/Event/RBAC Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared spine of the Business OS: a typed Action Registry, an actor/role policy matrix, an Event/Audit log, and a single `dispatch_action` path that both panel buttons and Justus will call, with one real action wired end to end.

**Architecture:** Four small, independently testable modules under `dashboard/` (no coupling to `app.py` internals; functions take a sqlite connection, mirroring `begin_funnel.py`). `app.py` adds four thin routes that resolve the actor, open `LOG_DB`, and delegate to the modules. Every dispatch writes an `events` row, giving the audit log and the future activity stream for free.

**Tech Stack:** Python 3, Flask, sqlite3 (stdlib), pytest. No new dependencies.

**Scope note:** This is Phase 1a (the spine). The Home signal board (Phase 1b) and the full Justus tool migration (Phase 1c) are separate plans that build on this. This plan ships a working, tested spine plus one real action (`tasks.complete_todo`).

---

## File Structure

- `dashboard/actions.py` (new): risk-tier constants, the `Action` dataclass, the `ACTION_REGISTRY`, the `@action` decorator, and registry accessors. One responsibility: declaring and finding actions.
- `dashboard/rbac.py` (new): role constants, `Actor`, the `POLICY` matrix, `policy_for()` (with the owner money threshold), and `resolve_actor()`. One responsibility: who may do what.
- `dashboard/events.py` (new): the `events` table schema, `append_event`, `list_events`, `get_event`, `set_event_status`. One responsibility: the audit/event log.
- `dashboard/dispatch.py` (new): `dispatch_action`, `approve_event`, `cancel_event`. One responsibility: the single execution path that ties registry + policy + events together.
- `dashboard/actions_tasks.py` (new): the first real registered action, `tasks.complete_todo`. Proves the pattern; later modules add their own `actions_*.py`.
- `tests/test_bos_spine.py` (new): unit tests for all four modules + the example action.
- `app.py` (modify): init the events table at startup, register the example action, and add four routes (`/api/action/<key>`, `/api/events`, `/api/events/<id>/approve`, `/api/events/<id>/cancel`).
- `tests/test_bos_routes.py` (new): route-level tests (run in CI where `app` imports).

---

## Task 1: Action Registry (`dashboard/actions.py`)

**Files:**
- Create: `dashboard/actions.py`
- Test: `tests/test_bos_spine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bos_spine.py`:

```python
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture(autouse=True)
def _clean_registry():
    from dashboard import actions as A
    saved = dict(A.ACTION_REGISTRY)
    A.ACTION_REGISTRY.clear()
    yield
    A.ACTION_REGISTRY.clear()
    A.ACTION_REGISTRY.update(saved)


def test_action_decorator_registers_and_finds():
    from dashboard import actions as A

    @A.action(key="demo.hello", module="demo", title="Hello",
              description="says hi", risk_tier=A.LOW_WRITE, permission=(A.  # noqa
                  )) if False else (lambda f: f)  # placeholder removed below
    def _ignore(params, ctx):
        return {}

    # real registration (the decorator above is a no-op guard; use the API directly)
    @A.action(key="demo.real", module="demo", title="Real",
              description="does a thing", risk_tier=A.LOW_WRITE,
              permission=("owner",))
    def real(params, ctx):
        return {"ran": True}

    got = A.get_action("demo.real")
    assert got is not None
    assert got.module == "demo"
    assert got.risk_tier == A.LOW_WRITE
    assert got.permission == ("owner",)
    assert got.executor({}, {}) == {"ran": True}
    assert [a.key for a in A.list_actions(module="demo")] == ["demo.real"]


def test_duplicate_key_raises():
    from dashboard import actions as A

    @A.action(key="demo.dup", module="demo", title="t", description="d",
              risk_tier=A.READ, permission=("owner",))
    def one(params, ctx):
        return {}

    with pytest.raises(ValueError):
        @A.action(key="demo.dup", module="demo", title="t2", description="d2",
                  risk_tier=A.READ, permission=("owner",))
        def two(params, ctx):
            return {}


def test_unknown_risk_tier_raises():
    from dashboard import actions as A
    with pytest.raises(ValueError):
        @A.action(key="demo.bad", module="demo", title="t", description="d",
                  risk_tier="banana", permission=("owner",))
        def bad(params, ctx):
            return {}
```

Note: simplify the first test by deleting the `_ignore` placeholder block before running; keep only the `demo.real` registration. The final test body should be:

```python
def test_action_decorator_registers_and_finds():
    from dashboard import actions as A

    @A.action(key="demo.real", module="demo", title="Real",
              description="does a thing", risk_tier=A.LOW_WRITE,
              permission=("owner",))
    def real(params, ctx):
        return {"ran": True}

    got = A.get_action("demo.real")
    assert got is not None
    assert got.module == "demo"
    assert got.risk_tier == A.LOW_WRITE
    assert got.permission == ("owner",)
    assert got.executor({}, {}) == {"ran": True}
    assert [a.key for a in A.list_actions(module="demo")] == ["demo.real"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bos_spine.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.actions'` ... actually `dashboard` exists, so it fails with `AttributeError`/`ImportError` for `action`.

- [ ] **Step 3: Write the implementation**

Create `dashboard/actions.py`:

```python
"""Business-OS Action Registry: every operator/agent action declared once."""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# Risk tiers (drive the autonomy policy in dashboard.rbac).
READ = "read"
LOW_WRITE = "low_write"
MONEY_SEND = "money_send"
IRREVERSIBLE = "irreversible"
RISK_TIERS = (READ, LOW_WRITE, MONEY_SEND, IRREVERSIBLE)


@dataclass
class Action:
    key: str                      # e.g. "finance.refund_order"
    module: str                   # e.g. "finance"
    title: str                    # human label (panel button + Justus tool)
    description: str              # one line; also the Justus tool description
    risk_tier: str               # one of RISK_TIERS
    permission: Tuple[str, ...]   # roles allowed (see dashboard.rbac.ROLES)
    executor: Callable            # (params: dict, ctx: dict) -> dict
    confirm_summary: Optional[Callable] = None  # (params) -> str
    reversible: bool = False


ACTION_REGISTRY = {}


def register_action(a: Action) -> Action:
    if a.risk_tier not in RISK_TIERS:
        raise ValueError(f"unknown risk tier: {a.risk_tier}")
    if a.key in ACTION_REGISTRY:
        raise ValueError(f"duplicate action key: {a.key}")
    ACTION_REGISTRY[a.key] = a
    return a


def action(*, key, module, title, description, risk_tier, permission,
           confirm_summary=None, reversible=False):
    """Decorator: register the decorated function as an Action's executor."""
    def deco(fn):
        register_action(Action(
            key=key, module=module, title=title, description=description,
            risk_tier=risk_tier, permission=tuple(permission), executor=fn,
            confirm_summary=confirm_summary, reversible=reversible))
        return fn
    return deco


def get_action(key):
    return ACTION_REGISTRY.get(key)


def list_actions(module=None):
    return [a for a in ACTION_REGISTRY.values()
            if module is None or a.module == module]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bos_spine.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/actions.py tests/test_bos_spine.py
git commit -m "feat(bos): action registry + @action decorator"
```

---

## Task 2: RBAC policy matrix (`dashboard/rbac.py`)

**Files:**
- Create: `dashboard/rbac.py`
- Test: `tests/test_bos_spine.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bos_spine.py`)

```python
def test_policy_matrix_cells():
    from dashboard import rbac as R
    from dashboard import actions as A
    assert R.policy_for(R.OWNER, A.LOW_WRITE) == R.AUTO
    assert R.policy_for(R.OWNER, A.IRREVERSIBLE) == R.CONFIRM
    assert R.policy_for(R.OPS, A.MONEY_SEND) == R.CONFIRM
    assert R.policy_for(R.VA, A.MONEY_SEND) == R.QUEUE
    assert R.policy_for(R.VA, A.IRREVERSIBLE) == R.DENY
    assert R.policy_for(R.AGENT, A.MONEY_SEND) == R.QUEUE
    assert R.policy_for(R.AGENT, A.IRREVERSIBLE) == R.DENY
    assert R.policy_for(R.SYSTEM, A.READ) == R.AUTO


def test_owner_money_threshold():
    from dashboard import rbac as R
    from dashboard import actions as A
    # threshold 0 => confirm everything
    assert R.policy_for(R.OWNER, A.MONEY_SEND, amount=10, threshold=0) == R.CONFIRM
    # threshold 50 => auto under 50, confirm at/above
    assert R.policy_for(R.OWNER, A.MONEY_SEND, amount=20, threshold=50) == R.AUTO
    assert R.policy_for(R.OWNER, A.MONEY_SEND, amount=50, threshold=50) == R.CONFIRM
    assert R.policy_for(R.OWNER, A.MONEY_SEND, amount=None, threshold=50) == R.CONFIRM


def test_resolve_actor_owner_by_console_secret():
    from dashboard import rbac as R
    a = R.resolve_actor("SEKRET", console_secret="SEKRET")
    assert a is not None and a.role == R.OWNER
    assert R.resolve_actor("wrong", console_secret="SEKRET") is None
    assert R.resolve_actor("", console_secret="") is None


def test_resolve_actor_by_token_role():
    from dashboard import rbac as R
    a = R.resolve_actor("", console_secret="SEKRET",
                        token="tok_shaira", role_for_token=lambda t: R.VA)
    assert a is not None and a.role == R.VA
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_spine.py -k "policy or resolve_actor" -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.rbac'`.

- [ ] **Step 3: Write the implementation**

Create `dashboard/rbac.py`:

```python
"""Business-OS RBAC: roles, the autonomy policy matrix, actor resolution."""
import os
from dataclasses import dataclass

from dashboard.actions import READ, LOW_WRITE, MONEY_SEND, IRREVERSIBLE

# Roles
OWNER = "owner"     # Glen
OPS = "ops"         # Rae
VA = "va"           # Shaira (scoped)
AGENT = "agent"     # Justus unattended
SYSTEM = "system"   # crons / webhooks
ROLES = (OWNER, OPS, VA, AGENT, SYSTEM)

# Policy modes
AUTO = "auto"
CONFIRM = "confirm"
QUEUE = "queue"
DENY = "deny"

# Static (actor x risk tier) policy. Owner money_send is special-cased below.
POLICY = {
    OWNER:  {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: CONFIRM, IRREVERSIBLE: CONFIRM},
    OPS:    {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: CONFIRM, IRREVERSIBLE: CONFIRM},
    VA:     {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: QUEUE,   IRREVERSIBLE: DENY},
    AGENT:  {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: QUEUE,   IRREVERSIBLE: DENY},
    SYSTEM: {READ: AUTO, LOW_WRITE: AUTO, MONEY_SEND: QUEUE,   IRREVERSIBLE: DENY},
}


def owner_money_threshold():
    """0 (default) = confirm every owner money action. Set to e.g. 50 after the
    manual break-in period to auto-approve owner money actions under $50."""
    try:
        return float(os.environ.get("OWNER_MONEY_AUTO_THRESHOLD", "0"))
    except (TypeError, ValueError):
        return 0.0


@dataclass
class Actor:
    role: str
    name: str = ""


def policy_for(role, risk_tier, *, amount=None, threshold=None):
    """Return the policy mode (AUTO/CONFIRM/QUEUE/DENY) for this actor and tier."""
    if role == OWNER and risk_tier == MONEY_SEND:
        thr = owner_money_threshold() if threshold is None else threshold
        if thr > 0 and amount is not None and amount < thr:
            return AUTO
        return CONFIRM
    return POLICY.get(role, {}).get(risk_tier, DENY)


def resolve_actor(console_key, *, console_secret, token=None, role_for_token=None):
    """Owner master key first (backward compatible), then optional token->role."""
    if console_secret and console_key and console_key == console_secret:
        return Actor(role=OWNER, name="owner")
    if token and role_for_token:
        role = role_for_token(token)
        if role in ROLES:
            return Actor(role=role, name=str(token)[:8])
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_spine.py -k "policy or resolve_actor" -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/rbac.py tests/test_bos_spine.py
git commit -m "feat(bos): rbac roles + autonomy policy matrix + actor resolution"
```

---

## Task 3: Event/Audit log (`dashboard/events.py`)

**Files:**
- Create: `dashboard/events.py`
- Test: `tests/test_bos_spine.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
import sqlite3


def _evx():
    from dashboard import events as E
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    return E, cx


def test_event_append_and_get():
    E, cx = _evx()
    eid = E.append_event(cx, actor="owner", source="panel",
                         action_key="demo.x", module="demo", risk_tier="low_write",
                         params={"a": 1}, result={"ok": True}, status="done")
    ev = E.get_event(cx, eid)
    assert ev["actor"] == "owner"
    assert ev["params"] == {"a": 1}
    assert ev["result"] == {"ok": True}
    assert ev["status"] == "done"


def test_event_list_filters():
    E, cx = _evx()
    E.append_event(cx, actor="owner", source="panel", action_key="m.a",
                   module="money", risk_tier="read", params={}, result=None,
                   status="done")
    E.append_event(cx, actor="va", source="justus", action_key="o.b",
                   module="orders", risk_tier="money_send", params={}, result=None,
                   status="pending_approval")
    assert len(E.list_events(cx)) == 2
    assert len(E.list_events(cx, status="pending_approval")) == 1
    assert len(E.list_events(cx, module="money")) == 1


def test_event_set_status():
    E, cx = _evx()
    eid = E.append_event(cx, actor="va", source="justus", action_key="o.b",
                         module="orders", risk_tier="money_send", params={},
                         result=None, status="pending_approval")
    assert E.set_event_status(cx, eid, "confirmed") is True
    assert E.get_event(cx, eid)["status"] == "confirmed"
    assert E.set_event_status(cx, 9999, "confirmed") is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_spine.py -k event -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.events'`.

- [ ] **Step 3: Write the implementation**

Create `dashboard/events.py`:

```python
"""Business-OS Event/Audit log: one append-only timeline of business events and
operator/agent actions. Functions take a sqlite connection for testability."""
import json
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_event_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT,
            source TEXT,
            action_key TEXT,
            module TEXT,
            risk_tier TEXT,
            params_json TEXT,
            result_json TEXT,
            status TEXT NOT NULL,
            reversible INTEGER DEFAULT 0,
            ref_type TEXT,
            ref_id TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_events_module ON events(module)")
    cx.commit()


def _row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json") or "{}")
    rj = d.pop("result_json")
    d["result"] = json.loads(rj) if rj else None
    d["reversible"] = bool(d.get("reversible"))
    return d


def append_event(cx, *, actor, source, action_key, module, risk_tier,
                 params, result, status, reversible=False,
                 ref_type=None, ref_id=None):
    cur = cx.execute(
        """INSERT INTO events
           (ts, actor, source, action_key, module, risk_tier,
            params_json, result_json, status, reversible, ref_type, ref_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (_now(), actor, source, action_key, module, risk_tier,
         json.dumps(params or {}),
         json.dumps(result) if result is not None else None,
         status, 1 if reversible else 0, ref_type, ref_id))
    cx.commit()
    return cur.lastrowid


def get_event(cx, event_id):
    cur = cx.execute("SELECT * FROM events WHERE id=?", (event_id,))
    return _row_to_dict(cur.fetchone())


def list_events(cx, *, limit=50, status=None, module=None):
    q = "SELECT * FROM events"
    clauses, args = [], []
    if status:
        clauses.append("status=?"); args.append(status)
    if module:
        clauses.append("module=?"); args.append(module)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    cur = cx.execute(q, tuple(args))
    return [_row_to_dict(r) for r in cur.fetchall()]


def set_event_status(cx, event_id, status):
    cur = cx.execute("UPDATE events SET status=? WHERE id=?", (status, event_id))
    cx.commit()
    return cur.rowcount > 0
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_spine.py -k event -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/events.py tests/test_bos_spine.py
git commit -m "feat(bos): event/audit log (append, list, get, set_status)"
```

---

## Task 4: The dispatch path (`dashboard/dispatch.py`)

**Files:**
- Create: `dashboard/dispatch.py`
- Test: `tests/test_bos_spine.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
def _dispatch_env():
    from dashboard import actions as A, events as E, dispatch as D, rbac as R
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    calls = {"n": 0}

    @A.action(key="demo.low", module="demo", title="Low", description="d",
              risk_tier=A.LOW_WRITE, permission=(R.OWNER, R.VA))
    def low(params, ctx):
        calls["n"] += 1
        return {"did": "low"}

    @A.action(key="demo.money", module="demo", title="Money", description="d",
              risk_tier=A.MONEY_SEND, permission=(R.OWNER, R.VA, R.AGENT),
              confirm_summary=lambda p: f"refund ${p.get('amount')}")
    def money(params, ctx):
        calls["n"] += 1
        return {"did": "money"}

    @A.action(key="demo.del", module="demo", title="Del", description="d",
              risk_tier=A.IRREVERSIBLE, permission=(R.OWNER, R.VA, R.AGENT))
    def dele(params, ctx):
        calls["n"] += 1
        return {"did": "del"}

    @A.action(key="demo.boom", module="demo", title="Boom", description="d",
              risk_tier=A.LOW_WRITE, permission=(R.OWNER,))
    def boom(params, ctx):
        raise RuntimeError("kaboom")

    return A, E, D, R, cx, calls


def test_dispatch_owner_low_write_auto_done():
    A, E, D, R, cx, calls = _dispatch_env()
    res = D.dispatch_action(cx, "demo.low", {}, R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    assert res["result"] == {"did": "low"}
    assert calls["n"] == 1
    assert E.get_event(cx, res["event_id"])["status"] == "done"


def test_dispatch_owner_money_needs_confirmation_then_runs():
    A, E, D, R, cx, calls = _dispatch_env()
    res = D.dispatch_action(cx, "demo.money", {"amount": 80}, R.Actor(role=R.OWNER))
    assert res["status"] == "needs_confirmation"
    assert "80" in res["summary"]
    assert calls["n"] == 0
    res2 = D.dispatch_action(cx, "demo.money", {"amount": 80},
                             R.Actor(role=R.OWNER), confirmed=True)
    assert res2["status"] == "done"
    assert calls["n"] == 1


def test_dispatch_va_money_queues():
    A, E, D, R, cx, calls = _dispatch_env()
    res = D.dispatch_action(cx, "demo.money", {"amount": 5}, R.Actor(role=R.VA))
    assert res["status"] == "queued"
    assert calls["n"] == 0
    assert E.get_event(cx, res["event_id"])["status"] == "pending_approval"


def test_dispatch_va_irreversible_denied():
    A, E, D, R, cx, calls = _dispatch_env()
    res = D.dispatch_action(cx, "demo.del", {}, R.Actor(role=R.VA))
    assert res["status"] == "denied"
    assert calls["n"] == 0


def test_dispatch_unknown_action_and_no_actor():
    A, E, D, R, cx, calls = _dispatch_env()
    assert D.dispatch_action(cx, "nope", {}, R.Actor(role=R.OWNER))["status"] == "error"
    assert D.dispatch_action(cx, "demo.low", {}, None)["status"] == "denied"


def test_dispatch_executor_failure_logs_failed():
    A, E, D, R, cx, calls = _dispatch_env()
    res = D.dispatch_action(cx, "demo.boom", {}, R.Actor(role=R.OWNER))
    assert res["status"] == "failed"
    assert "kaboom" in res["error"]
    assert E.get_event(cx, res["event_id"])["status"] == "failed"


def test_approve_event_runs_queued_action():
    A, E, D, R, cx, calls = _dispatch_env()
    q = D.dispatch_action(cx, "demo.money", {"amount": 5}, R.Actor(role=R.VA))
    res = D.approve_event(cx, q["event_id"], R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    assert calls["n"] == 1
    assert E.get_event(cx, q["event_id"])["status"] == "confirmed"


def test_cancel_event_marks_cancelled():
    A, E, D, R, cx, calls = _dispatch_env()
    q = D.dispatch_action(cx, "demo.money", {"amount": 5}, R.Actor(role=R.VA))
    res = D.cancel_event(cx, q["event_id"])
    assert res["status"] == "cancelled"
    assert calls["n"] == 0
    assert E.get_event(cx, q["event_id"])["status"] == "cancelled"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_spine.py -k "dispatch or approve_event or cancel_event" -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.dispatch'`.

- [ ] **Step 3: Write the implementation**

Create `dashboard/dispatch.py`:

```python
"""Business-OS single dispatch path. Panels and Justus both call dispatch_action.
Every dispatch resolves policy and writes an event."""
from dashboard.actions import get_action
from dashboard.rbac import policy_for, AUTO, CONFIRM, QUEUE, DENY
from dashboard import events as _events


def _amount_of(params):
    try:
        if params and params.get("amount") is not None:
            return float(params.get("amount"))
    except (TypeError, ValueError):
        return None
    return None


def _execute(cx, action, params, actor, source):
    actor_name = (actor.name or actor.role) if actor else "system"
    try:
        result = action.executor(params or {}, {"actor": actor})
        eid = _events.append_event(
            cx, actor=actor_name, source=source, action_key=action.key,
            module=action.module, risk_tier=action.risk_tier, params=params,
            result=result, status="done", reversible=action.reversible)
        return {"status": "done", "result": result, "event_id": eid}
    except Exception as e:  # noqa: BLE001 - we log every failure as an event
        eid = _events.append_event(
            cx, actor=actor_name, source=source, action_key=action.key,
            module=action.module, risk_tier=action.risk_tier, params=params,
            result={"error": str(e)}, status="failed", reversible=action.reversible)
        return {"status": "failed", "error": str(e), "event_id": eid}


def dispatch_action(cx, key, params, actor, *, source="panel",
                    attended=True, confirmed=False):
    action = get_action(key)
    if action is None:
        return {"status": "error", "error": f"unknown action: {key}"}
    if actor is None or actor.role not in action.permission:
        return {"status": "denied", "reason": "permission"}

    mode = policy_for(actor.role, action.risk_tier, amount=_amount_of(params))
    if mode == DENY:
        return {"status": "denied", "reason": "policy"}
    if mode == QUEUE:
        eid = _events.append_event(
            cx, actor=actor.name or actor.role, source=source, action_key=key,
            module=action.module, risk_tier=action.risk_tier, params=params,
            result=None, status="pending_approval", reversible=action.reversible)
        return {"status": "queued", "event_id": eid}
    if mode == CONFIRM and not confirmed:
        summary = (action.confirm_summary(params) if action.confirm_summary
                   else f"Confirm: {action.title}")
        return {"status": "needs_confirmation", "summary": summary,
                "key": key, "params": params}
    return _execute(cx, action, params, actor, source)


def approve_event(cx, event_id, actor):
    ev = _events.get_event(cx, event_id)
    if not ev or ev["status"] != "pending_approval":
        return {"status": "error", "error": "not a pending approval"}
    action = get_action(ev["action_key"])
    if action is None:
        return {"status": "error", "error": "unknown action"}
    res = _execute(cx, action, ev["params"], actor, source="approval")
    _events.set_event_status(cx, event_id, "confirmed")
    return res


def cancel_event(cx, event_id):
    ev = _events.get_event(cx, event_id)
    if not ev or ev["status"] != "pending_approval":
        return {"status": "error", "error": "not a pending approval"}
    _events.set_event_status(cx, event_id, "cancelled")
    return {"status": "cancelled", "event_id": event_id}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_spine.py -q`
Expected: all spine tests pass (Tasks 1-4).

- [ ] **Step 5: Commit**

```bash
git add dashboard/dispatch.py tests/test_bos_spine.py
git commit -m "feat(bos): dispatch_action + approve/cancel event flow"
```

---

## Task 5: First real action (`dashboard/actions_tasks.py`)

**Files:**
- Create: `dashboard/actions_tasks.py`
- Test: `tests/test_bos_spine.py` (append)

- [ ] **Step 1: Write the failing test** (append)

```python
def test_complete_todo_action_marks_done():
    from dashboard import actions_tasks  # registers tasks.complete_todo
    from dashboard import actions as A, dispatch as D, events as E, rbac as R
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    cx.execute("CREATE TABLE todos (id INTEGER PRIMARY KEY, status TEXT, done_at TEXT)")
    cx.execute("INSERT INTO todos (id, status) VALUES (7, 'open')")
    cx.commit()

    act = A.get_action("tasks.complete_todo")
    assert act is not None and act.risk_tier == A.LOW_WRITE

    res = D.dispatch_action(cx, "tasks.complete_todo", {"todo_id": 7, "cx": cx},
                            R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    row = cx.execute("SELECT status FROM todos WHERE id=7").fetchone()
    assert row["status"] == "done"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_spine.py -k complete_todo -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.actions_tasks'`.

- [ ] **Step 3: Write the implementation**

Create `dashboard/actions_tasks.py`:

```python
"""First real BOS action: complete a todo. Proves the registry/dispatch pattern.
The executor receives the live sqlite connection via params['cx'] (the route
layer injects LOG_DB's connection; tests inject an in-memory one)."""
from datetime import datetime, timezone

from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS, VA


@action(key="tasks.complete_todo", module="tasks", title="Complete todo",
        description="Mark a todo as done by id.", risk_tier=LOW_WRITE,
        permission=(OWNER, OPS, VA))
def complete_todo(params, ctx):
    cx = params["cx"]
    todo_id = int(params["todo_id"])
    now = datetime.now(timezone.utc).isoformat()
    cx.execute("UPDATE todos SET status='done', done_at=? WHERE id=?", (now, todo_id))
    cx.commit()
    return {"todo_id": todo_id, "status": "done"}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_spine.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/actions_tasks.py tests/test_bos_spine.py
git commit -m "feat(bos): first real action tasks.complete_todo end to end"
```

---

## Task 6: Routes in `app.py` + route tests

**Files:**
- Modify: `app.py` (add startup init + 4 routes near the other `/api/*` console routes)
- Test: `tests/test_bos_routes.py` (new)

- [ ] **Step 1: Write the failing route test**

Create `tests/test_bos_routes.py`:

```python
import importlib
import sys
from pathlib import Path

import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:  # missing env (Pinecone etc.) -> runs in CI only
        pytest.skip(f"app not importable in this env: {e}")


def test_action_route_completes_todo(monkeypatch, tmp_path):
    app_module = _load_app()
    import sqlite3
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY, status TEXT, done_at TEXT)")
    cx.execute("INSERT INTO todos (id, status) VALUES (3, 'open')")
    cx.commit(); cx.close()

    client = app_module.app.test_client()
    key = app_module.dashboard.CONSOLE_SECRET or ""
    r = client.post("/api/action/tasks.complete_todo",
                    json={"todo_id": 3},
                    headers={"X-Console-Key": key})
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "done"

    r2 = client.get("/api/events", headers={"X-Console-Key": key})
    assert r2.status_code == 200
    assert any(e["action_key"] == "tasks.complete_todo"
               for e in r2.get_json()["data"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_routes.py -q`
Expected: FAIL or SKIP. If `app` imports locally it FAILS (404 on the route). If env is missing it SKIPS (expected; this test is a CI gate).

- [ ] **Step 3: Add startup init + action import near the other initializers**

In `app.py`, just after the dashboard import block (search for `import dashboard` / `from dashboard`), add:

```python
# Business OS spine: event log + action registry population.
import sqlite3 as _sqlite3
from dashboard import events as _bos_events
from dashboard import dispatch as _bos_dispatch
from dashboard import rbac as _bos_rbac
import dashboard.actions_tasks  # noqa: F401  (registers tasks.* actions)


def _init_bos_events():
    cx = _sqlite3.connect(LOG_DB)
    try:
        _bos_events.init_event_tables(cx)
    finally:
        cx.close()


_init_bos_events()
```

- [ ] **Step 4: Add the four routes**

In `app.py`, near the other `@app.route("/api/...")` console routes, add:

```python
def _bos_actor():
    """Resolve the calling actor. Owner master key (CONSOLE_SECRET) for now;
    scoped token->role mapping is added in the RBAC-UX task of Phase 1."""
    key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
    return _bos_rbac.resolve_actor(key, console_secret=dashboard.CONSOLE_SECRET)


@app.route("/api/action/<path:key>", methods=["POST"])
def bos_action(key):
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    confirmed = bool(body.pop("confirmed", False))
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        params = dict(body)
        params["cx"] = cx  # executors that touch the DB use the live connection
        res = _bos_dispatch.dispatch_action(
            cx, key, params, actor, source="panel", confirmed=confirmed)
    finally:
        cx.close()
    res.pop("params", None)  # never echo the injected cx
    return jsonify(res)


@app.route("/api/events", methods=["GET"])
def bos_events():
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        rows = _bos_events.list_events(
            cx, limit=int(request.args.get("limit", 50)),
            status=request.args.get("status"),
            module=request.args.get("module"))
    finally:
        cx.close()
    return jsonify({"ok": True, "data": rows})


@app.route("/api/events/<int:event_id>/approve", methods=["POST"])
def bos_event_approve(event_id):
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        res = _bos_dispatch.approve_event(cx, event_id, actor)
    finally:
        cx.close()
    res.pop("params", None)
    return jsonify(res)


@app.route("/api/events/<int:event_id>/cancel", methods=["POST"])
def bos_event_cancel(event_id):
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        res = _bos_dispatch.cancel_event(cx, event_id)
    finally:
        cx.close()
    return jsonify(res)
```

Note: `tasks.complete_todo` reads `params["cx"]`; the route injects the live connection. The approve path re-runs from the stored event params, which do not include `cx`; add a guard in `complete_todo` so the executor falls back to the route-provided connection. Update `dashboard/actions_tasks.py` `complete_todo` to:

```python
def complete_todo(params, ctx):
    cx = params.get("cx") or (ctx or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection provided")
    todo_id = int(params["todo_id"])
    now = datetime.now(timezone.utc).isoformat()
    cx.execute("UPDATE todos SET status='done', done_at=? WHERE id=?", (now, todo_id))
    cx.commit()
    return {"todo_id": todo_id, "status": "done"}
```

And in `dashboard/dispatch.py` `_execute`, pass the connection into ctx so approval re-runs work:

```python
    result = action.executor(params or {}, {"actor": actor, "cx": cx})
```

- [ ] **Step 5: Run tests + commit**

Run: `python3 -m pytest tests/test_bos_spine.py -q` (must still pass; the ctx change is backward compatible since the spine tests pass `cx` in params).
Run: `python3 -m pytest tests/test_bos_routes.py -q` (PASS if `app` imports, else SKIP).

```bash
git add app.py dashboard/actions_tasks.py dashboard/dispatch.py tests/test_bos_routes.py
git commit -m "feat(bos): /api/action and /api/events routes + events table init"
```

---

## Self-Review

**Spec coverage** (against the blueprint section 3, the spine):
- 3.1 Action Registry -> Task 1.
- 3.2 single dispatch path (permission, policy, confirm/queue/deny/auto, audit) -> Task 4.
- 3.3 Event/Audit stream + pending-approval lifecycle -> Tasks 3 and 4.
- 3.4 RBAC actor identity (owner master key, role model) -> Task 2 and the `_bos_actor` route helper. Full token->role mapping is intentionally deferred to the Phase 1 RBAC-UX task (noted in the blueprint risks).
- 3.5 autonomy policy matrix incl. owner threshold + agent-always-queue money -> Task 2 (`policy_for`) and Task 4 tests.
- Generic `/api/action/<key>`, `/api/events`, approve, cancel -> Task 6.
- One real action proving the pattern -> Task 5.

Not in this plan (by design, separate plans): the Home signal board + `signal()` contract (Phase 1b), the full Justus tool migration onto the registry (Phase 1c), and the unified shell. These are listed in the blueprint sequencing.

**Placeholder scan:** the only narrative note is the deliberate cleanup of the `_ignore` block in Task 1 Step 1 (explicitly shown as the final body). No TBDs.

**Type consistency:** `Actor(role=...)`, `dispatch_action(cx, key, params, actor, ...)`, the event dict keys (`params`, `result`, `status`, `action_key`, `module`), and the result `status` values (`done|failed|queued|needs_confirmation|denied|cancelled|error`) are used identically across Tasks 3-6.
