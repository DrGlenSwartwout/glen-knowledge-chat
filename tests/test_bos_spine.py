import sqlite3
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


def test_complete_todo_action_marks_done():
    import importlib
    import dashboard.actions_tasks as actions_tasks
    from dashboard import actions as A, dispatch as D, events as E, rbac as R
    # The autouse _clean_registry fixture empties the registry; if the module was
    # already imported (cached) by another test/file, re-run its registration.
    if A.get_action("tasks.complete_todo") is None:
        importlib.reload(actions_tasks)

    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    cx.execute("CREATE TABLE todos (id INTEGER PRIMARY KEY, status TEXT, done_at TEXT)")
    cx.execute("INSERT INTO todos (id, status) VALUES (7, 'open')")
    cx.commit()

    act = A.get_action("tasks.complete_todo")
    assert act is not None and act.risk_tier == A.LOW_WRITE

    res = D.dispatch_action(cx, "tasks.complete_todo", {"todo_id": 7},
                            R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    row = cx.execute("SELECT status FROM todos WHERE id=7").fetchone()
    assert row["status"] == "done"


def test_approve_event_denies_actor_without_permission():
    A, E, D, R, cx, calls = _dispatch_env()
    # demo.money permits OWNER, VA, AGENT (not OPS); VA queues it
    q = D.dispatch_action(cx, "demo.money", {"amount": 5}, R.Actor(role=R.VA))
    assert q["status"] == "queued"
    # an OPS actor (not in the action's permission) must not be able to approve it
    res = D.approve_event(cx, q["event_id"], R.Actor(role=R.OPS))
    assert res["status"] == "denied"
    assert calls["n"] == 0
    # the queued event is untouched (still pending), approver was rejected
    assert E.get_event(cx, q["event_id"])["status"] == "pending_approval"


def test_va_cannot_approve_own_queued_money_action():
    # Separation of duties: a va can SUBMIT a money_send action (it queues), but
    # cannot approve it -- only an owner/ops may. demo.money permits va + is money_send.
    A, E, D, R, cx, calls = _dispatch_env()
    q = D.dispatch_action(cx, "demo.money", {"amount": 5}, R.Actor(role=R.VA))
    assert q["status"] == "queued"
    res = D.approve_event(cx, q["event_id"], R.Actor(role=R.VA))
    assert res["status"] == "denied"
    assert calls["n"] == 0
    assert E.get_event(cx, q["event_id"])["status"] == "pending_approval"
    # an owner CAN approve it
    res2 = D.approve_event(cx, q["event_id"], R.Actor(role=R.OWNER))
    assert res2["status"] == "done"
    assert calls["n"] == 1
