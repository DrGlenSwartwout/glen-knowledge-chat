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
