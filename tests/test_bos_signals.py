import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import events as E
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    cx.execute("CREATE TABLE todos (id INTEGER PRIMARY KEY, status TEXT, priority TEXT)")
    cx.commit()
    return cx


def test_worst_level_ordering():
    from dashboard import signals as S
    assert S.worst_level([S.GRAY, S.GREEN, S.AMBER]) == S.AMBER
    assert S.worst_level([S.GREEN, S.RED, S.AMBER]) == S.RED
    assert S.worst_level([]) == S.GRAY
    assert S.worst_level([S.GRAY, S.GREEN]) == S.GREEN


def test_aggregate_returns_nine_cells_in_order():
    from dashboard import signals as S
    cx = _db()
    cells = S.aggregate_signals(cx, actor=None)
    assert [c["module"] for c in cells] == list(S.MODULES)
    for c in cells:
        assert set(("module", "title", "level", "summary", "top_actions", "count")) <= set(c)
        assert c["level"] in (S.RED, S.AMBER, S.GREEN, S.GRAY)


def test_tasks_signal_levels():
    from dashboard import signals as S
    cx = _db()
    # no open todos -> green
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    assert cells["tasks"]["level"] == S.GREEN
    # an open normal todo -> amber
    cx.execute("INSERT INTO todos (status, priority) VALUES ('open','normal')")
    cx.commit()
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    assert cells["tasks"]["level"] == S.AMBER
    assert cells["tasks"]["count"] == 1
    # an open high-priority todo -> red
    cx.execute("INSERT INTO todos (status, priority) VALUES ('open','high')")
    cx.commit()
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    assert cells["tasks"]["level"] == S.RED


def test_unwired_module_defaults_gray():
    from dashboard import signals as S
    cx = _db()
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    assert cells["money"]["level"] == S.GRAY


def test_pending_approval_overlay_lights_up_module():
    from dashboard import signals as S
    from dashboard import events as E
    cx = _db()
    E.append_event(cx, actor="va", source="justus", action_key="finance.refund_order",
                   module="money", risk_tier="money_send", params={"amount": 5},
                   result=None, status="pending_approval")
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    money = cells["money"]
    assert money["level"] in (S.AMBER, S.RED)  # bumped up from gray
    assert money["count"] >= 1
    assert any("pending" in (a.get("label", "").lower()) for a in money["top_actions"])
