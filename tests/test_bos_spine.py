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
