import pytest
from dashboard import courses_access as ca


@pytest.mark.parametrize(
    "access,level,visible",
    [
        ("public", 0, True),
        ("public", 1, True),
        ("member", 0, False),
        ("member", 1, True),
        ("member", 2, True),
        ("paid", 1, False),
        ("paid", 2, True),
    ],
)
def test_is_visible(access, level, visible):
    assert ca.is_visible(access, level) is visible


@pytest.mark.parametrize(
    "access,level,state",
    [
        ("public", 0, "open"),
        ("member", 0, "locked_register"),
        ("member", 1, "open"),
        ("paid", 1, "locked_upgrade"),
        ("paid", 2, "open"),
    ],
)
def test_lock_state(access, level, state):
    assert ca.lock_state(access, level) == state


def test_unknown_access_is_never_visible():
    assert ca.is_visible("bogus", 2) is False
