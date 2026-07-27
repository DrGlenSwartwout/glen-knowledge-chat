import sqlite3
import pytest
from dashboard import course_module_unlocks as u


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:"); u.init_unlock_tables(c); yield c; c.close()


def test_unlock_idempotent_and_scoped(cx):
    u.unlock_module(cx, "A@x.com", "ash", "02-body")
    u.unlock_module(cx, "a@x.com", "ash", "02-body")
    assert u.unlocked_modules(cx, "a@x.com", "ash") == {"02-body"}
    assert u.unlocked_modules(cx, "a@x.com", "other") == set()


def test_pref_set_take_clears(cx):
    u.set_unlock_pref(cx, "m@x.com", "ash", "05-family")
    assert u.take_unlock_pref(cx, "m@x.com", "ash") == "05-family"
    assert u.take_unlock_pref(cx, "m@x.com", "ash") is None  # consumed


def test_next_module_pref_then_sequential(cx):
    order = ["02-body", "03-mind", "04-spirit"]
    assert u.next_module_to_unlock(cx, "n@x.com", "ash", order) == "02-body"  # sequential default
    u.set_unlock_pref(cx, "n@x.com", "ash", "04-spirit")
    assert u.next_module_to_unlock(cx, "n@x.com", "ash", order) == "04-spirit"  # honors pref
    u.unlock_module(cx, "n@x.com", "ash", "02-body")
    # pref still 04-spirit until taken; after clearing pref, sequential skips unlocked 02-body
    u.take_unlock_pref(cx, "n@x.com", "ash")
    assert u.next_module_to_unlock(cx, "n@x.com", "ash", order) == "03-mind"
    u.unlock_module(cx, "n@x.com", "ash", "03-mind"); u.unlock_module(cx, "n@x.com", "ash", "04-spirit")
    assert u.next_module_to_unlock(cx, "n@x.com", "ash", order) is None  # all unlocked


def test_reads_never_raise():
    class Boom:
        def execute(self, *a, **k): raise RuntimeError("down")
    assert u.unlocked_modules(Boom(), "a", "b") == set()
    assert u.take_unlock_pref(Boom(), "a", "b") is None
