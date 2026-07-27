import sqlite3
import pytest
from dashboard import course_progress as cp


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:")
    cp.init_progress_tables(c)
    yield c
    c.close()


def test_watched_is_idempotent_and_scoped(cx):
    cp.mark_watched(cx, "A@x.com", "ash", "02-body", "01-a")
    cp.mark_watched(cx, "a@x.com", "ash", "02-body", "01-a")  # normalized + idempotent
    assert cp.watched_lessons(cx, "a@x.com", "ash", "02-body") == {"01-a"}
    cp.mark_watched(cx, "a@x.com", "ash", "02-body", "02-b")
    assert cp.watched_lessons(cx, "a@x.com", "ash", "02-body") == {"01-a", "02-b"}
    assert cp.watched_lessons(cx, "a@x.com", "ash", "03-mind") == set()


def test_homework_upsert(cx):
    assert cp.homework(cx, "m@x.com", "ash", "02-body") is None
    cp.record_homework(cx, "m@x.com", "ash", "02-body", "my reflection", ai_rating="good", ai_feedback="go deeper")
    hw = cp.homework(cx, "m@x.com", "ash", "02-body")
    assert hw["payload"] == "my reflection" and hw["ai_rating"] == "good" and hw["submitted_at"]
    cp.record_homework(cx, "m@x.com", "ash", "02-body", "revised")  # upsert, no ai fields
    assert cp.homework(cx, "m@x.com", "ash", "02-body")["payload"] == "revised"


def test_module_completed_requires_all_watched_and_homework(cx):
    lessons = ["01-a", "02-b"]
    assert cp.module_completed(cx, "c@x.com", "ash", "02-body", lessons) is False
    cp.mark_watched(cx, "c@x.com", "ash", "02-body", "01-a")
    cp.record_homework(cx, "c@x.com", "ash", "02-body", "done")
    assert cp.module_completed(cx, "c@x.com", "ash", "02-body", lessons) is False  # 02-b not watched
    cp.mark_watched(cx, "c@x.com", "ash", "02-body", "02-b")
    assert cp.module_completed(cx, "c@x.com", "ash", "02-body", lessons) is True
    assert cp.module_completed(cx, "c@x.com", "ash", "02-body", []) is False  # no lessons → not complete


def test_reads_never_raise_on_broken_cx():
    class Boom:
        def execute(self, *a, **k): raise RuntimeError("db down")
    assert cp.watched_lessons(Boom(), "a", "b", "c") == set()
    assert cp.homework(Boom(), "a", "b", "c") is None
    assert cp.module_completed(Boom(), "a", "b", "c", ["x"]) is False
