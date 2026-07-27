import sqlite3

from dashboard import module_certifications as mc


def _cx():
    cx = sqlite3.connect(":memory:")
    mc.init_table(cx)
    return cx


def test_record_purchase_inserts_pending_and_is_idempotent_on_stripe_ref():
    cx = _cx()
    ok1 = mc.record_purchase(cx, "Learner@Example.com", "course-a", "module-1", "ch_1", 20000)
    assert ok1 is True
    assert mc.status_for(cx, "learner@example.com", "course-a", "module-1") == "pending"

    ok2 = mc.record_purchase(cx, "learner@example.com", "course-a", "module-1", "ch_1", 20000)
    assert ok2 is False

    rows = cx.execute("SELECT COUNT(*) FROM module_certifications").fetchone()[0]
    assert rows == 1


def test_record_purchase_different_stripe_ref_different_row_succeeds():
    cx = _cx()
    assert mc.record_purchase(cx, "a@example.com", "course-a", "module-1", "ch_1", 20000) is True
    assert mc.record_purchase(cx, "b@example.com", "course-a", "module-2", "ch_2", 20000) is True
    rows = cx.execute("SELECT COUNT(*) FROM module_certifications").fetchone()[0]
    assert rows == 2


def test_approve_flips_pending_to_approved_and_is_idempotent():
    cx = _cx()
    mc.record_purchase(cx, "learner@example.com", "course-a", "module-1", "ch_1", 20000)

    ok = mc.approve(cx, "learner@example.com", "course-a", "module-1", "2026-07-27T00:00:00Z")
    assert ok is True
    assert mc.status_for(cx, "learner@example.com", "course-a", "module-1") == "approved"

    ok2 = mc.approve(cx, "learner@example.com", "course-a", "module-1", "2026-07-27T00:00:01Z")
    assert ok2 is False


def test_approve_on_nonexistent_row_returns_false():
    cx = _cx()
    assert mc.approve(cx, "nobody@example.com", "course-a", "module-1", "2026-07-27T00:00:00Z") is False


def test_record_purchase_returns_false_when_already_approved():
    cx = _cx()
    mc.record_purchase(cx, "learner@example.com", "course-a", "module-1", "ch_1", 20000)
    mc.approve(cx, "learner@example.com", "course-a", "module-1", "2026-07-27T00:00:00Z")

    ok = mc.record_purchase(cx, "learner@example.com", "course-a", "module-1", "ch_new", 20000)
    assert ok is False
    rows = cx.execute("SELECT COUNT(*) FROM module_certifications").fetchone()[0]
    assert rows == 1


def test_certified_modules_returns_only_approved():
    cx = _cx()
    mc.record_purchase(cx, "learner@example.com", "course-a", "module-1", "ch_1", 20000)
    mc.record_purchase(cx, "learner@example.com", "course-a", "module-2", "ch_2", 20000)
    mc.approve(cx, "learner@example.com", "course-a", "module-1", "2026-07-27T00:00:00Z")

    assert mc.certified_modules(cx, "learner@example.com", "course-a") == {"module-1"}


def test_all_certified_true_only_when_every_required_module_approved():
    cx = _cx()
    mc.record_purchase(cx, "learner@example.com", "course-a", "module-1", "ch_1", 20000)
    mc.record_purchase(cx, "learner@example.com", "course-a", "module-2", "ch_2", 20000)
    mc.approve(cx, "learner@example.com", "course-a", "module-1", "2026-07-27T00:00:00Z")

    assert mc.all_certified(cx, "learner@example.com", "course-a", ["module-1", "module-2"]) is False

    mc.approve(cx, "learner@example.com", "course-a", "module-2", "2026-07-27T00:00:01Z")
    assert mc.all_certified(cx, "learner@example.com", "course-a", ["module-1", "module-2"]) is True


def test_all_certified_false_on_empty_required_list():
    cx = _cx()
    assert mc.all_certified(cx, "learner@example.com", "course-a", []) is False
    assert mc.all_certified(cx, "learner@example.com", "course-a", None) is False
