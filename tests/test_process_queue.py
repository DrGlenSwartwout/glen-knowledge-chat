import sqlite3
from dashboard import process_queue as Q


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); Q.init_table(cx); return cx


def test_enqueue_idempotent_list_and_done(tmp_path):
    cx = _cx(tmp_path)
    Q.enqueue(cx, "a@x.com", "2026-06-05")
    Q.enqueue(cx, "a@x.com", "2026-06-05")            # idempotent: still one pending row
    Q.enqueue(cx, "b@x.com", "")
    pend = Q.list_pending(cx)
    emails = sorted(p["email"] for p in pend)
    assert emails == ["a@x.com", "b@x.com"] and len(pend) == 2
    assert Q.mark_done(cx, "a@x.com") is True
    assert [p["email"] for p in Q.list_pending(cx)] == ["b@x.com"]
    assert Q.mark_done(cx, "nobody@x.com") is False


def test_done_then_reenqueue(tmp_path):
    cx = _cx(tmp_path)
    Q.enqueue(cx, "a@x.com", "")
    Q.mark_done(cx, "a@x.com")
    Q.enqueue(cx, "a@x.com", "2026-07-01")            # a later scan re-requests
    assert len(Q.list_pending(cx)) == 1
