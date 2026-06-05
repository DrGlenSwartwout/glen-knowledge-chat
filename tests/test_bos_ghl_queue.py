import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import ghl_queue as Q
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    Q.init_ghl_queue_table(cx)
    return Q, cx


def test_enqueue_validates_and_lists():
    Q, cx = _db()
    qid = Q.enqueue(cx, op="tag_add", email="a@b.com", payload={"tag": "vip"}, actor="glen")
    assert qid > 0
    pend = Q.list_pending(cx)
    assert len(pend) == 1 and pend[0]["op"] == "tag_add" and pend[0]["status"] == "pending"
    import json
    assert json.loads(pend[0]["payload_json"]) == {"tag": "vip"}


def test_enqueue_rejects_bad_op_and_blank_email():
    Q, cx = _db()
    for bad in (lambda: Q.enqueue(cx, op="nope", email="a@b.com", payload={}),
                lambda: Q.enqueue(cx, op="note", email="", payload={})):
        try:
            bad(); assert False, "expected ValueError"
        except ValueError:
            pass


def test_mark_result_removes_from_pending():
    Q, cx = _db()
    qid = Q.enqueue(cx, op="note", email="a@b.com", payload={"note": "called"})
    Q.mark_result(cx, qid, "done", "ok")
    assert Q.list_pending(cx) == []
    row = cx.execute("SELECT status, result FROM ghl_write_queue WHERE id=?", (qid,)).fetchone()
    assert row["status"] == "done" and row["result"] == "ok"


def test_crm_add_tag_action_enqueues():
    Q, cx = _db()
    from dashboard import dispatch as D, events as E, rbac as R, actions as A
    E.init_event_tables(cx)
    assert A.get_action("crm.add_tag") is not None
    res = D.dispatch_action(cx, "crm.add_tag", {"email": "a@b.com", "tag": "warm"},
                            R.Actor(role=R.OWNER, name="glen"))
    assert res["status"] == "done"
    pend = Q.list_pending(cx)
    assert len(pend) == 1 and pend[0]["op"] == "tag_add" and pend[0]["email"] == "a@b.com"
    ev = E.list_events(cx, module="crm")
    assert ev and ev[0]["action_key"] == "crm.add_tag"


def test_crm_actions_registered():
    from dashboard import ghl_queue as Q  # noqa: F401
    from dashboard import actions as A
    for k in ("crm.add_tag", "crm.log_outreach", "crm.create_opportunity", "crm.enroll_workflow"):
        assert A.get_action(k) is not None, k
    # opportunity/workflow are owner/ops only
    assert A.get_action("crm.create_opportunity").permission == ("owner", "ops")
