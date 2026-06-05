import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_role_for_owner():
    from dashboard import rbac as R
    assert R.role_for_owner("rae") == R.OWNER
    assert R.role_for_owner("Rae") == R.OWNER
    assert R.role_for_owner("shaira") == R.VA
    assert R.role_for_owner("someone-else") == R.VA
    assert R.role_for_owner("") == R.VA


def test_actor_for_scope():
    from dashboard import rbac as R
    a = R.actor_for_scope("admin")
    assert a.role == R.OWNER
    assert R.actor_for_scope("").role == R.OWNER
    assert R.actor_for_scope("workspace:rae").role == R.OWNER
    assert R.actor_for_scope("workspace:shaira").role == R.VA
    assert R.actor_for_scope("workspace:shaira").name == "shaira"
    assert R.actor_for_scope("workspace:unknown").role == R.VA


def test_read_and_key_mapping():
    from dashboard import justus_adapter as J
    assert J.is_read("list_todos")
    assert J.is_read("draft_todo_reply")
    assert not J.is_read("apply_pending_merge")
    assert J.action_key_for("complete_todo") == "tasks.complete_todo"
    assert J.action_key_for("apply_pending_merge") == "crm.apply_pending_merge"
    assert J.action_key_for("add_idea") == "tasks.add_idea"
    assert J.action_key_for("not_a_tool") is None


def test_write_actions_risk_tiers():
    from dashboard import justus_adapter as J
    from dashboard import actions as A
    # apply_pending_merge is the only irreversible one
    assert J.JUSTUS_WRITE_ACTIONS["apply_pending_merge"][2] == A.IRREVERSIBLE
    assert J.JUSTUS_WRITE_ACTIONS["delegate_todo"][2] == A.LOW_WRITE
    assert J.JUSTUS_WRITE_ACTIONS["confirm_household_candidate"][1] == "crm"
    assert J.JUSTUS_WRITE_ACTIONS["add_idea"][1] == "tasks"
    # complete_todo is NOT here (it reuses the 1a action)
    assert "complete_todo" not in J.JUSTUS_WRITE_ACTIONS


def test_format_justus_result():
    from dashboard import justus_adapter as J
    assert J.format_justus_result("delegate_todo",
        {"status": "done", "result": {"message": "Delegated #5"}}) == "Delegated #5"
    assert J.format_justus_result("complete_todo",
        {"status": "done", "result": {"todo_id": 9, "status": "done"}}) == "Completed todo #9."
    q = J.format_justus_result("apply_pending_merge", {"status": "queued", "event_id": 12})
    assert "queued" in q.lower() and "12" in q
    assert "not permitted" in J.format_justus_result("apply_pending_merge",
        {"status": "denied", "reason": "policy"}).lower()
    assert J.format_justus_result("x", {"status": "failed", "error": "boom"}) == "Error: boom."
