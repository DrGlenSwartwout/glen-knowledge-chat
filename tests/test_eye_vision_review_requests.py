import sqlite3

from dashboard import eye_vision_review_requests as requests


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    return cx


def test_request_is_idempotent_and_enters_queue():
    cx = _cx()
    first = requests.request_review(
        cx, "Client@Example.com", "2026-07-25", {"title": "Eye report"})
    second = requests.request_review(
        cx, "client@example.com", "2026-07-25", {"title": "Updated"})
    assert first["id"] == second["id"]
    assert second["status"] == "requested"
    assert second["report"]["title"] == "Updated"
    assert len(requests.pending(cx)) == 1


def test_draft_edit_then_approval_persists_snapshot():
    cx = _cx()
    row = requests.request_review(
        cx, "client@example.com", "2026-07-25", {"title": "Generated"})
    draft = requests.review(
        cx, row["id"], "save_draft", {"title": "Edited"}, "Check wording")
    assert draft["status"] == "draft"
    approved = requests.review(
        cx, row["id"], "approve", {"title": "Approved"}, "Reviewed")
    assert approved["status"] == "approved"
    assert approved["report"]["title"] == "Approved"
    assert requests.pending(cx) == []


def test_repeat_request_does_not_overwrite_approved_report():
    cx = _cx()
    row = requests.request_review(
        cx, "client@example.com", "2026-07-25", {"title": "Generated"})
    requests.review(
        cx, row["id"], "approve", {"title": "Approved"}, "Reviewed")
    again = requests.request_review(
        cx, "client@example.com", "2026-07-25", {"title": "Regenerated"})
    assert again["status"] == "approved"
    assert again["report"]["title"] == "Approved"


def test_decline_removes_request_from_pending_queue():
    cx = _cx()
    row = requests.request_review(
        cx, "client@example.com", "2026-07-25", {"title": "Generated"})
    declined = requests.review(cx, row["id"], "decline", None, "Not appropriate")
    assert declined["status"] == "declined"
    assert requests.pending(cx) == []
