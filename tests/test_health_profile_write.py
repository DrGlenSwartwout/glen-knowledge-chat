import sqlite3
from dashboard import intake


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    intake.init_intake_table(cx)
    return cx


def test_self_edit_preserves_submitted_status():
    cx = _cx()
    intake.submit(cx, "a@b.com", {"sleep": "poor"}, "2026-07-01T00:00:00")  # a submitted row
    intake.save_self_edit(cx, "a@b.com", {"sleep": "improving"})
    row = intake.get_response(cx, "a@b.com")
    assert row["status"] == "submitted"            # not reset to draft
    assert row["answers"]["sleep"] == "improving"  # value updated
    assert row["answers"].get("self_edited_at") or row.get("self_edited_at")


def test_self_edit_accepts_dimension_rejects_consent():
    cx = _cx()
    intake.submit(cx, "a@b.com", {"sleep": "poor"}, "2026-07-01T00:00:00")
    intake.save_self_edit(cx, "a@b.com", {"terrain": 5, "terms": {"agreed": True}})
    ans = intake.get_response(cx, "a@b.com")["answers"]
    assert ans["terrain"] == 5      # dimension IS client-editable (self-reported, evolves with healing)
    assert "terms" not in ans       # consent stays excluded


def test_self_edit_creates_draft_when_no_existing_row():
    cx = _cx()
    intake.save_self_edit(cx, "new@b.com", {"terrain": 2})
    row = intake.get_response(cx, "new@b.com")
    assert row["status"] == "draft"
    assert row["answers"]["terrain"] == 2


def test_self_edit_merges_without_wiping_unedited_answers():
    cx = _cx()
    intake.submit(cx, "a@b.com", {"sleep": "poor", "terrain": 3}, "2026-07-01T00:00:00")
    intake.save_self_edit(cx, "a@b.com", {"terrain": 4})
    ans = intake.get_response(cx, "a@b.com")["answers"]
    assert ans["terrain"] == 4
    assert ans["sleep"] == "poor"  # untouched field survives the merge
