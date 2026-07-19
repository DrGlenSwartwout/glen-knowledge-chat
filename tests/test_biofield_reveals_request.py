"""Option B: the reveal-page 'request review' button stamps a per-reveal
requested_at flag that the console bucketing reads to promote the scan into
'Review these'. Pure-sqlite unit test of the store function."""
import sqlite3

from dashboard import biofield_reveals as br


def _db():
    cx = sqlite3.connect(":memory:")
    br.init_table(cx)
    return cx


def test_mark_requested_stamps_once_and_is_idempotent():
    cx = _db()
    rid, is_new = br.upsert(cx, "a@x.com", "2026-07-18", {}, [], "test")
    assert is_new
    assert not br.get(cx, rid).get("requested_at")      # not requested yet
    assert br.mark_requested(cx, rid) is True            # first request stamps it
    ts = br.get(cx, rid)["requested_at"]
    assert ts
    assert br.mark_requested(cx, rid) is False           # already requested -> no-op
    assert br.get(cx, rid)["requested_at"] == ts         # original timestamp preserved


def test_requested_at_flows_through_the_row_dict():
    cx = _db()
    rid, _ = br.upsert(cx, "b@x.com", "2026-07-18", {}, [], "test")
    br.mark_requested(cx, rid)
    # _row() must surface requested_at so the console API can bucket on it
    assert br.get(cx, rid).get("requested_at")
