# tests/test_cert_submissions.py
import sqlite3
from dashboard import cert_submissions as cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cs.init_tables(cx)
    return cx


def _make(cx, email="doc@x.com", modules=(1, 2), formats=("article",)):
    return cs.create(
        cx, sid="id-" + email + "-" + str(modules),
        email=email, practitioner_id="p1", title="My case",
        description="What happened", url="https://ex.com/post", file_path="",
        formats=list(formats), format_other="", modules=list(modules),
        module_other="", topic_angle="transformation", permission=1,
    )


def test_create_and_get_roundtrips_json():
    cx = _cx()
    sid = _make(cx, modules=(1, 2, 3), formats=("article", "demo_video"))
    row = cs.get(cx, sid)
    assert row["email"] == "doc@x.com"
    assert row["status"] == "submitted"
    assert row["modules"] == [1, 2, 3]
    assert row["formats"] == ["article", "demo_video"]
    assert row["credited_modules"] == []   # empty until approved
    assert row["permission"] == 1


def test_list_for_email_and_by_status():
    cx = _cx()
    _make(cx, email="a@x.com", modules=(1,))
    _make(cx, email="a@x.com", modules=(2,))
    _make(cx, email="b@x.com", modules=(3,))
    assert len(cs.list_for_email(cx, "a@x.com")) == 2
    assert len(cs.list_by_status(cx, "submitted")) == 3
    assert len(cs.list_by_status(cx, None)) == 3
    assert cs.list_by_status(cx, "approved") == []


def test_set_status_approve_sets_credited_modules():
    cx = _cx()
    sid = _make(cx, modules=(1, 2))
    cs.set_status(cx, sid, "approved", credited_modules=[1, 2], review_note="great")
    row = cs.get(cx, sid)
    assert row["status"] == "approved"
    assert row["credited_modules"] == [1, 2]
    assert row["review_note"] == "great"


def test_set_status_publish_records_case_study_id():
    cx = _cx()
    sid = _make(cx)
    cs.set_status(cx, sid, "approved", credited_modules=[1])
    cs.set_status(cx, sid, "published", case_study_id="cert-id-1")
    row = cs.get(cx, sid)
    assert row["status"] == "published"
    assert row["case_study_id"] == "cert-id-1"
    assert row["credited_modules"] == [1]   # preserved across the publish update


def test_get_missing_returns_none():
    cx = _cx()
    assert cs.get(cx, "nope") is None
