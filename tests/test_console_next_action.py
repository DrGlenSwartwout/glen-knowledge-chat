import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dashboard.console_next_action as na


def test_reveal_draft_offers_approve_send_plus_approve_only():
    d = na.resolve_biofield_reveal(
        {"id": 7, "email": "a@b.co", "scan_date": "2026-07-01",
         "first_approved": 0, "notified_at": None, "age_ts": "2026-07-01T00:00:00"})
    assert d["actionable"] and d["state"] == "draft"
    assert d["label"] == "Approve & send" and d["confirm"] is True
    assert d["action"] == {"kind": "dispatch",
                           "keys": ["biofield_reveal.approve", "biofield_reveal.send"],
                           "body": {"id": 7}}
    assert d["secondary"]["label"] == "Approve only, don't email"
    assert d["secondary"]["action"]["keys"] == ["biofield_reveal.approve"]


def test_reveal_approved_unsent_offers_send():
    d = na.resolve_biofield_reveal(
        {"id": 9, "email": "a@b.co", "scan_date": "x", "first_approved": 1,
         "notified_at": None, "age_ts": "t"})
    assert d["actionable"] and d["state"] == "approved_unsent"
    assert d["label"] == "Send reveal link"
    assert d["action"]["keys"] == ["biofield_reveal.send"] and d["secondary"] is None


def test_reveal_sent_is_done():
    d = na.resolve_biofield_reveal(
        {"id": 9, "first_approved": 1, "notified_at": "2026-07-02T00:00:00"})
    assert d == {"actionable": False}


def test_ff_draft_offers_publish_and_edit_link():
    d = na.resolve_ff_match_draft(
        {"email": "a@b.co", "scan_date": "2026-07-01", "status": "draft", "age_ts": "t"})
    assert d["actionable"] and d["label"] == "Publish" and d["confirm"] is True
    assert d["action"] == {"kind": "post", "url": "/api/console/ff-match-drafts/publish",
                           "body": {"email": "a@b.co", "scan_date": "2026-07-01"}}
    assert d["secondary"]["action"] == {"kind": "link", "url": "/console/ff-drafts"}


def test_ff_published_is_done():
    assert na.resolve_ff_match_draft({"status": "published"}) == {"actionable": False}


def test_handoff_ai_draft_offers_publish_only_primary_and_notify_secondary():
    d = na.resolve_handoff({"email": "a@b.co", "biofield_status": "ai_draft", "age_ts": "t"})
    assert d["actionable"] and d["label"] == "Review & publish"
    assert d["action"] == {"kind": "post", "url": "/api/console/biofield/publish",
                           "body": {"email": "a@b.co", "send": False}}
    assert d["secondary"]["label"] == "Publish & notify client"
    assert d["secondary"]["action"]["body"] == {"email": "a@b.co", "send": True}


def test_handoff_confirmed_is_done():
    assert na.resolve_handoff({"biofield_status": "confirmed"}) == {"actionable": False}
