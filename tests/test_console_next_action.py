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


def test_handoff_ai_draft_offers_composer_deep_link():
    d = na.resolve_handoff({"email": "a@b.co", "biofield_status": "ai_draft", "age_ts": "t"})
    assert d["actionable"] and d["label"] == "Review & publish"
    assert d["action"] == {"kind": "link",
                           "url": "/console/biofield-portal?email=a%40b.co"}
    assert d["confirm"] is False
    assert d["secondary"] is None


def test_handoff_confirmed_is_done():
    assert na.resolve_handoff({"biofield_status": "confirmed"}) == {"actionable": False}


def _seed_cx():
    import sqlite3
    from dashboard import biofield_reveals, ff_match_drafts
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    biofield_reveals.init_table(cx)
    ff_match_drafts.init_table(cx)
    return cx


def test_aggregate_orders_by_type_then_age_and_skips_done(monkeypatch):
    cx = _seed_cx()
    now = "2026-07-01T00:00:00"; later = "2026-07-02T00:00:00"
    # two reveals: one draft (older) + one sent (done); one ff draft; one handoff
    cx.execute("INSERT INTO biofield_reveals (email,scan_date,interpretation_json,"
               "remedies_json,first_approved,notified_at,created_at,updated_at) "
               "VALUES ('r@b.co','s1','{}','[]',0,NULL,?,?)", (now, now))
    cx.execute("INSERT INTO biofield_reveals (email,scan_date,interpretation_json,"
               "remedies_json,first_approved,notified_at,created_at,updated_at) "
               "VALUES ('done@b.co','s2','{}','[]',1,?,?,?)", (later, later, later))
    cx.execute("INSERT INTO ff_match_drafts (email,scan_date,items_json,status,"
               "created_at,updated_at) VALUES ('f@b.co','s3','[]','draft',?,?)", (now, now))
    cx.commit()
    # stub the handoff lister to avoid needing the client_portals schema in this unit
    monkeypatch.setattr(na, "_handoff_records",
                        lambda cx: [{"email": "h@b.co", "biofield_status": "ai_draft",
                                     "age_ts": later}])
    items = na.list_actionable(cx)
    types = [d["type"] for d in items]
    assert types == ["biofield_reveal", "handoff", "ff_match_draft"]  # TYPE_PRIORITY order
    assert "done@b.co" not in [d["summary"].split(" ")[0] for d in items]  # sent reveal skipped
    assert all(d["actionable"] for d in items)
