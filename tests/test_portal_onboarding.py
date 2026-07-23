import datetime
import sqlite3
import uuid
from dashboard import portal_onboarding as ob
from dashboard import (client_scans, intake, client_photos, portal_biofield_reports,
                        recommendation_events)


def _cx():
    cx = sqlite3.connect(":memory:")
    client_scans.init_client_scans_table(cx)
    intake.init_intake_table(cx)
    client_photos.init_table(cx)
    portal_biofield_reports.init_table(cx)
    recommendation_events.init_recommendation_events(cx)
    return cx


def test_all_open_when_nothing_on_file():
    cx = _cx()
    s = ob.build_status(cx, "a@x.com")
    be = {st["key"]: st["done"] for st in s["phases"][0]["steps"]}
    assert be == {"voice": False, "intake": False, "photo": False, "biofield": False}
    assert s["member"] is False


def test_photo_and_intake_flip_done():
    cx = _cx()
    client_photos.put(cx, "b@x.com", b"\x89PNG", "image/png", source="portal-self")
    intake.mark_on_file(cx, "b@x.com", "2026-07-23T00:00:00Z", note="test")
    s = ob.build_status(cx, "b@x.com")
    be = {st["key"]: st["done"] for st in s["phases"][0]["steps"]}
    assert be["photo"] is True and be["intake"] is True
    assert be["voice"] is False


def test_scan_match_flips_done_on_biofield_source():
    cx = _cx()
    recommendation_events.record_event(
        cx, "c@x.com", "some-product", "biofield",
        occurred_at="2026-07-23T00:00:00Z", origin_ref="test")
    s = ob.build_status(cx, "c@x.com")
    match = {st["key"]: st["done"] for st in s["phases"][1]["steps"]}
    assert match["scan_match"] is True
    assert match["history"] is False


def test_member_true_when_membership_grant_owned():
    cx = _cx()
    cx.execute("""CREATE TABLE memberships (id TEXT PRIMARY KEY, email TEXT NOT NULL,
        granted_at TEXT NOT NULL, expires_at TEXT, granted_by TEXT, source TEXT,
        truly_vip_ref TEXT, notes TEXT, last_reminder_at TEXT)""")
    now = datetime.datetime.utcnow()
    exp = (now + datetime.timedelta(days=34)).isoformat()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,granted_by,source) "
               "VALUES (?,?,?,?,?,?)",
               (uuid.uuid4().hex, "d@x.com", now.isoformat(), exp,
                "membership_month", "membership_month"))
    cx.commit()
    s = ob.build_status(cx, "d@x.com")
    assert s["member"] is True
