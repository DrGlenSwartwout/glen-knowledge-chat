import datetime
import sqlite3
import uuid
from dashboard import portal_onboarding as ob
from dashboard import (client_scans, intake, client_photos, portal_biofield_reports,
                        recommendation_events, condition_triage,
                        portal_health_history, portal_extended_history,
                        scan_freshness)


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
    steps = {st["key"]: st for st in s["phases"][0]["steps"]}
    be = {key: st["done"] for key, st in steps.items()}
    assert be == {"voice": False, "intake": False, "photo": False, "biofield": False}
    assert steps["voice"]["href"] == "https://truly.vip/E4L"
    assert s["member"] is False


def test_voice_link_opens_e4l_portal_for_existing_account():
    cx = _cx()
    client_scans.upsert_scans(
        cx, "existing@x.com",
        [{"scan_date": "2026-07-28", "scan_id": "123"}],
    )
    s = ob.build_status(cx, "existing@x.com")
    voice = next(st for st in s["phases"][0]["steps"] if st["key"] == "voice")
    assert voice["done"] is True
    assert voice["href"] == "https://portal.e4l.com"


def test_voice_link_uses_account_signal_when_scan_manifest_is_absent():
    cx = _cx()
    scan_freshness.init_table(cx)
    scan_freshness.upsert(
        cx, [{"email": "known@x.com", "last_scan_date": "2026-07-28"}]
    )
    s = ob.build_status(cx, "known@x.com")
    voice = next(st for st in s["phases"][0]["steps"] if st["key"] == "voice")
    assert voice["done"] is False
    assert voice["href"] == "https://portal.e4l.com"


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


def test_history_requires_conditions_and_current_products_sections():
    cx = _cx()
    condition_triage.init_table(cx)
    condition_triage.seed_from_triage(
        cx, "other@x.com", "other", {"other_condition": "Uveitis"})
    s = ob.build_status(cx, "other@x.com")
    match = {st["key"]: st["done"] for st in s["phases"][1]["steps"]}
    assert s["history_conditions_done"] is True
    assert s["history_products_done"] is False
    assert match["history"] is False

    portal_health_history.save(cx, "other@x.com", {
        "prescriptions_yes": False,
        "otc_yes": False,
        "supplements_yes": True,
        "supplements_text": "Brand X Product Y",
    })
    s = ob.build_status(cx, "other@x.com")
    match = {st["key"]: st["done"] for st in s["phases"][1]["steps"]}
    assert s["history_products_done"] is True
    assert s["history_extended_done"] is False
    assert match["history"] is False

    portal_extended_history.save(cx, "other@x.com", {
        "surgeries_yes": True,
        "surgeries_text": "Appendectomy; age 12",
        "family_history_yes": True,
        "family_history_text": "Maternal grandmother; diabetes",
    })
    s = ob.build_status(cx, "other@x.com")
    match = {st["key"]: st["done"] for st in s["phases"][1]["steps"]}
    assert s["history_extended_done"] is True
    assert match["history"] is True


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
