import sqlite3
from dashboard import portal_onboarding as ob
from dashboard import client_scans, intake, client_photos, portal_biofield_reports


def _cx():
    cx = sqlite3.connect(":memory:")
    client_scans.init_client_scans_table(cx)
    intake.init_intake_table(cx)
    client_photos.init_table(cx)
    portal_biofield_reports.init_table(cx)
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
