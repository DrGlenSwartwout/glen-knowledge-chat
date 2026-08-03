import sqlite3
from dashboard import portal_view as pv
from dashboard import biofield_reveals as br
from dashboard import portal_biofield_reports as reports


def _db():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    br.init_table(cx); return cx


def test_reveal_shows_blurred_when_unpaid():
    cx = _db()
    br.upsert(cx, "c@x.com", "2026-07-18", {"greeting": "Aloha"},
              [{"name": "Calm"}], "t",
              layers=[{"n": 1, "title": "T", "meaning": "m", "remedy": {"name": "Calm"}}])
    blk = pv._biofield_block(cx, "c@x.com", unlocked=False)
    assert blk["visible"] is True
    assert blk["blurred"] is True
    assert blk["scan_dates"] == ["2026-07-18"]
    assert "remedy" not in blk["layers"][0]          # remedy never leaves server when blurred


def test_reveal_shows_remedy_when_paid():
    cx = _db()
    br.upsert(cx, "c@x.com", "2026-07-18", {"greeting": "Aloha"},
              [{"name": "Calm"}], "t",
              layers=[{"n": 1, "title": "T", "meaning": "m", "remedy": {"name": "Calm"}}])
    blk = pv._biofield_block(cx, "c@x.com", unlocked=True)
    assert blk["blurred"] is False
    assert blk["layers"][0]["remedy"] == "Calm"


def test_no_biofield_data_is_not_visible():
    assert pv._biofield_block(_db(), "nobody@x.com", unlocked=True) == {"visible": False}


def test_newer_reveal_is_selectable_beside_older_authored_report():
    cx = _db()
    reports.init_table(cx)
    reports.upsert_report(
        cx, "c@x.com", "2026-06-20", "old",
        {"greeting": "June", "layers": [{"n": 1, "title": "Old", "meaning": "old"}]},
        "ai_draft")
    br.upsert(
        cx, "c@x.com", "2026-08-02", {"greeting": "August"}, [], "e4l",
        layers=[{"n": 1, "title": "New", "summary": "new", "remedy": None}])

    default = pv._biofield_block(cx, "c@x.com", unlocked=True)
    selected = pv._biofield_block(cx, "c@x.com", scan_date="2026-08-02", unlocked=True)

    assert default["scan_date"] == "2026-06-20"
    assert default["scan_dates"] == ["2026-08-02", "2026-06-20"]
    assert selected["scan_date"] == "2026-08-02"
    assert selected["greeting"] == "August"
    assert selected["layers"][0]["title"] == "New"
