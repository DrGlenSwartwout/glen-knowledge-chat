import sqlite3
from dashboard import portal_view as pv
from dashboard import biofield_reveals as br


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
