import sqlite3
from dashboard import coach_directory as _cd


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _cd.init_coach_tables(cx)
    return cx


def test_upsert_and_get():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="C@X.com", name="Cora", focus="sleep",
                         intro_video_url="https://rumble.com/v-c", capacity=3, cert_ok=1)
    row = _cd.get_volunteer(cx, "c@x.com")
    assert row["name"] == "Cora" and row["cert_ok"] == 1 and row["active"] == 1


def test_upsert_is_idempotent_on_email():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora", focus="a",
                         intro_video_url="u1", capacity=3, cert_ok=1)
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora B", focus="sleep",
                         intro_video_url="u2", capacity=5, cert_ok=1)
    assert cx.execute("SELECT COUNT(*) FROM coach_volunteers").fetchone()[0] == 1
    row = _cd.get_volunteer(cx, "c@x.com")
    assert row["focus"] == "sleep" and row["intro_video_url"] == "u2" and row["capacity"] == 5


def test_list_active_member_safe_no_email():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora", focus="sleep",
                         intro_video_url="u", capacity=3, cert_ok=1)
    lst = _cd.list_active(cx)
    assert lst == [{"name": "Cora", "focus": "sleep", "intro_video_url": "u"}]
    assert "email" not in lst[0] and "capacity" not in lst[0]


def test_list_active_excludes_inactive_and_uncertified():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="ok@x.com", name="Ok", focus="f", intro_video_url="u",
                         capacity=3, cert_ok=1)
    _cd.upsert_volunteer(cx, email="nocert@x.com", name="NoCert", focus="f",
                         intro_video_url="u", capacity=3, cert_ok=0)   # not certified
    _cd.upsert_volunteer(cx, email="off@x.com", name="Off", focus="f", intro_video_url="u",
                         capacity=3, cert_ok=1)
    _cd.set_active(cx, "off@x.com", 0)                                  # deactivated
    names = [c["name"] for c in _cd.list_active(cx)]
    assert names == ["Ok"]


def test_set_active_toggles():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora", focus="f", intro_video_url="u",
                         capacity=3, cert_ok=1)
    _cd.set_active(cx, "c@x.com", 0)
    assert _cd.get_volunteer(cx, "c@x.com")["active"] == 0
    assert _cd.list_active(cx) == []
