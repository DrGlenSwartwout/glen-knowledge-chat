import sqlite3
from unittest import mock
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed_member(email):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _c.init_community_tables(cx)
        token = _ev.ensure_portal_token(cx, email, "")
        f = _c.upsert_full(cx, type="coaching_replay", title="Week 1", description="d",
                           video_ref="https://rumble.com/v-secret", interest_tags=["sleep"],
                           transcript="t"); _c.publish(cx, f)
        o = _c.add_outtake(cx, parent_id=f, title="teaser", video_ref="/portal-asset/x.mp4",
                           interest_tags=["sleep"]); _c.publish(cx, o)
        cx.commit()
    return token


def test_paid_member_sees_full_video_ref():
    c = _client(); tok = _seed_member("p@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        d = c.get(f"/api/community/library?token={tok}").get_json()
    assert d["tier"] == "paid"
    assert d["full"][0]["video_ref"] == "https://rumble.com/v-secret"
    assert d["full"][0]["outtakes"][0]["title"] == "teaser"


def test_free_member_never_sees_full_video_ref():
    c = _client(); tok = _seed_member("f@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=False):
        d = c.get(f"/api/community/library?token={tok}").get_json()
    assert d["tier"] == "free"
    item = d["full"][0]
    assert "video_ref" not in item                 # Rumble link withheld
    assert item["title"] == "Week 1"               # metadata still shown
    assert item["teaser_outtakes"][0]["title"] == "teaser"  # free clip shown


def test_bad_token_404():
    c = _client()
    assert c.get("/api/community/library?token=nope").status_code == 404
