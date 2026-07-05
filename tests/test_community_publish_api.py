import sqlite3
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _payload(video_ref="https://rumble.com/v-1"):
    return {"type": "coaching_replay", "title": "Week 1", "description": "d",
            "video_ref": video_ref, "interest_tags": ["sleep"], "transcript": "t",
            "outtakes": [{"title": "clip A", "video_ref": "/portal-asset/a.mp4",
                          "interest_tags": ["sleep"]},
                         {"title": "clip B", "video_ref": "/portal-asset/b.mp4",
                          "interest_tags": []}]}


def test_publish_requires_console_key():
    c = _client()
    r = c.post("/api/console/community/publish", json=_payload())
    assert r.status_code == 401


def test_publish_creates_full_and_outtakes():
    c = _client()
    r = c.post("/api/console/community/publish", json=_payload(),
               headers={"X-Console-Key": appmod.CONSOLE_SECRET})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["outtakes"] == 2
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _c.init_community_tables(cx)
        full = _c.list_full(cx)
        mine = [f for f in full if f["video_ref"] == "https://rumble.com/v-1"][0]
        assert mine["title"] == "Week 1"
        assert sorted(o["title"] for o in mine["outtakes"]) == ["clip A", "clip B"]


def test_publish_is_idempotent():
    c = _client()
    h = {"X-Console-Key": appmod.CONSOLE_SECRET}
    c.post("/api/console/community/publish", json=_payload("https://rumble.com/v-dup"), headers=h)
    c.post("/api/console/community/publish", json=_payload("https://rumble.com/v-dup"), headers=h)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _c.init_community_tables(cx)
        dup = [f for f in _c.list_full(cx) if f["video_ref"] == "https://rumble.com/v-dup"]
        assert len(dup) == 1                 # one full row, not two
        assert len(dup[0]["outtakes"]) == 2  # out-takes replaced, not doubled


def test_portal_asset_accepts_mp4_and_still_mp3():
    c = _client()
    h = {"X-Console-Key": appmod.CONSOLE_SECRET}
    r4 = c.put("/portal-asset/upload?filename=outtake-0.mp4", data=b"\x00\x01", headers=h)
    assert r4.status_code == 200 and r4.get_json()["url"].endswith("outtake-0.mp4")
    g4 = c.get("/portal-asset/outtake-0.mp4")
    assert g4.status_code == 200 and g4.mimetype == "video/mp4"
    # regression: mp3 still accepted
    r3 = c.put("/portal-asset/upload?filename=snippet.mp3", data=b"\x00", headers=h)
    assert r3.status_code == 200
