# tests/test_community_feed_api.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(email, *, tags, n=1):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp, community_signals as _cs
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _c.init_community_tables(cx); _c.init_feed_tables(cx); _cs.init_signal_tables(cx)
        ids = []
        for i in range(n):
            cid = _c.upsert_full(cx, type="coaching_replay", title=f"T{i}", description="",
                                 video_ref=f"https://rumble.com/v-{i}",
                                 interest_tags=tags[i] if isinstance(tags[0], list) else tags,
                                 transcript="body"); _c.publish(cx, cid); ids.append(cid)
        token = _ev.ensure_portal_token(cx, email, "")
        cx.commit()
    return token, ids


def test_feed_bad_token_404():
    assert _client().get("/api/community/feed?token=nope").status_code == 404


def test_feed_paid_returns_items_with_reason():
    c = _client(); tok, ids = _seed("p@x.com", tags=["sleep"], n=2)
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "embed", return_value=[0.1, 0.2, 0.3]):
        d = c.get(f"/api/community/feed?token={tok}").get_json()
    assert "items" in d and len(d["items"]) >= 1
    assert all("reason" in it for it in d["items"])
    assert all("score" not in it for it in d["items"])


def test_feed_free_no_full_video_ref_and_capped():
    c = _client()
    tok, ids = _seed("f@x.com", tags=[["a"], ["b"], ["c"], ["d"]], n=4)
    with mock.patch.object(appmod, "_is_paid_member", return_value=False), \
         mock.patch.object(appmod, "embed", return_value=[0.1, 0.2, 0.3]):
        d = c.get(f"/api/community/feed?token={tok}").get_json()
    assert len(d["items"]) <= 3                       # FREE_K cap
    assert all("video_ref" not in it for it in d["items"])  # no Rumble link for free


def test_feed_cold_start_when_embed_unavailable():
    c = _client(); tok, ids = _seed("c@x.com", tags=["sleep"], n=1)
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "embed", side_effect=RuntimeError("no key")):
        d = c.get(f"/api/community/feed?token={tok}").get_json()
    assert d["cold_start"] is True
    assert len(d["items"]) >= 1                        # still not empty


def test_feed_uses_likes_not_cold_start_when_liked():
    from dashboard import community_signals as _cs
    c = _client(); tok, ids = _seed("liker@x.com", tags=["sleep"], n=1)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cs.init_signal_tables(cx)
        _cs.set_signal(cx, "liker@x.com", "topic", "sleep", "like")
        cx.commit()
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "embed", return_value=[0.1, 0.2, 0.3]):
        d = c.get(f"/api/community/feed?token={tok}").get_json()
    assert d["cold_start"] is False


def test_feed_ignores_glen_journal():
    from dashboard import journal_store
    c = _client(); tok, ids = _seed("nolikes@x.com", tags=["sleep"], n=1)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        journal_store.init_table(cx)
        journal_store.insert(cx, {"user_id": "glen", "recorded_at": "2026-07-01T00:00:00Z",
                                   "transcript": "sleep adrenals"})
        cx.commit()
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "embed", return_value=[0.1, 0.2, 0.3]):
        d = c.get(f"/api/community/feed?token={tok}").get_json()
    assert d["cold_start"] is True
