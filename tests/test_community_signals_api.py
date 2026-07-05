# tests/test_community_signals_api.py
import sqlite3
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(email="m@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp, community_signals as _s
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _c.init_community_tables(cx); _s.init_signal_tables(cx)
        cid = _c.upsert_full(cx, type="coaching_replay", title="W1", description="",
                             video_ref="https://rumble.com/v-b", interest_tags=["sleep"],
                             transcript=""); _c.publish(cx, cid)
        token = _ev.ensure_portal_token(cx, email, "")
        cx.commit()
    return token, cid


def test_react_toggle_and_counts():
    c = _client(); tok, cid = _seed()
    r1 = c.post(f"/api/community/react?token={tok}", json={"content_id": cid, "reaction": "helpful"})
    d1 = r1.get_json()
    assert d1["ok"] and d1["on"] is True and d1["counts"]["helpful"] == 1
    r2 = c.post(f"/api/community/react?token={tok}", json={"content_id": cid, "reaction": "helpful"})
    assert r2.get_json()["on"] is False  # toggled off


def test_react_bad_reaction_400():
    c = _client(); tok, cid = _seed()
    r = c.post(f"/api/community/react?token={tok}", json={"content_id": cid, "reaction": "nope"})
    assert r.status_code == 400


def test_react_unknown_content_404():
    c = _client(); tok, _ = _seed()
    r = c.post(f"/api/community/react?token={tok}", json={"content_id": 999999, "reaction": "helpful"})
    assert r.status_code == 404


def test_reactions_get_aggregate_no_identity():
    c = _client(); tok, cid = _seed()
    c.post(f"/api/community/react?token={tok}", json={"content_id": cid, "reaction": "inspiring"})
    d = c.get(f"/api/community/reactions?token={tok}&content_id={cid}").get_json()
    assert d["counts"]["inspiring"] == 1
    assert d["mine"] == ["inspiring"]
    # no identity fields anywhere in the payload
    assert "email" not in repr(d)


def test_signal_set_clear_and_scope():
    c = _client(); tok, _ = _seed()
    c.post(f"/api/community/signal?token={tok}",
           json={"target_type": "topic", "target_key": "sleep", "signal": "like"})
    d = c.get(f"/api/community/signals?token={tok}").get_json()
    assert d["likes"] == [{"target_type": "topic", "target_key": "sleep"}]
    c.post(f"/api/community/signal?token={tok}",
           json={"target_type": "topic", "target_key": "sleep", "signal": "none"})
    assert c.get(f"/api/community/signals?token={tok}").get_json()["likes"] == []


def test_signal_bad_type_400():
    c = _client(); tok, _ = _seed()
    r = c.post(f"/api/community/signal?token={tok}",
               json={"target_type": "planet", "target_key": "x", "signal": "like"})
    assert r.status_code == 400


def test_bad_token_404():
    c = _client()
    assert c.get("/api/community/signals?token=nope").status_code == 404


def test_bad_token_wins_over_bad_body():
    c = _client()
    r = c.post("/api/community/react?token=nope",
               json={"content_id": 1, "reaction": "nope"})
    assert r.status_code == 404


def test_signals_scoped_across_members():
    c = _client(); tok_a, _ = _seed(email="member-a@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev
        tok_b = _ev.ensure_portal_token(cx, "member-b@x.com", "")
        cx.commit()
    c.post(f"/api/community/signal?token={tok_a}",
           json={"target_type": "topic", "target_key": "sleep", "signal": "like"})
    d_a = c.get(f"/api/community/signals?token={tok_a}").get_json()
    assert d_a["likes"] == [{"target_type": "topic", "target_key": "sleep"}]
    d_b = c.get(f"/api/community/signals?token={tok_b}").get_json()
    assert d_b["likes"] == []
