# tests/test_community_chat_related.py
import sqlite3
from unittest import mock
import app as appmod
from dashboard import community as _c


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed_member_and_content(email):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _c.init_community_tables(cx); _c.init_feed_tables(cx)
        cid = _c.upsert_full(cx, type="coaching_replay", title="Sleep Reset", description="",
                             video_ref="https://rumble.com/v-s", interest_tags=["sleep"],
                             transcript=""); _c.publish(cx, cid)
        _c.set_embedding(cx, cid, [1.0, 0.0, 0.0], appmod.COMMUNITY_FEED_MODEL)
        token = _ev.ensure_portal_token(cx, email, "You")
        cx.commit()
    return token, cid


def _sse_events(resp):
    text = resp.get_data(as_text=True)
    import json
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            try: out.append(json.loads(line[6:]))
            except Exception: pass
    return out


def test_chat_emits_related_event(monkeypatch):
    c = _client(); tok, cid = _seed_member_and_content("m@x.com")
    # deterministic query embedding near the seeded item; stub the LLM stream + KB RAG
    monkeypatch.setattr(appmod, "embed", lambda t: [1.0, 0.0, 0.0])
    monkeypatch.setattr(appmod, "_match_query_namespaces", lambda v: [])
    with mock.patch.object(appmod, "_is_paid_member", return_value=True), \
         mock.patch.object(appmod, "_stream_claude_tokens", create=True):
        # If the route streams via anthropic directly, this test asserts on the related
        # event regardless of tokens; see Step 3 note on stubbing the LLM.
        resp = c.post(f"/api/portal/{tok}/chat", json={"query": "help me sleep", "history": []})
    evts = _sse_events(resp)
    related = [e for e in evts if "related" in e]
    assert related and related[0]["related"][0]["title"] == "Sleep Reset"
    assert related[0]["related"][0]["kind"] == "full"
    assert "video_ref" not in related[0]["related"][0]


def test_chat_related_failure_does_not_break(monkeypatch):
    c = _client(); tok, cid = _seed_member_and_content("m2@x.com")
    monkeypatch.setattr(appmod, "embed", lambda t: [1.0, 0.0, 0.0])
    monkeypatch.setattr(appmod, "_match_query_namespaces", lambda v: [])
    monkeypatch.setattr(appmod, "_community_related",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        resp = c.post(f"/api/portal/{tok}/chat", json={"query": "hi", "history": []})
    assert resp.status_code == 200   # stream still returned, no crash
