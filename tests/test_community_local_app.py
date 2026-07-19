from unittest import mock
import community_local_app as cla


def _client():
    app = cla.create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_analyze_returns_transcript_and_suggestions():
    c = _client()
    with mock.patch.object(cla, "transcribe",
                           return_value={"text": "hello", "segments": []}), \
         mock.patch.object(cla, "suggest_catalog",
                           return_value={"title": "T", "interest_tags": ["a"],
                                         "outtakes": [{"start": 1, "end": 3,
                                                       "title": "clip", "reason": "r"}]}):
        r = c.post("/analyze", json={"path": "/tmp/x.mp4",
                                     "rumble_url": "https://rumble.com/v-1",
                                     "type": "coaching_replay"})
    d = r.get_json()
    assert d["suggestions"]["title"] == "T"
    assert d["transcript"] == "hello"


def test_publish_cuts_and_publishes(monkeypatch):
    # /publish reads os.environ["CONSOLE_SECRET"] (bracket access -> KeyError -> 500 when
    # unset). Set it here rather than relying on an ambient value: this passed locally only
    # because a dev shell, or leakage from test_biofield_local_app, happened to define it,
    # and failed on a clean CI runner. monkeypatch restores it automatically, so it can't
    # leak into other tests.
    monkeypatch.setenv("CONSOLE_SECRET", "test-console-secret")
    c = _client()
    with mock.patch.object(cla, "cut_outtakes",
                           return_value=[{"title": "clip", "interest_tags": [], "path": "/tmp/o0.mp4"}]) as cut, \
         mock.patch.object(cla, "publish_session",
                           return_value={"ok": True, "content_id": 9, "outtakes": 1}) as pub:
        r = c.post("/publish", json={
            "path": "/tmp/x.mp4",
            "full": {"type": "coaching_replay", "title": "T", "description": "d",
                     "video_ref": "https://rumble.com/v-1", "interest_tags": ["a"],
                     "transcript": "hello"},
            "outtakes": [{"start": 1, "end": 3, "title": "clip", "interest_tags": []}]})
    assert r.get_json()["content_id"] == 9
    assert cut.called and pub.called
