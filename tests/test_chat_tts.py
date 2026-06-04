"""Tests for the /chat/tts audio-output endpoint.

The endpoint turns a short chat reply into spoken audio via ElevenLabs
(`_el_tts`), with an in-memory LRU cache and a per-IP rate limit. The
front-end falls back to the browser voice on any non-200 response, so the
contract here is: 200 + audio/mpeg on success, 400/503/429 otherwise.
"""

import pytest


@pytest.fixture
def client(monkeypatch):
    import app as app_module

    # Pretend ElevenLabs is configured so the 503 guard passes.
    monkeypatch.setattr(app_module, "_EL_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(app_module, "_EL_VOICE_ID", "test-voice", raising=False)

    # Reset cache + rate-limit state so tests don't bleed into each other.
    app_module._TTS_CACHE.clear()
    app_module._TTS_RATE.clear()

    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_tts_returns_audio_mpeg(client, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_el_tts", lambda text: (b"FAKEMP3BYTES", None))

    r = client.post("/chat/tts", json={"text": "Hello there, this is a reply."})

    assert r.status_code == 200
    assert r.mimetype == "audio/mpeg"
    assert r.data == b"FAKEMP3BYTES"


def test_tts_empty_text_is_400(client):
    r = client.post("/chat/tts", json={"text": "   "})
    assert r.status_code == 400


def test_tts_not_configured_is_503(client, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_EL_API_KEY", "", raising=False)

    r = client.post("/chat/tts", json={"text": "Hello"})
    assert r.status_code == 503


def test_tts_caches_identical_text(client, monkeypatch):
    import app as app_module
    calls = {"n": 0}

    def _counting_tts(text):
        calls["n"] += 1
        return (b"AUDIO", None)

    monkeypatch.setattr(app_module, "_el_tts", _counting_tts)

    r1 = client.post("/chat/tts", json={"text": "same reply"})
    r2 = client.post("/chat/tts", json={"text": "same reply"})

    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.data == r2.data == b"AUDIO"
    assert calls["n"] == 1  # second request served from cache


def test_tts_rate_limited_is_429(client, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_el_tts", lambda text: (b"AUDIO", None))
    monkeypatch.setattr(app_module, "_TTS_RATE_MAX", 3, raising=False)

    statuses = []
    for i in range(5):
        # distinct text each time so the cache never short-circuits the limiter
        r = client.post("/chat/tts", json={"text": f"reply number {i}"})
        statuses.append(r.status_code)

    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:]


def test_tts_upstream_error_is_502(client, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_el_tts", lambda text: (None, "boom"))

    r = client.post("/chat/tts", json={"text": "Hello"})
    assert r.status_code == 502
