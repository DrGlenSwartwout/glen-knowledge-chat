"""Audio auto-attach: making a test's audio refreshes an already-published portal
(send=False) so the new mp3 shows without a manual re-publish; not-yet-published
tests are left alone. Network + PDF are stubbed."""
import sqlite3

import pytest

import biofield_local_app as bla
from dashboard import biofield_portal_publish as bpp
from dashboard.biofield_narrative import save_video_script


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://h")
    # stub the network + PDF render used by _do_publish
    monkeypatch.setattr(bla, "report_pdf_bytes", lambda html: b"PDF")
    monkeypatch.setattr(bpp, "upload_asset", lambda data, name, **k: f"https://h/portal-asset/{name}")
    monkeypatch.setattr(bpp, "build_portal_content",
                        lambda cx, tid, **k: {"unresolved": [], "content": {"x": 1},
                                              "email": "k@x.com", "name": "K"})
    calls = []
    def fake_publish(payload, base_url=None, console_key=None, send=None):
        calls.append(send)
        return {"url": "https://h/portal/tok", "updated": True, "emailed": bool(send)}
    monkeypatch.setattr(bpp, "publish_to_portal", fake_publish)
    db = str(tmp_path / "c.db")
    c = bla.create_app(db, tts=lambda script: b"MP3BYTES",
                       scan_lookup=lambda e: {"status": "none", "found": False,
                                              "findings": [], "fresh": False}).test_client()
    c._db, c._calls = db, calls
    return c


def _seed(c):
    tid = c.post("/author/new").headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    c.post(f"/author/{tid}/row", json={"layer": 1, "head": "H", "remedy": "Vitality"})
    with sqlite3.connect(c._db) as cx:
        save_video_script(cx, tid, "This is the walkthrough script.")
    return tid


def test_audio_autoattaches_after_publish(client):
    tid = _seed(client)
    assert client.post(f"/test/{tid}/publish-portal", json={}).get_json()["ok"] is True
    client._calls.clear()
    j = client.post(f"/test/{tid}/audio", json={}).get_json()
    assert j["bytes"] == len(b"MP3BYTES")
    assert j["portal_attached"] is True
    assert client._calls == [False]          # refreshed with send=False (no re-email)


def test_audio_does_not_attach_when_not_published(client):
    tid = _seed(client)
    j = client.post(f"/test/{tid}/audio", json={}).get_json()
    assert j["portal_attached"] is False
    assert client._calls == []                # no portal call at all
