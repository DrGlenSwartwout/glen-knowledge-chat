import sqlite3
import biofield_local_app
from dashboard import biofield_portal_publish as bpp
from dashboard.biofield_authoring import create_test, add_chain_row


def _client(tmp_path):
    db = str(tmp_path / "t.db")
    cx = sqlite3.connect(db)
    tid = create_test(cx, "Karin", "k@example.com", "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED3", most_affected="Circ",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    cx.commit(); cx.close()
    app = biofield_local_app.create_app(db_path=db)
    return app.test_client(), aid


def test_publish_route_success(tmp_path, monkeypatch):
    monkeypatch.setattr(bpp, "load_catalog", lambda: {"vitality": {"name": "Vitality"}})
    monkeypatch.setattr(bpp, "upload_asset",
                        lambda data, name, **kw: f"https://illtowell.com/portal-asset/{name}")
    monkeypatch.setattr(bpp, "publish_to_portal",
                        lambda payload, **kw: {"ok": True,
                                               "url": "https://illtowell.com/portal/xyz"})
    monkeypatch.setattr(biofield_local_app, "report_pdf_bytes", lambda html: b"%PDF-FAKE")
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")     # gate open in tests
    client, aid = _client(tmp_path)
    r = client.post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["url"] == "https://illtowell.com/portal/xyz"


def test_publish_route_409_on_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(bpp, "load_catalog", lambda: {})   # nothing resolves
    called = {"n": 0}
    monkeypatch.setattr(bpp, "publish_to_portal",
                        lambda payload, **kw: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setenv("CONSOLE_SECRET", "")
    client, aid = _client(tmp_path)
    r = client.post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 409
    assert r.get_json()["unresolved"] == ["Vitality"]
    assert called["n"] == 0      # no publish attempted


def test_publish_route_repub_no_url(tmp_path, monkeypatch):
    """Re-publish: upsert returns updated=True but no url key — route surfaces note."""
    monkeypatch.setattr(bpp, "load_catalog", lambda: {"vitality": {"name": "Vitality"}})
    monkeypatch.setattr(bpp, "upload_asset",
                        lambda data, name, **kw: f"https://illtowell.com/portal-asset/{name}")
    monkeypatch.setattr(bpp, "publish_to_portal",
                        lambda payload, **kw: {
                            "ok": True, "updated": True, "portal_id": 1,
                            "note": "existing portal updated; prior link unchanged"})
    monkeypatch.setattr(biofield_local_app, "report_pdf_bytes", lambda html: b"%PDF-FAKE")
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")
    client, aid = _client(tmp_path)
    r = client.post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["url"] == ""
    assert body["updated"] is True
    assert "existing portal updated" in body["note"]


def test_publish_route_502_on_runtime_error(tmp_path, monkeypatch):
    """publish_to_portal raises -> route returns 502 with ok=False."""
    monkeypatch.setattr(bpp, "load_catalog", lambda: {"vitality": {"name": "Vitality"}})
    monkeypatch.setattr(bpp, "upload_asset",
                        lambda data, name, **kw: f"https://illtowell.com/portal-asset/{name}")
    monkeypatch.setattr(biofield_local_app, "report_pdf_bytes", lambda html: b"%PDF-FAKE")

    def _fail(payload, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(bpp, "publish_to_portal", _fail)
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")
    client, aid = _client(tmp_path)
    r = client.post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 502
    assert r.get_json()["ok"] is False


def test_publish_route_uploads_assets_and_autosends(tmp_path, monkeypatch):
    import os as _os
    import biofield_local_app
    from dashboard import biofield_portal_publish as bpp
    from dashboard.biofield_authoring import create_test, add_chain_row
    db = str(tmp_path / "t.db")
    cx = sqlite3.connect(db)
    tid = create_test(cx, "Karin", "k@example.com", "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED3", most_affected="C",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    cx.commit(); cx.close()
    # a dummy audio file so the route finds one
    _os.makedirs(biofield_local_app.AUDIO_DIR, exist_ok=True)
    with open(_os.path.join(biofield_local_app.AUDIO_DIR, f"test_{aid}.mp3"), "wb") as f:
        f.write(b"ID3AUDIO")

    monkeypatch.setattr(bpp, "load_catalog", lambda: {"vitality": {"name": "Vitality"}})
    uploads = []
    monkeypatch.setattr(bpp, "upload_asset",
        lambda data, name, **kw: (uploads.append(name) or f"https://h/portal-asset/{name}"))
    captured = {}
    def fake_publish(payload, **kw):
        captured["content"] = payload["content"]; captured["send"] = kw.get("send")
        return {"ok": True, "url": "https://illtowell.com/portal/xyz", "emailed": True}
    monkeypatch.setattr(bpp, "publish_to_portal", fake_publish)
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")
    monkeypatch.setattr(biofield_local_app, "report_pdf_bytes", lambda html: b"%PDF-FAKE")

    app = biofield_local_app.create_app(db_path=db)
    r = app.test_client().post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 200
    assert captured["send"] is True
    assert captured["content"]["audio"]["url"].endswith(".mp3")
    assert captured["content"]["report_pdf"]["url"].endswith(".pdf")
    assert len(uploads) == 2     # pdf + mp3 both uploaded


def test_publish_route_missing_audio_still_publishes_pdf(tmp_path, monkeypatch):
    import biofield_local_app
    from dashboard import biofield_portal_publish as bpp
    from dashboard.biofield_authoring import create_test, add_chain_row
    db = str(tmp_path / "t2.db")
    cx = sqlite3.connect(db)
    tid = create_test(cx, "Karin", "k2@example.com", "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED3", most_affected="C",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    cx.commit(); cx.close()
    # ensure NO audio file for this aid
    import os as _os
    p = _os.path.join(biofield_local_app.AUDIO_DIR, f"test_{aid}.mp3")
    if _os.path.exists(p): _os.remove(p)

    monkeypatch.setattr(bpp, "load_catalog", lambda: {"vitality": {"name": "Vitality"}})
    monkeypatch.setattr(bpp, "upload_asset", lambda data, name, **kw: f"https://h/portal-asset/{name}")
    captured = {}
    monkeypatch.setattr(bpp, "publish_to_portal",
        lambda payload, **kw: (captured.update(content=payload["content"]) or {"ok": True, "url": "u"}))
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")
    monkeypatch.setattr(biofield_local_app, "report_pdf_bytes", lambda html: b"%PDF-FAKE")
    app = biofield_local_app.create_app(db_path=db)
    r = app.test_client().post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 200
    assert "report_pdf" in captured["content"]
    assert "audio" not in captured["content"]    # no mp3 -> no audio field
