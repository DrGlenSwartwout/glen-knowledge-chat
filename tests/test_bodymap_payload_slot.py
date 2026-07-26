import importlib, sqlite3, sys
from pathlib import Path


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_payload_reports_slot_transform_for_current_system(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    bmp.put(cx, "c@x.com", "hand", "", b"H", "image/jpeg", "portal-self")
    bmp.set_transform(cx, "c@x.com", "hand", "", {"mx": 1, "my": 0, "tx": 5, "ty": 6})
    cx.commit()
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="hand")
    assert out["has_photo"] is True
    assert out["slot_transform"] == {"mx": 1.0, "my": 0.0, "tx": 5.0, "ty": 6.0}


def test_payload_no_transform_when_unaligned(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    bmp.put(cx, "c@x.com", "hand", "", b"H", "image/jpeg", "portal-self"); cx.commit()
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="hand")
    assert out["has_photo"] is True and out["slot_transform"] is None


def test_payload_face_photo_via_client_photos_fallback(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import client_photos as cph
    cx = sqlite3.connect(appmod.LOG_DB)
    cph.put(cx, "c@x.com", b"PORTRAIT", "image/jpeg", source="fmp"); cx.commit()
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="face")
    assert out["has_photo"] is True          # fallback still counts as a photo
    assert out["slot_transform"] is None      # no saved transform for the portrait


def test_payload_no_photo_no_slot(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB)
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="hand")
    assert out["has_photo"] is False and out["slot_transform"] is None


def test_payload_surfaces_latest_scan_date_and_exact_time_when_available(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import biofield_e4l as e4l
    monkeypatch.setattr(e4l, "scan_context", lambda email, today: {
        "scan_date": "2026-07-25",
        "scan_at": "2026-07-25T18:39:00-10:00",
        "findings": [],
    })
    cx = sqlite3.connect(appmod.LOG_DB)
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="organclock")
    assert out["latest_scan_date"] == "2026-07-25"
    assert out["latest_scan_at"] == "2026-07-25T18:39:00-10:00"


def test_payload_fallback_face_reports_transform_only_row(tmp_path, monkeypatch):
    """The bug: a fallback-face client (client_photos portrait, no body_map_photos
    photo bytes) aligns their face. The browser PUTs the transform, which persists
    as a transform-only body_map_photos row (image_blob NULL). The payload must
    still surface it -- has_photo True (via the portrait fallback) AND
    slot_transform == the saved transform -- so the JS applies it and skips
    re-detect. Before the fix (rec["transform"] from the blob-gated `get()`),
    slot_transform stayed None because get() returns None for a blob-less row."""
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import client_photos as cph
    from dashboard import body_map_photos as bmp
    cx = sqlite3.connect(appmod.LOG_DB)
    cph.put(cx, "c@x.com", b"PORTRAIT", "image/jpeg", source="fmp")
    bmp.set_transform(cx, "c@x.com", "face", "", {"mx": 1.3, "my": 1.1, "tx": 12, "ty": -4})
    cx.commit()
    out = appmod._portal_bodymap_data(cx, "c@x.com", {}, system="face")
    assert out["has_photo"] is True
    assert out["slot_transform"] == {"mx": 1.3, "my": 1.1, "tx": 12.0, "ty": -4.0}
