# tests/test_biofield_photo_routes.py
"""Console-gated PHI photo viewer (/admin/biofield/photo) + retention purge cron
(/api/cron/biofield-photo-purge). The viewer must never serve a file that
resolves outside the private biofield-photos dir."""
import os
import sqlite3
import time
from pathlib import Path

import app as appmod
from dashboard import biofield_store


def _db(monkeypatch, tmp_path):
    """Point LOG_DB + the runtime DATA_DIR (PHI photo root) at tmp_path."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BIOFIELD_CHECKOUT_ENABLED", "1")
    cx = sqlite3.connect(db)
    biofield_store.init_table(cx)
    cx.close()
    return db


def _photo_dir(tmp_path):
    d = Path(tmp_path) / "biofield-photos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed_photo(db, tmp_path, email, blob=b"\x89PNG\r\n\x1a\nFAKE", ext="png"):
    """Write a real file under the photo dir + record it in the DB. Returns path."""
    d = _photo_dir(tmp_path)
    p = d / f"seed-{email.replace('@', '_').replace('.', '_')}.{ext}"
    p.write_bytes(blob)
    cx = sqlite3.connect(db)
    biofield_store.init_table(cx)
    biofield_store.set_photo_on_file(cx, email, str(p))
    cx.close()
    return p


# ── viewer auth ────────────────────────────────────────────────────────────

def test_viewer_no_key_401_when_secret_set(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    # Only meaningful if a console secret is configured; otherwise auth is a no-op.
    if not getattr(appmod, "CONSOLE_SECRET", ""):
        return
    c = appmod.app.test_client()
    r = c.get("/admin/biofield/photo?email=p@x.com")
    assert r.status_code == 401


def _key_headers():
    sec = getattr(appmod, "CONSOLE_SECRET", "")
    return {"X-Console-Key": sec} if sec else {}


def test_viewer_no_photo_404(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    r = c.get("/admin/biofield/photo?email=nobody-photo@x.com",
              headers=_key_headers())
    assert r.status_code == 404


def test_viewer_serves_seeded_photo_200(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    blob = b"\x89PNG\r\n\x1a\nHELLO-BIOFIELD"
    _seed_photo(db, tmp_path, "view@x.com", blob=blob, ext="png")
    c = appmod.app.test_client()
    r = c.get("/admin/biofield/photo?email=view@x.com", headers=_key_headers())
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.headers.get("Content-Type", "").startswith("image/")
    assert r.get_data() == blob


def test_viewer_path_guard_blocks_outside_file(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    cx = sqlite3.connect(db)
    biofield_store.init_table(cx)
    # Point at a real file OUTSIDE the photo dir — must never be served.
    biofield_store.set_photo_on_file(cx, "evil@x.com", "/etc/passwd")
    cx.close()
    c = appmod.app.test_client()
    r = c.get("/admin/biofield/photo?email=evil@x.com", headers=_key_headers())
    assert r.status_code == 404


# ── on-request delete (console-gated; the ONLY deletion path — no auto purge) ──

def test_photo_delete_no_key_401_when_secret_set(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    if not getattr(appmod, "CONSOLE_SECRET", ""):
        return
    c = appmod.app.test_client()
    r = c.post("/admin/biofield/photo/delete?email=p@x.com")
    assert r.status_code == 401


def test_photo_delete_removes_file_and_flag(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    p = _seed_photo(db, tmp_path, "del@x.com")
    assert p.exists()
    c = appmod.app.test_client()
    r = c.post("/admin/biofield/photo/delete?email=del@x.com", headers=_key_headers())
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True and body["deleted_file"] is True
    assert not p.exists()                       # file actually removed
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    row = biofield_store.get(cx, "del@x.com"); cx.close()
    assert not row["photo_on_file"] and not row["photo_path"]   # flag + path cleared


def test_photo_delete_no_photo_still_ok(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    r = c.post("/admin/biofield/photo/delete?email=none@x.com", headers=_key_headers())
    assert r.status_code == 200
    assert r.get_json()["deleted_file"] is False
