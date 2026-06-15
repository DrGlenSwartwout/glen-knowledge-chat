# tests/test_biofield_gate_routes.py
"""Routes + page for the Biofield readiness gate (magic-link auth, photo upload,
self-confirm). PHI photos must be stored OFF the static path, under DATA_DIR."""
import io
import sqlite3
from pathlib import Path

import app as appmod
from dashboard import biofield_store


def _db(monkeypatch, tmp_path):
    """Point LOG_DB + the runtime DATA_DIR (PHI photo root) at tmp_path."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BIOFIELD_CHECKOUT_ENABLED", "1")
    # init the readiness table once so the test can write directly
    cx = sqlite3.connect(db)
    biofield_store.init_table(cx)
    cx.close()
    return db


def _auth_client(email="p@x.com"):
    """A test client carrying the biofield cookie set directly (the
    _biofield_email() helper reads the plain rm_biofield_email cookie)."""
    c = appmod.app.test_client()
    c.set_cookie("rm_biofield_email", email, domain="localhost")
    return c


def test_ready_unauthenticated_401(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    r = c.get("/api/biofield/ready")
    assert r.status_code == 401


def test_ready_paid_items_needed(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    cx = sqlite3.connect(db)
    biofield_store.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    cx.close()
    c = _auth_client()
    r = c.get("/api/biofield/ready")
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["paid"] is True
    assert body["items"]["photo"]["status"] == "needed"
    assert body["items"]["intake"]["status"] == "needed"
    assert body["items"]["scan"]["status"] == "needed"
    assert body["booking_unlocked"] is False


def test_confirm_scan_flips_green(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    cx = sqlite3.connect(db)
    biofield_store.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    cx.close()
    c = _auth_client()
    r = c.post("/api/biofield/confirm", json={"item": "scan"})
    assert r.status_code == 200, r.get_data(as_text=True)
    r2 = c.get("/api/biofield/ready")
    assert r2.get_json()["items"]["scan"]["status"] == "green"


def test_photo_upload_stored_private_not_static(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    cx = sqlite3.connect(db)
    biofield_store.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    cx.close()
    c = _auth_client()
    data = {"file": (io.BytesIO(b"\x89PNG\r\n\x1a\nfakeimg"), "scan.png", "image/png")}
    r = c.post("/api/biofield/photo", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json().get("ok") is True
    # photo now green
    assert c.get("/api/biofield/ready").get_json()["items"]["photo"]["status"] == "green"
    # the stored path is under DATA_DIR (tmp_path) and NOT under STATIC
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    row = biofield_store.get(cx, "p@x.com")
    cx.close()
    stored = Path(row["photo_path"])
    assert stored.exists()
    assert str(tmp_path) in str(stored)
    assert str(appmod.STATIC) not in str(stored)


def test_confirm_payment_seeds_paid_via_pb(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    c = _auth_client("unpaid@x.com")
    # not paid yet
    assert c.get("/api/biofield/ready").get_json()["paid"] is False
    r = c.post("/api/biofield/confirm", json={"item": "payment"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert c.get("/api/biofield/ready").get_json()["paid"] is True
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    row = biofield_store.get(cx, "unpaid@x.com")
    cx.close()
    assert row["paid_via"] == "pb"


def test_intake_autodetected_from_inbound_leads(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    cx = sqlite3.connect(db)
    cx.execute(
        "CREATE TABLE IF NOT EXISTS inbound_leads ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, received_at TEXT, source TEXT, "
        "email TEXT, first_name TEXT, raw_json TEXT)")
    cx.execute(
        "INSERT INTO inbound_leads (received_at, source, email, first_name, raw_json) "
        "VALUES (?,?,?,?,?)",
        ("2026-06-14T00:00:00Z", "practice-better", "p@x.com", "Pat", "{}"))
    biofield_store.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    cx.commit()
    cx.close()
    c = _auth_client()
    assert c.get("/api/biofield/ready").get_json()["items"]["intake"]["status"] == "green"


def test_booking_unlocked_with_url(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    monkeypatch.setenv("BIOFIELD_BOOKING_URL", "https://book.example/biofield")
    cx = sqlite3.connect(db)
    biofield_store.seed_paid(cx, "p@x.com", via="stripe", order_ref="INV1")
    biofield_store.set_photo_on_file(cx, "p@x.com", str(tmp_path / "x.png"))
    biofield_store.set_intake_confirmed(cx, "p@x.com", True)
    biofield_store.set_scan_confirmed(cx, "p@x.com", True)
    cx.close()
    c = _auth_client()
    body = c.get("/api/biofield/ready").get_json()
    assert body["booking_unlocked"] is True
    assert body["booking_url"] == "https://book.example/biofield"


def test_routes_disabled_when_flag_off(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    monkeypatch.delenv("BIOFIELD_CHECKOUT_ENABLED", raising=False)
    c = _auth_client()
    r = c.get("/api/biofield/ready")
    assert r.status_code in (403, 404)


def test_ready_page_served_no_store(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    r = c.get("/biofield/ready")
    assert r.status_code == 200
    assert "no-store" in r.headers.get("Cache-Control", "")
    assert "Biofield" in r.get_data(as_text=True)
