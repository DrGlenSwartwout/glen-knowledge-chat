# tests/test_scan_freshness_routes.py
"""Routes for the e4l scan-freshness ingest endpoint + the Biofield gate
auto-verifying a fresh voice scan from that server-side index."""
import datetime as _dt
import sqlite3

import app as appmod
from dashboard import biofield_store
from dashboard import scan_freshness as sf

TEST_EMAIL = "p@x.com"


def _db(monkeypatch, tmp_path):
    """Point LOG_DB + DATA_DIR at tmp_path and enable the biofield gate."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BIOFIELD_CHECKOUT_ENABLED", "1")
    # Clean any pre-existing row for the shared test email so the index is
    # deterministic regardless of LOG_DB state.
    cx = sqlite3.connect(db)
    biofield_store.init_table(cx)
    sf.init_table(cx)
    cx.execute("DELETE FROM scan_freshness WHERE email=lower(?)", (TEST_EMAIL,))
    cx.commit()
    cx.close()
    return db


def _cron_secret():
    import os
    return os.environ.get("CRON_SECRET") or appmod.CONSOLE_SECRET


def _auth_client(email=TEST_EMAIL):
    c = appmod.app.test_client()
    c.set_cookie("rm_biofield_email", email, domain="localhost")
    return c


def _two_days_ago_iso():
    return (_dt.date.today() - _dt.timedelta(days=2)).isoformat()


# ── Ingest endpoint ──────────────────────────────────────────────────────────

def test_ingest_requires_cron_secret(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    body = {"rows": [{"email": TEST_EMAIL, "last_scan_date": _two_days_ago_iso()}]}
    # No secret
    assert c.post("/api/e4l/scan-freshness", json=body).status_code == 401
    # Wrong secret
    r = c.post("/api/e4l/scan-freshness", json=body,
               headers={"X-Cron-Secret": "nope-wrong"})
    assert r.status_code == 401


def test_ingest_upserts_with_secret(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    d = _two_days_ago_iso()
    body = {"rows": [{"email": TEST_EMAIL, "last_scan_date": d}]}
    r = c.post("/api/e4l/scan-freshness", json=body,
               headers={"X-Cron-Secret": _cron_secret()})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"] is True
    assert j["upserted"] == 1
    # The same tmp LOG_DB reflects the upsert
    cx = sqlite3.connect(db)
    try:
        assert sf.latest_scan_date(cx, TEST_EMAIL) == d
    finally:
        cx.close()


# ── Gate auto-verify (fresh scan greens the scan item, no self-confirm) ──────

def test_gate_auto_verifies_fresh_scan(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    # Seed paid + photo + an inbound_leads intake row so payment/photo/intake
    # are already green — scan is the only item under test.
    cx = sqlite3.connect(db)
    biofield_store.seed_paid(cx, TEST_EMAIL, via="stripe", order_ref="INV1")
    biofield_store.set_photo_on_file(cx, TEST_EMAIL, str(tmp_path / "x.png"))
    cx.execute(
        "CREATE TABLE IF NOT EXISTS inbound_leads ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, received_at TEXT, source TEXT, "
        "email TEXT, first_name TEXT, raw_json TEXT)")
    cx.execute(
        "INSERT INTO inbound_leads (received_at, source, email, first_name, raw_json) "
        "VALUES (?,?,?,?,?)",
        ("2026-06-14T00:00:00Z", "practice-better", TEST_EMAIL, "Pat", "{}"))
    cx.commit()
    cx.close()

    c = _auth_client()
    # Scan NOT self-confirmed yet → scan item still needed.
    body = c.get("/api/biofield/ready").get_json()
    assert body["items"]["scan"]["status"] == "needed"
    assert body["booking_unlocked"] is False

    # Push a fresh scan via the ingest endpoint (no self-confirm call).
    tc = appmod.app.test_client()
    r = tc.post("/api/e4l/scan-freshness",
                json={"rows": [{"email": TEST_EMAIL,
                                "last_scan_date": _two_days_ago_iso()}]},
                headers={"X-Cron-Secret": _cron_secret()})
    assert r.status_code == 200, r.get_data(as_text=True)

    # Gate now auto-greens scan and unlocks booking.
    body = c.get("/api/biofield/ready").get_json()
    assert body["items"]["scan"]["status"] == "green"
    assert body["booking_unlocked"] is True
