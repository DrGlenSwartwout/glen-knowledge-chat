"""The GrooveKart webhook must record its RAW body even when it can't parse an
email (the 400 path) — so we can see the real payload and fix the field mapping.

Imports app (needs real secrets + writable DATA_DIR); runs under the Doppler harness:
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 -m pytest tests/test_webhook_debug.py
"""
import importlib, sqlite3, sys
from pathlib import Path
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import app
except Exception as _e:  # pragma: no cover
    pytest.skip(f"app import requires real secrets: {_e}", allow_module_level=True)


def _fresh(monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    # require_console_key reads dashboard.CONSOLE_SECRET (bound at import), so patch the
    # module attribute — an env var set after import isn't seen.
    import dashboard as _dash
    monkeypatch.setattr(_dash, "CONSOLE_SECRET", "testkey")
    # keep the webhook off the network for the (rare) email-present path
    monkeypatch.setattr(app, "ghl_onboard_contact", lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app, "_attribute_conversion_by_email", lambda *a, **k: None)
    monkeypatch.setattr(app, "_log_inbound_lead", lambda *a, **k: None)
    monkeypatch.setattr(app, "_ingest_order", lambda *a, **k: None)
    return db


HDRS = {"X-Console-Key": "testkey"}


def test_no_email_payload_still_records_raw_and_400s(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path)
    c = app.app.test_client()
    r = c.post("/webhook/groovekart", json={"order_id": 42, "note": "no email here"})
    assert r.status_code == 400                       # unchanged reject behavior
    # ...but the raw body is now captured for diagnosis
    dbg = c.get("/api/console/webhook-debug?source=groovekart", headers=HDRS).get_json()
    assert dbg["ok"] and len(dbg["data"]) == 1
    assert "no email here" in dbg["data"][0]["raw"]
    assert dbg["data"][0]["source"] == "groovekart"


def test_non_json_body_is_captured_too(monkeypatch, tmp_path):
    """A form-encoded / non-JSON GrooveKart body must not vanish — silent parsing +
    raw capture means we still see exactly what was POSTed."""
    _fresh(monkeypatch, tmp_path)
    c = app.app.test_client()
    r = c.post("/webhook/groovekart", data="customer_email=jdoe%40x.com&id=7",
               content_type="application/x-www-form-urlencoded")
    assert r.status_code == 400                       # our JSON handler finds no email
    dbg = c.get("/api/console/webhook-debug", headers=HDRS).get_json()
    assert dbg["ok"] and dbg["data"], "raw non-JSON body must be recorded"
    assert "customer_email=jdoe" in dbg["data"][0]["raw"]


def test_webhook_debug_requires_console_key(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path)
    c = app.app.test_client()
    assert c.get("/api/console/webhook-debug").status_code == 401
