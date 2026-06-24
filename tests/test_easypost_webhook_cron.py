import hashlib
import hmac
import json
import sqlite3
from datetime import datetime, timedelta
import app as appmod
from dashboard import coaching, tracking as T


def _reset_and_seed_delivered_ready(email="wh@x.com", tn="WHTN1", src="reorder"):
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    coaching.init_coaching_table(cx); T.init_tracking_schema(cx); T.migrate_add_delivery_columns(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS memberships (id TEXT PRIMARY KEY, email TEXT, granted_at TEXT, "
               "expires_at TEXT, granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT, last_reminder_at TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, "
               "source TEXT, external_ref TEXT, email TEXT, status TEXT DEFAULT 'new', shipment_id INTEGER)")
    for t in ("coaching_windows", "shipments", "memberships", "orders"):
        cx.execute(f"DELETE FROM {t} WHERE 1=1")
    iso = lambda d: (datetime.utcnow() + timedelta(days=d)).isoformat() + "Z"
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("m", email, iso(-30), iso(20), "membership"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (iso(-5), src, "U-" + tn, email))
    T.record_shipment(cx, tracking_number=tn, order_uuid="U-" + tn, status="sent")
    cx.commit(); cx.close()


def _evt(tn):
    return json.dumps({"description": "tracker.updated",
                       "result": {"status": "delivered", "tracking_code": tn}}).encode()


def test_webhook_inert_when_secret_unset(monkeypatch):
    monkeypatch.delenv("EASYPOST_WEBHOOK_SECRET", raising=False)
    r = appmod.app.test_client().post("/webhook/easypost", data=_evt("X"),
                                      content_type="application/json")
    assert r.status_code == 200


def test_webhook_bad_signature_400(monkeypatch):
    monkeypatch.setenv("EASYPOST_WEBHOOK_SECRET", "s")
    r = appmod.app.test_client().post("/webhook/easypost", data=_evt("X"),
                                      headers={"X-Hmac-Signature": "hmac-sha256-hex=bad"},
                                      content_type="application/json")
    assert r.status_code == 400


def test_webhook_delivered_opens_window(monkeypatch):
    monkeypatch.setenv("EASYPOST_WEBHOOK_SECRET", "s")
    _reset_and_seed_delivered_ready(email="wh2@x.com", tn="WHTN2")
    body = _evt("WHTN2")
    sig = hmac.new(b"s", body, hashlib.sha256).hexdigest()
    r = appmod.app.test_client().post("/webhook/easypost", data=body,
                                      headers={"X-Hmac-Signature": "hmac-sha256-hex=" + sig},
                                      content_type="application/json")
    assert r.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    assert coaching.active_window(cx, "wh2@x.com") is not None
    assert coaching.active_window(cx, "wh2@x.com")["source"] == "delivery"
    cx.close()


def test_cron_unconfigured_skips(monkeypatch):
    monkeypatch.delenv("EASYPOST_API_KEY", raising=False)
    key = appmod.os.environ.get("CRON_SECRET") or appmod.os.environ.get("CONSOLE_SECRET", "")
    r = appmod.app.test_client().post("/api/cron/easypost-sync", headers={"X-Cron-Secret": key})
    assert r.status_code == 200
    assert json.loads(r.get_data(as_text=True)).get("skipped") == "easypost_unconfigured"


def test_cron_registers_and_activates(monkeypatch):
    monkeypatch.setenv("EASYPOST_API_KEY", "ezk_test")
    _reset_and_seed_delivered_ready(email="wh3@x.com", tn="WHTN3")
    from dashboard import easypost as EP
    monkeypatch.setattr(EP, "create_tracker", lambda tc, carrier=None: {"tracker_id": "trk_" + tc, "status": "delivered"})
    monkeypatch.setattr(EP, "get_tracker", lambda tid: {"tracker_id": tid, "status": "delivered", "tracking_code": "WHTN3"})
    key = appmod.os.environ.get("CRON_SECRET") or appmod.os.environ.get("CONSOLE_SECRET", "")
    r = appmod.app.test_client().post("/api/cron/easypost-sync", headers={"X-Cron-Secret": key})
    assert r.status_code == 200
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    assert coaching.active_window(cx, "wh3@x.com") is not None
    cx.close()
