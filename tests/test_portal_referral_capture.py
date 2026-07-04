import importlib
import sqlite3
from dashboard import referrals as rf


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_capture_writes_dispensary_portal_row(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    appmod._capture_portal_referral("DISPCODE", "Patient@X.com", "doc@x.com", "INV-9")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = rf.redemption_by_order_ref(cx, "INV-9")
    assert row["referee_email"] == "patient@x.com"   # normalized
    assert row["owner_email"] == "doc@x.com"
    assert row["kind"] == "dispensary_portal"


def test_capture_is_first_touch(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "AMB", "ambassador@x.com", "patient@x.com", "INV-A")
    appmod._capture_portal_referral("DISPCODE", "patient@x.com", "doc@x.com", "INV-9")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rf.owner_of_referee(cx, "patient@x.com") == "ambassador@x.com"  # unchanged


def test_capture_noops_without_practitioner_email(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    appmod._capture_portal_referral("DISPCODE", "patient@x.com", "", "INV-9")  # unresolved pid
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rf.redemption_by_order_ref(cx, "INV-9") is None


def test_client_checkout_records_referral(monkeypatch, tmp_path):
    """The route resolves pid->email and captures the referral after ingest."""
    appmod = _reload(monkeypatch, tmp_path)
    appmod.app.config["TESTING"] = True
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: "p1")
    monkeypatch.setattr(appmod._pp, "practitioner_email_by_id", lambda pid: "doc@x.com")
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid, **kw: {"modules_completed": 1})
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: True)
    monkeypatch.setattr(appmod._dropship, "build_client_order",
                        lambda *a, **k: {"ok": True, "invoice_id": "INV-77", "total": 70.0,
                                         "get_cents": 0})
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    c = appmod.app.test_client()
    r = c.post("/api/client/DISPCODE/checkout",
               json={"email": "patient@x.com", "name": "Pat", "method": "zelle",
                     "items": [{"slug": "bone-builder", "qty": 1}],
                     "address": {"street": "1 A St", "city": "Hilo", "state": "HI",
                                 "zip": "96720", "country": "US"}})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = rf.redemption_by_order_ref(cx, "INV-77")
    assert row and row["owner_email"] == "doc@x.com" and row["kind"] == "dispensary_portal"
