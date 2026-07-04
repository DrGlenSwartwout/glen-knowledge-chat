import importlib
import sqlite3


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config["TESTING"] = True
    return appmod


def test_dispensary_checkout_stamps_method_and_pid(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: "prac-9")
    monkeypatch.setattr(appmod._pp, "practitioner_email_by_id", lambda pid: "doc@x.com")
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid, **kw: {"modules_completed": 1})
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: True)
    monkeypatch.setattr(appmod._dropship, "build_client_order",
                        lambda *a, **k: {"ok": True, "invoice_id": "INV-Z", "total": 70.0,
                                         "get_cents": 0})
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)
    c = appmod.app.test_client()
    # mirror the valid body shape from tests/test_client_routes.py::_VALID_BODY
    r = c.post("/api/client/DCODE/checkout",
               json={"email": "pat@x.com", "name": "Pat", "method": "zelle",
                     "items": [{"slug": "bone-builder", "qty": 1}],
                     "address": {"street": "1 A St", "city": "Hilo", "state": "HI",
                                 "zip": "96720", "country": "US"}})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = cx.execute("SELECT pay_method, practitioner_id FROM orders "
                         "WHERE external_ref='INV-Z' AND source='dispensary'").fetchone()
    assert row == ("zelle", "prac-9")
