# tests/test_founding_checkout_return.py
import sqlite3
import app as appmod
from dashboard import subscriptions as subs


def test_founding_return_creates_reservation_and_comp_membership(monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    # Seed schema
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx); subs.migrate_add_founding_columns(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS memberships (id TEXT, email TEXT, granted_at TEXT,"
               " expires_at TEXT, granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT)")
    cx.commit(); cx.close()

    from dashboard import stripe_pay as sp, founding as founding_mod
    monkeypatch.setattr(sp, "get_session", lambda s: {
        "payment_status": "no_payment_required", "setup_intent": "seti_1",
        "customer": "cus_1",
        "metadata": {"kind": "founding_reserve", "slug": "neuro-magnesium", "email": "f@x.com",
                     "items": '[{"slug":"neuro-magnesium","qty":1}]', "ship": '{"state":"HI"}'}})
    monkeypatch.setattr(sp, "get_setup_intent", lambda i: {"customer": "cus_1", "payment_method": "pm_1"})
    monkeypatch.setattr(founding_mod, "is_open", lambda cx, slug, **kw: True)
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)

    c = appmod.app.test_client()
    r = c.get("/begin/checkout-return?session_id=cs_1")
    assert r.status_code in (200, 302)

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    sub = cx.execute("SELECT * FROM subscriptions WHERE email='f@x.com'").fetchone()
    assert sub is not None and sub["founding"] == 1 and sub["founding_state"] == "pending"
    mem = cx.execute("SELECT * FROM memberships WHERE email='f@x.com' AND source='founding'").fetchone()
    assert mem is not None
    cx.close()
