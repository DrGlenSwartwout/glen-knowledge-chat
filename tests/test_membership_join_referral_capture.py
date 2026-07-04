import sqlite3
import app as appmod
from dashboard import subscriptions as subs, referrals as rf

OWNER = "join-owner@example.com"
JOINER = "join-referee@example.com"


def _clean(cx):
    rf.init_tables(cx)
    for e in (OWNER, JOINER):
        cx.execute("DELETE FROM subscriptions WHERE email=?", (e,))
    cx.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (JOINER,))
    cx.execute("DELETE FROM referral_codes WHERE email=?", (OWNER,))
    cx.commit()


def test_membership_return_records_referral_from_metadata(monkeypatch):
    monkeypatch.setattr(appmod, "_portal_offers_enabled", lambda: True)
    monkeypatch.setattr(appmod, "_REFERRALS", True)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _clean(cx)
    code = rf.get_or_create_code(cx, OWNER)

    monkeypatch.setattr(appmod.stripe_pay, "get_session",
                        lambda sid: {"metadata": {"kind": "group_join", "email": JOINER,
                                                  "referral_code": code},
                                     "setup_intent": "seti_1"})
    monkeypatch.setattr(appmod.stripe_pay, "get_setup_intent",
                        lambda si: {"customer": "cus_j", "payment_method": "pm_j"})
    monkeypatch.setattr(appmod, "_member_join_welcome", lambda *a, **k: None)

    appmod.app.test_client().get("/portal/offer/live-group/return?session_id=cs_test")

    cx2 = sqlite3.connect(appmod.LOG_DB); cx2.row_factory = sqlite3.Row
    row = cx2.execute("SELECT owner_email FROM referral_redemptions WHERE referee_email=?",
                      (JOINER,)).fetchone()
    assert row is not None and row[0] == OWNER
    for e in (OWNER, JOINER):
        cx2.execute("DELETE FROM subscriptions WHERE email=?", (e,))
    cx2.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (JOINER,))
    cx2.execute("DELETE FROM referral_codes WHERE email=?", (OWNER,)); cx2.commit()
    cx.close(); cx2.close()


def test_membership_return_no_code_writes_no_row(monkeypatch):
    monkeypatch.setattr(appmod, "_portal_offers_enabled", lambda: True)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row; _clean(cx)
    monkeypatch.setattr(appmod.stripe_pay, "get_session",
                        lambda sid: {"metadata": {"kind": "group_join", "email": JOINER},
                                     "setup_intent": "seti_1"})
    monkeypatch.setattr(appmod.stripe_pay, "get_setup_intent",
                        lambda si: {"customer": "cus_j", "payment_method": "pm_j"})
    monkeypatch.setattr(appmod, "_member_join_welcome", lambda *a, **k: None)
    appmod.app.test_client().get("/portal/offer/live-group/return?session_id=cs_test")
    row = cx.execute("SELECT 1 FROM referral_redemptions WHERE referee_email=?", (JOINER,)).fetchone()
    assert row is None
    cx.execute("DELETE FROM subscriptions WHERE email=?", (JOINER,)); cx.commit(); cx.close()
