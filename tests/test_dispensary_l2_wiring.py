import importlib
import sqlite3
import dashboard.practitioner_portal as pp_mod
from dashboard import referrals as rf, points, orders as o


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    monkeypatch.setenv("REFERRAL_TIER2_ENABLED", "true")
    monkeypatch.setenv("REFERRER_REWARD_PCT", "20")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _dispensary_order(email="pat@x.com", ref="INV1", total=7000, shipping=1300):
    return {"source": "dispensary", "practitioner_id": "prac-1", "email": email,
            "total_cents": total, "shipping_cents": shipping, "get_cents": 0,
            "external_ref": ref}


def test_card_path_settle_order_points_credits_dispensary_l2(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.init_tables(cx)
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    appmod._settle_order_points(_dispensary_order(), order_ref="INV1")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "upline@x.com") == 570


def test_altpay_record_payment_credits_dispensary_l2(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        o.init_orders_table(cx)
        rf.init_tables(cx)
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
        oid = o.upsert_order(cx, source="dispensary", external_ref="INV1", email="pat@x.com",
                             total_cents=7000, shipping_cents=1300, practitioner_id="prac-1")
        cx.commit()
        # invoke the alt-pay confirmation exec directly
        from dashboard.orders import _record_payment_exec
        _record_payment_exec({"order_id": oid, "method": "zelle", "amount_cents": 7000},
                             {"cx": cx})
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "upline@x.com") == 570


def test_non_dispensary_order_no_dispensary_l2(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.init_tables(cx)
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    order = _dispensary_order()
    order["source"] = "reorder"   # not dispensary
    appmod._settle_order_points(order, order_ref="INV1")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "upline@x.com") == 0
