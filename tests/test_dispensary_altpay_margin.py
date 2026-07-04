"""Alt-pay parity: confirming a Zelle/Wise dispensary payment credits the practitioner's
own Wellness Credit (wallet margin) + records the sale — through the real
_record_payment_exec, proving margin_cents/practitioner_id thread onto the order and the
alt-pay hook fires. External effects mocked (wallet=Supabase; record path caches DATA_DIR)."""
import importlib
import sqlite3
import dashboard.practitioner_portal as pp_mod
import dashboard.wallet as wallet_mod
from dashboard import referrals as rf, orders as o


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_altpay_dispensary_credits_practitioner_margin(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    earn, rec = [], []
    monkeypatch.setattr(wallet_mod, "earn_dropship_margin",
                        lambda pid, margin, *, qbo_invoice_id, ref=None:
                        (earn.append((pid, margin, qbo_invoice_id)), margin)[1])
    monkeypatch.setattr(pp_mod, "record_dispensary_order",
                        lambda pid, *, invoice_id, credit_earned_cents, **k:
                        rec.append((pid, invoice_id, credit_earned_cents)))
    with sqlite3.connect(appmod.LOG_DB) as cx:
        o.init_orders_table(cx)
        rf.init_tables(cx)
        oid = o.upsert_order(cx, source="dispensary", external_ref="INV1", email="pat@x.com",
                             total_cents=7000, shipping_cents=1300, practitioner_id="prac-1",
                             margin_cents=2000)
        cx.commit()
        cx.row_factory = sqlite3.Row
        from dashboard.orders import _record_payment_exec
        _record_payment_exec({"order_id": oid, "method": "zelle", "amount_cents": 7000}, {"cx": cx})
    assert earn == [("prac-1", 2000, "INV1")]     # margin threaded off the order → credited
    assert rec == [("prac-1", "INV1", 2000)]
