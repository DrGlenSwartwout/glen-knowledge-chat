import os
import uuid
os.environ.setdefault("OPENAI_API_KEY", "test"); os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ["CAREGIVER_PAY_ENABLED"] = "1"
import app as appmod
from dashboard import order_payments as op, household as hh, rbac as _rbac
from dashboard import orders as O


def _seed_order(michael_email):
    """Insert a fresh order owned by michael_email, return its id."""
    ext_ref = f"caregiver-pay-manual-test-{uuid.uuid4().hex}"
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        op.ensure_table(cx)
        hh.init_household_tables(cx)
        O.upsert_order(cx, source="test", external_ref=ext_ref,
                        email=michael_email, total_cents=5000)
        oid = cx.execute(
            "SELECT id FROM orders WHERE external_ref=?", (ext_ref,)).fetchone()[0]
        cx.commit()
    return oid


def test_manual_payment_stamps_authorized_payer(monkeypatch):
    michael = f"michael-{uuid.uuid4().hex}@x.com"
    steve = f"steve-{uuid.uuid4().hex}@x.com"
    oid = _seed_order(michael)
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        hh.add_member(cx, steve, michael, relationship="partner")
        hh.set_pay_consent(cx, steve, michael, 1)
        cx.commit()

    monkeypatch.setattr(appmod, "_bos_actor",
                         lambda: _rbac.Actor(role=_rbac.OWNER, name="owner"))
    client = appmod.app.test_client()
    r = client.post(f"/api/orders/{oid}/payments",
                     json={"amount": 50.00, "method": "Zelle", "payer_email": steve})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["ok"] is True

    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        row = cx.execute(
            "SELECT payer_email FROM order_payments WHERE order_id=?", (oid,)).fetchone()
    assert row[0] == steve


def test_manual_payment_rejects_unauthorized_payer(monkeypatch):
    """No pay-consent on file for this payer -> 403, and NO payment row written."""
    michael = f"michael-{uuid.uuid4().hex}@x.com"
    stranger = f"stranger-{uuid.uuid4().hex}@x.com"
    oid = _seed_order(michael)
    # deliberately no add_member / set_pay_consent for stranger

    monkeypatch.setattr(appmod, "_bos_actor",
                         lambda: _rbac.Actor(role=_rbac.OWNER, name="owner"))
    client = appmod.app.test_client()
    r = client.post(f"/api/orders/{oid}/payments",
                     json={"amount": 50.00, "method": "Zelle", "payer_email": stranger})
    assert r.status_code == 403
    assert r.get_json()["ok"] is False

    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        rows = cx.execute(
            "SELECT * FROM order_payments WHERE order_id=?", (oid,)).fetchall()
    assert rows == []
