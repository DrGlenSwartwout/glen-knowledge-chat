import os
import uuid
os.environ.setdefault("OPENAI_API_KEY", "test"); os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ["CAREGIVER_PAY_ENABLED"] = "1"
import app as appmod
from dashboard import order_payments as op

def test_fulfill_books_payment_to_payer(monkeypatch):
    ext_ref = f"caregiver-pay-test-{uuid.uuid4().hex}"
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        op.ensure_table(cx)
        cx.execute(
            "INSERT INTO orders (created_at, source, external_ref, email, name) "
            "VALUES ('2026-07-23T00:00:00', 'test', ?, 'michael@x.com','Michael')",
            (ext_ref,))
        oid = cx.execute("SELECT id FROM orders WHERE email='michael@x.com' ORDER BY id DESC LIMIT 1").fetchone()[0]
        cx.commit()
    fake = {"payment_status": "paid", "payment_intent": "pi_test_1", "amount_total": 5000,
            "metadata": {"kind": "caregiver-pay", "order_id": str(oid), "payer_email": "steve@x.com"}}
    monkeypatch.setattr(appmod, "_bos_actor", lambda: "system", raising=False)
    monkeypatch.setattr("dashboard.stripe_pay.get_session", lambda sid: fake)
    appmod._fulfill_caregiver_pay("cs_test_1")
    appmod._fulfill_caregiver_pay("cs_test_1")  # idempotent — same pi, no double row
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        rows = cx.execute("SELECT payer_email FROM order_payments WHERE order_id=? AND kind='payment'", (oid,)).fetchall()
    assert [r[0] for r in rows] == ["steve@x.com"]
