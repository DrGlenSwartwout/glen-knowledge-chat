import os
import uuid
os.environ.setdefault("OPENAI_API_KEY", "test"); os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ["CAREGIVER_PAY_ENABLED"] = "1"
import app as appmod
from dashboard import order_payments as op, household as hh


def _seed_order(email, *, status="confirmed", pay_status="unpaid", total_cents=5000):
    """Insert a fresh order owned by email, return its id."""
    ext_ref = f"caregiver-pay-initiate-test-{uuid.uuid4().hex}"
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        op.ensure_table(cx)
        hh.init_household_tables(cx)
        cx.execute(
            "INSERT INTO orders (created_at, source, external_ref, email, name, "
            "status, pay_status, total_cents) "
            "VALUES ('2026-07-23T00:00:00', 'test', ?, ?, 'Michael', ?, ?, ?)",
            (ext_ref, email, status, pay_status, total_cents))
        oid = cx.execute("SELECT id FROM orders WHERE external_ref=?", (ext_ref,)).fetchone()[0]
        cx.commit()
    return oid


def test_initiate_flag_off_returns_404(monkeypatch):
    michael = f"michael-{uuid.uuid4().hex}@x.com"
    oid = _seed_order(michael)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "steve@x.com"})
    client = appmod.app.test_client()
    monkeypatch.setenv("CAREGIVER_PAY_ENABLED", "")
    try:
        r = client.post("/api/portal/toksteve/caregiver-pay", json={"order_id": oid})
    finally:
        monkeypatch.setenv("CAREGIVER_PAY_ENABLED", "1")
    assert r.status_code == 404


def test_initiate_no_consent_returns_403(monkeypatch):
    michael = f"michael-{uuid.uuid4().hex}@x.com"
    steve = f"steve-{uuid.uuid4().hex}@x.com"
    oid = _seed_order(michael)
    # deliberately no add_member/set_pay_consent for steve -> can_pay is False
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": steve})
    client = appmod.app.test_client()
    r = client.post("/api/portal/toksteve/caregiver-pay", json={"order_id": oid})
    assert r.status_code == 403
    assert r.get_json()["ok"] is False


def test_initiate_with_consent_returns_checkout_url(monkeypatch):
    michael = f"michael-{uuid.uuid4().hex}@x.com"
    steve = f"steve-{uuid.uuid4().hex}@x.com"
    oid = _seed_order(michael)
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        hh.add_member(cx, steve, michael, relationship="partner")
        hh.set_pay_consent(cx, steve, michael, 1)
        cx.commit()
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": steve})
    monkeypatch.setattr("dashboard.stripe_pay.create_checkout_session",
                         lambda *a, **k: {"url": "https://x"})
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    client = appmod.app.test_client()
    r = client.post("/api/portal/toksteve/caregiver-pay", json={"order_id": oid})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["ok"] is True
    assert body["url"] == "https://x"


def test_initiate_already_paid_returns_409(monkeypatch):
    michael = f"michael-{uuid.uuid4().hex}@x.com"
    steve = f"steve-{uuid.uuid4().hex}@x.com"
    oid = _seed_order(michael, pay_status="paid")
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        hh.add_member(cx, steve, michael, relationship="partner")
        hh.set_pay_consent(cx, steve, michael, 1)
        cx.commit()
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": steve})
    client = appmod.app.test_client()
    r = client.post("/api/portal/toksteve/caregiver-pay", json={"order_id": oid})
    assert r.status_code == 409
    assert r.get_json()["ok"] is False
