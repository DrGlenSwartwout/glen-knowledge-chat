import os
os.environ.setdefault("OPENAI_API_KEY", "test"); os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ["CAREGIVER_PAY_ENABLED"] = "1"
import app as appmod

def test_pay_consent_endpoint_sets_member_consent(monkeypatch):
    client = appmod.app.test_client()
    # portal token resolves to the MEMBER (michael)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "michael@x.com"})
    # seed the link so the UPDATE has a row
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        from dashboard import household as hh
        hh.init_household_tables(cx)
        hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
    r = client.post("/api/portal/tok123/pay-consent",
                    json={"caregiver_email": "steve@x.com", "consent": True, "share_scope": "amount_only"})
    assert r.get_json()["recorded"] is True
    with appmod._db_lock, appmod.db.connect(appmod.LOG_DB) as cx:
        from dashboard import household as hh
        assert hh.can_pay(cx, "steve@x.com", "michael@x.com") is True
