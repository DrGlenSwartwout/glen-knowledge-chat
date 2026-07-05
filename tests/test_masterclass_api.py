import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    monkeypatch.setattr("dashboard.zoom.get_token", lambda *a, **k: "tok")
    monkeypatch.setattr("dashboard.zoom.create_meeting",
                        lambda *a, **k: {"join_url": "https://zoom.us/j/mc", "meeting_id": "mc1", "start_url": "x"})
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

ADMIN = {"X-Console-Key": "test-secret"}

def test_console_create_requires_auth(client):
    r = client.post("/api/console/masterclass", json={"topic": "T", "start_ts": "2026-07-10T18:00:00"})
    assert r.status_code == 401

def test_console_create_makes_event_and_zoom(client):
    r = client.post("/api/console/masterclass",
                    json={"topic": "Terrain 101", "description": "d", "start_ts": "2026-07-10T18:00:00",
                          "duration_min": 60, "price_cents": 5000, "member_price_cents": 0}, headers=ADMIN)
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["zoom_ok"] is True and "/masterclass/" in d["event_url"]
    import sqlite3
    from dashboard import masterclass as mc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        ev = mc.get_event(cx, d["event_id"])
        assert ev["zoom_join_url"] == "https://zoom.us/j/mc"


def _mk_event(client, price=0, mprice=0):
    r = client.post("/api/console/masterclass",
                    json={"topic": "T", "description": "d", "start_ts": "2026-07-10T18:00:00",
                          "duration_min": 60, "price_cents": price, "member_price_cents": mprice}, headers=ADMIN)
    return r.get_json()["event_id"]

def test_public_get_event(client):
    eid = _mk_event(client, price=5000)
    d = client.get(f"/api/masterclass/{eid}").get_json()
    assert d["topic"] == "T" and d["price_cents"] == 5000 and "zoom_join_url" not in d

def test_register_free_returns_join_link(client, monkeypatch):
    eid = _mk_event(client, price=0, mprice=0)
    r = client.post(f"/api/masterclass/{eid}/register", json={"email": "free@x.com", "name": "F"})
    d = r.get_json()
    assert r.status_code == 200 and d["registered"] is True and d["join_url"] == "https://zoom.us/j/mc"

def test_register_nonmember_paid_returns_checkout(client, monkeypatch):
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True, raising=False)
    import dashboard.stripe_pay as _sp
    cap = {}
    def fake_session(amount_cents, *, customer_email, description, metadata, success_url, cancel_url, save_card=False):
        cap["amount"] = amount_cents; cap["metadata"] = metadata
        return {"id": "cs_test", "url": "https://stripe/mc"}
    monkeypatch.setattr(_sp, "create_checkout_session", fake_session)
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session", fake_session, raising=False)
    eid = _mk_event(client, price=5000, mprice=0)
    r = client.post(f"/api/masterclass/{eid}/register", json={"email": "nonmember@x.com", "name": "N"})
    d = r.get_json()
    assert r.status_code == 200 and d["checkout_url"] == "https://stripe/mc"
    assert cap["amount"] == 5000 and cap["metadata"]["kind"] == "masterclass" and cap["metadata"]["event_id"] == str(eid)

def test_fulfill_masterclass_marks_paid_and_sends(client, monkeypatch):
    sent = []
    monkeypatch.setattr(appmod, "send_evox_email", lambda to, *a, **k: sent.append(to) or ("console-log", None), raising=False)
    eid = _mk_event(client, price=5000, mprice=0)
    import sqlite3
    from dashboard import masterclass as mc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        mc.register(cx, eid, "buyer@x.com", "B", is_member=False, amount_cents=5000, paid=False)
        cx.commit()
    import dashboard.stripe_pay as _sp
    monkeypatch.setattr(_sp, "get_session",
                        lambda sid: {"metadata": {"kind": "masterclass", "event_id": str(eid), "email": "buyer@x.com", "name": "B"},
                                     "payment_intent": "pi_1"})
    monkeypatch.setattr(_sp, "get_payment_intent", lambda pi: {"status": "succeeded"})
    out = appmod._fulfill_masterclass("cs_test")
    assert out["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert mc.is_registered(cx, eid, "buyer@x.com") is True
    assert "buyer@x.com" in sent
    # non-masterclass session is a no-op
    monkeypatch.setattr(_sp, "get_session", lambda sid: {"metadata": {"kind": "retail"}})
    assert appmod._fulfill_masterclass("cs_other")["ok"] is False


def test_masterclass_page_served(client):
    r = client.get("/masterclass/1")
    assert r.status_code == 200 and b"MasterClass" in r.data
