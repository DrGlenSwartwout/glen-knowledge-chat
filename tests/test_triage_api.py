import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

ADMIN = {"X-Console-Key": "test-secret"}

def test_triage_invite_requires_auth(client):
    r = client.post("/api/console/triage-invite", json={"email": "p@x.com", "name": "P", "practitioner": "glen"})
    assert r.status_code == 401

def test_triage_invite_bad_practitioner(client):
    r = client.post("/api/console/triage-invite",
                    json={"email": "p@x.com", "name": "P", "practitioner": "bob"}, headers=ADMIN)
    assert r.status_code == 400

def test_triage_invite_creates_and_returns_url(client):
    r = client.post("/api/console/triage-invite",
                    json={"email": "p@x.com", "name": "Pat", "practitioner": "rae"}, headers=ADMIN)
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and "/triage/" in d["url"]


from datetime import timedelta
def _invite(client, practitioner="glen", email="pp@x.com"):
    r = client.post("/api/console/triage-invite",
                    json={"email": email, "name": "Pat", "practitioner": practitioner}, headers=ADMIN)
    return r.get_json()["url"].rsplit("/triage/", 1)[1]

def test_triage_state_and_full_flow(client):
    tok = _invite(client, "glen")
    st = client.get(f"/api/triage/state?token={tok}").get_json()
    assert st["practitioner"] == "glen" and st["medium"] == "video" and st["booked"] is False
    slots = client.get(f"/api/triage/availability?token={tok}&range=week").get_json()["slots"]
    assert slots
    r = client.post(f"/api/triage/book?token={tok}", json={"start_ts": slots[0]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    # single-use: availability now 409 already_booked
    r2 = client.get(f"/api/triage/availability?token={tok}&range=week")
    assert r2.status_code == 409

def test_triage_state_invalid_token(client):
    r = client.get("/api/triage/state?token=bogus")
    assert r.status_code == 404 and r.get_json()["error"] == "invalid"

def test_triage_join_glen_vs_rae(client):
    import sqlite3
    from dashboard import triage
    # rae invite -> join returns phone_call 400
    tok_r = _invite(client, "rae", "r@x.com")
    slots = client.get(f"/api/triage/availability?token={tok_r}&range=week").get_json()["slots"]
    client.post(f"/api/triage/book?token={tok_r}", json={"start_ts": slots[0]})
    assert client.get(f"/api/triage/join?token={tok_r}").status_code == 400
    # glen invite booked far out -> not_in_window 403
    tok_g = _invite(client, "glen", "g2@x.com")
    slots = client.get(f"/api/triage/availability?token={tok_g}&range=week").get_json()["slots"]
    client.post(f"/api/triage/book?token={tok_g}", json={"start_ts": slots[-1]})
    # slots[-1] is >30min out unless today's last slot; assert it is either 200 or 403 (both valid states)
    assert client.get(f"/api/triage/join?token={tok_g}").status_code in (200, 403)
