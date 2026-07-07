# tests/test_healing_oasis.py  (needs doppler — imports app)
# "My Healing Oasis" public front door: a visitor gives name+email and gets
# their /portal/<token> magic link emailed. Gated by HEALING_OASIS_ENABLED.
import os, sqlite3, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    sent = []

    def _cap(to_email, name, subject, body):
        sent.append({"to": to_email, "name": name, "subject": subject, "body": body})
        return ("console-log", None)

    monkeypatch.setattr(appmod, "_send_full_report_email", _cap)
    appmod.app.config["TESTING"] = True
    c = appmod.app.test_client()
    c.sent = sent
    return c


def test_status_reflects_flag(client, monkeypatch):
    monkeypatch.delenv("HEALING_OASIS_ENABLED", raising=False)
    assert client.get("/api/healing-oasis/status").get_json()["enabled"] is False
    monkeypatch.setenv("HEALING_OASIS_ENABLED", "1")
    assert client.get("/api/healing-oasis/status").get_json()["enabled"] is True


def test_request_404_when_disabled(client, monkeypatch):
    monkeypatch.delenv("HEALING_OASIS_ENABLED", raising=False)
    r = client.post("/api/healing-oasis/request", json={"name": "A", "email": "a@x.com"})
    assert r.status_code == 404
    assert client.sent == []


def test_request_sends_link_and_creates_portal(client, monkeypatch):
    monkeypatch.setenv("HEALING_OASIS_ENABLED", "1")
    r = client.post("/api/healing-oasis/request",
                    json={"name": "Jane Doe", "email": "Jane@X.com"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert len(client.sent) == 1
    msg = client.sent[0]
    assert msg["to"] == "jane@x.com"          # normalized
    assert "/portal/" in msg["body"]          # the magic link
    # the token is never returned to the unauthenticated caller
    assert "token" not in (r.get_json() or {})
    # a real portal row now exists for the email
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = cx.execute("SELECT email FROM client_portals WHERE email=?",
                         ("jane@x.com",)).fetchone()
    assert row is not None


def test_invalid_email_rejected(client, monkeypatch):
    monkeypatch.setenv("HEALING_OASIS_ENABLED", "1")
    r = client.post("/api/healing-oasis/request",
                    json={"name": "A", "email": "notanemail"})
    assert r.status_code == 400
    assert client.sent == []


def test_no_enumeration_generic_message(client, monkeypatch):
    monkeypatch.setenv("HEALING_OASIS_ENABLED", "1")
    r1 = client.post("/api/healing-oasis/request", json={"name": "A", "email": "new1@x.com"})
    r2 = client.post("/api/healing-oasis/request", json={"name": "B", "email": "new2@x.com"})
    assert r1.get_json()["message"] == r2.get_json()["message"]


def test_rate_limit_suppresses_send_per_email(client, monkeypatch):
    monkeypatch.setenv("HEALING_OASIS_ENABLED", "1")
    for _ in range(6):
        r = client.post("/api/healing-oasis/request",
                        json={"name": "A", "email": "spam@x.com"})
        # always the same generic response — never reveals the throttle
        assert r.status_code == 200 and r.get_json()["ok"] is True
    # but the actual sends are capped
    assert 1 <= len(client.sent) <= 3
