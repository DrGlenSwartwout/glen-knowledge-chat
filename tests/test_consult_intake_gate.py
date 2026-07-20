# tests/test_consult_intake_gate.py
import importlib, sqlite3, pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Set DATA_DIR before the reload, never at module import. A module-level
    # os.environ.setdefault("DATA_DIR", ...) runs during COLLECTION -- pytest imports every
    # collected module before any test runs -- so it leaked a global DATA_DIR into the whole
    # session and pointed unrelated tests at a shared /tmp DB. See test_intake_routes.py.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app as appmod
    importlib.reload(appmod)

    class _Ident:
        def __init__(self, email): self.email = email
    monkeypatch.setattr(appmod, "_evox_ident",
                        lambda cx, token: _Ident("m@x.com") if token == "good" else None)
    # consult is "ready" for our member so we exercise the intake gate, not the ready gate
    from dashboard import consult as _c
    monkeypatch.setattr(_c, "consult_is_ready", lambda cx, email: True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_availability_blocked_until_intake(client):
    r = client.get("/api/consult/availability?token=good")
    assert r.status_code == 409 and r.get_json()["error"] == "intake_required"


def test_book_blocked_until_intake(client):
    r = client.post("/api/consult/book?token=good", json={"start_ts": "2026-07-10T09:00:00"})
    assert r.status_code == 409 and r.get_json()["error"] == "intake_required"


def test_state_reports_intake_submitted_flag(client):
    r = client.get("/api/consult/state?token=good")
    assert r.get_json().get("intake_submitted") is False
