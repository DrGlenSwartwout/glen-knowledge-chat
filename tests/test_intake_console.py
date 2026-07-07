import os
os.environ.setdefault("DATA_DIR", "/tmp/intake-con")
import importlib, sqlite3, pytest


@pytest.fixture
def client(monkeypatch):
    import app as appmod
    importlib.reload(appmod)
    monkeypatch.setattr(appmod, "_portal_console_ok",
                        lambda: bool(__import__("flask").request.args.get("key") == "K"))
    # seed a submitted intake
    from dashboard import intake
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        intake.init_intake_table(cx)
        intake.submit(cx, "seed@x.com", {"first_name": "Seed"}, "2026-07-07T00:00:00")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_console_intake_requires_key(client):
    assert client.get("/api/console/intake/seed@x.com").status_code == 401


def test_console_intake_returns_response(client):
    r = client.get("/api/console/intake/seed@x.com?key=K")
    assert r.status_code == 200 and r.get_json()["answers"]["first_name"] == "Seed"


def test_console_submissions_list(client):
    r = client.get("/api/console/intake-submissions?key=K")
    assert any(x["email"] == "seed@x.com" for x in r.get_json()["submissions"])
