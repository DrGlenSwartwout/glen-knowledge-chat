# tests/test_portal_base_url.py  (needs doppler — imports app)
# PORTAL_BASE_URL: client portal links move to their own host (myhealingoasis.com)
# while the funnel stays on PUBLIC_BASE_URL. Defaults to PUBLIC_BASE_URL so the
# code is a no-op until the env var is set.
import os, sqlite3, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

OASIS = "https://myhealingoasis.com"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    sent = []
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda to, name, subj, body: (sent.append({"to": to, "body": body}), ("console-log", None))[1])
    appmod.app.config["TESTING"] = True
    c = appmod.app.test_client()
    c.sent = sent
    return c


def test_portal_base_defaults_to_public(monkeypatch):
    monkeypatch.delenv("PORTAL_BASE_URL", raising=False)
    assert appmod.portal_base() == appmod.PUBLIC_BASE_URL
    assert appmod.portal_link("abc") == f"{appmod.PUBLIC_BASE_URL}/portal/abc"


def test_portal_base_honors_env(monkeypatch):
    monkeypatch.setenv("PORTAL_BASE_URL", OASIS + "/")   # trailing slash stripped
    assert appmod.portal_base() == OASIS
    assert appmod.portal_link("t0k") == f"{OASIS}/portal/t0k"


def test_portal_host_only_when_distinct(monkeypatch):
    monkeypatch.delenv("PORTAL_BASE_URL", raising=False)
    assert appmod._portal_host() == ""                    # shares funnel domain
    monkeypatch.setenv("PORTAL_BASE_URL", OASIS)
    assert appmod._portal_host() == "myhealingoasis.com"
    monkeypatch.setenv("PORTAL_BASE_URL", appmod.PUBLIC_BASE_URL)  # same as funnel
    assert appmod._portal_host() == ""


def test_root_redirects_on_portal_host(client, monkeypatch):
    monkeypatch.setenv("PORTAL_BASE_URL", OASIS)
    r = client.get("/", headers={"Host": "myhealingoasis.com"}, base_url=OASIS)
    assert r.status_code == 302 and r.headers["Location"].endswith("/portal/login")


def test_root_serves_funnel_on_main_host(client, monkeypatch):
    monkeypatch.setenv("PORTAL_BASE_URL", OASIS)
    # a request on the funnel host must NOT redirect (serves the funnel home)
    r = client.get("/", headers={"Host": "illtowell.com"}, base_url="https://illtowell.com")
    assert r.status_code != 302


def test_healing_oasis_link_uses_portal_host(client, monkeypatch):
    monkeypatch.setenv("HEALING_OASIS_ENABLED", "1")
    monkeypatch.setenv("PORTAL_BASE_URL", OASIS)
    r = client.post("/api/healing-oasis/request", json={"name": "Jane", "email": "jane@x.com"})
    assert r.status_code == 200
    assert len(client.sent) == 1
    assert f"{OASIS}/portal/" in client.sent[0]["body"]
