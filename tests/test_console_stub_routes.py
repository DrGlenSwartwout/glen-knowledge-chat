import app as appmod


def _client(monkeypatch):
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "", raising=False)
    return appmod.app.test_client()


def test_social_and_rnd_stub_pages_served(monkeypatch):
    c = _client(monkeypatch)
    for path, marker in [("/console/social", b"Social Media"), ("/console/rnd", b"Formulations")]:
        r = c.get(path)
        assert r.status_code == 200
        assert marker in r.data
        assert b"op-nav.js" in r.data     # carries the shared nav
