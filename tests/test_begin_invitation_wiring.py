"""The served landing page must actually carry the invitation wiring.

These are deliberately shallow assertions on served HTML: the browser
behaviour (autoplay policy, audio unlock, fullscreen) cannot be asserted
headlessly and is covered by the manual verification record instead.
"""
import importlib


def _reload_app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FIRESIDE_ENABLED", "true")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_landing_page_carries_invitation_tile(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert 'id="fireside-invite"' in body
    assert 'id="fs-invite-video"' in body
    assert 'id="fs-invite-choices"' in body
    assert "/static/begin/invitation-mount.js" in body


def test_invitation_tile_starts_hidden(monkeypatch, tmp_path):
    """It must not occupy layout until a clip actually resolves."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    idx = body.index('id="fireside-invite"')
    tag = body[idx - 200 : idx + 200]
    assert "hidden" in tag


def test_invitation_ctas_use_exact_copy(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert "Come sit by the fire" in body
    assert "Ask here instead" in body
    assert 'href="/begin/fireside"' in body


def test_invitation_module_is_served(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    assert c.get("/static/begin/invitation.js").status_code == 200
    assert c.get("/static/begin/invitation-mount.js").status_code == 200
