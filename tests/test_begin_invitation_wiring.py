"""The served landing page must actually carry the speaker wiring.

These are deliberately shallow assertions on served HTML: the browser
behaviour (audio playback, click-through suppression, the audio unlock)
cannot be asserted headlessly and is covered by the manual verification
record instead.
"""
import importlib


def _reload_app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FIRESIDE_ENABLED", "true")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_landing_page_carries_the_speaker_button(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert 'id="avatar-speaker"' in body
    assert "/static/begin/invitation-mount.js" in body


def test_speaker_starts_hidden(monkeypatch, tmp_path):
    """It must not appear until a manifest audio source actually resolves."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    idx = body.index('id="avatar-speaker"')
    assert "hidden" in body[idx - 200 : idx + 200]


def test_speaker_is_a_sibling_of_the_avatar_anchor(monkeypatch, tmp_path):
    """The speaker rides on the avatar without nesting interactive content
    inside the anchor, which is invalid HTML and breaks screen readers."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    anchor = body.index('<a class="avatar"')
    closing = body.index("</a>", anchor)
    speaker = body.index('id="avatar-speaker"')
    assert speaker > closing, "speaker must not be nested inside the anchor"
    wrap = body.index('class="avatar-wrap"')
    assert wrap < anchor, "both must live inside the positioned wrapper"


def test_no_second_fireside_cta_was_added(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert body.count('href="/begin/fireside"') == 1


def test_orphaned_fireside_invite_css_is_gone(monkeypatch, tmp_path):
    """ec233b56 removed the standalone section but left its CSS behind."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert "fiGlow" not in body
    assert ".fi-inner" not in body


def test_invitation_modules_are_served(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    assert c.get("/static/begin/invitation.js").status_code == 200
    assert c.get("/static/begin/invitation-mount.js").status_code == 200
