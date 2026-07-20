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


def test_embed_listens_for_the_unlock_message(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/embed?mode=funnel").get_data(as_text=True)
    assert "begin:audio-unlocked" in body
    assert "__audioUnlocked" in body


def test_embed_can_auto_speak_and_keeps_the_listen_button(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/embed?mode=funnel").get_data(as_text=True)
    # both paths must survive: automatic playback AND the manual control
    assert "TTS.attachAndSpeak" in body
    assert "TTS.attach(" in body


def test_embed_unlock_listener_checks_origin(monkeypatch, tmp_path):
    """A cross-origin frame must not be able to force audio playback."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/embed?mode=funnel").get_data(as_text=True)
    idx = body.index("begin:audio-unlocked")
    assert "location.origin" in body[idx - 400 : idx + 400]


def test_fireside_has_a_fullscreen_button(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin/fireside").get_data(as_text=True)
    assert 'id="fsBtn"' in body
    assert "requestFullscreen" in body


def test_fireside_fullscreen_is_feature_detected(monkeypatch, tmp_path):
    """iOS Safari cannot fullscreen arbitrary elements; the button must hide
    there rather than sit on the page doing nothing."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin/fireside").get_data(as_text=True)
    assert "webkitRequestFullscreen" in body
    idx = body.index("webkitRequestFullscreen")
    assert "hidden" in body[idx : idx + 600]


# ── both chat surfaces ──────────────────────────────────────────────────────
# The landing page has TWO chats: #hero-chat, rendered by begin.html itself
# beside the avatar, and the #begin-chat iframe (/embed?mode=funnel) further
# down. The first release wired only the iframe, so tapping the speaker did
# nothing for the conversation the visitor was actually having. These assert
# both, because a text-presence check on one file is what let that through.

def test_hero_chat_can_speak_its_replies(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert "/static/tts-output.js" in body, "hero chat needs window.TTS on this page"
    assert "heroSpeak" in body
    assert "heroSpeak(box, acc)" in body, "must speak the FINALIZED streamed reply"


def test_hero_chat_has_a_microphone(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert "/static/mic-input.js" in body
    idx = body.index('id="hero-input"')
    assert "data-mic" in body[idx : idx + 200], "the hero input itself must be mic-wired"


def test_unlock_flag_is_set_on_the_parent_page_too(monkeypatch, tmp_path):
    """postMessage only reaches the iframe; the hero chat reads the flag locally."""
    appmod = _reload_app(monkeypatch, tmp_path)
    js = appmod.app.test_client().get("/static/begin/invitation-mount.js").get_data(as_text=True)
    assert "window.__audioUnlocked = true" in js


def test_both_chat_surfaces_are_voice_wired(monkeypatch, tmp_path):
    """Neither surface may be left behind when voice changes are made."""
    c = _reload_app(monkeypatch, tmp_path).app.test_client()
    hero = c.get("/begin").get_data(as_text=True)
    embed = c.get("/embed?mode=funnel").get_data(as_text=True)
    for name, doc in (("hero chat (begin.html)", hero), ("iframe chat (embed.html)", embed)):
        assert "attachAndSpeak" in doc, f"{name} cannot speak replies"
        assert "/transcribe" in doc or "mic-input.js" in doc, f"{name} has no microphone"


def test_invitation_and_replies_do_not_overlap(monkeypatch, tmp_path):
    """Two separate audio players; without arbitration they speak at once."""
    c = _reload_app(monkeypatch, tmp_path).app.test_client()
    page = c.get("/begin").get_data(as_text=True)
    inv = c.get("/static/begin/invitation.js").get_data(as_text=True)
    mount = c.get("/static/begin/invitation-mount.js").get_data(as_text=True)
    assert "window.__invitation" in mount, "the invitation must be reachable to stop it"
    assert "inv.whenFree(" in page, "a reply must WAIT for the invitation, not cut it off"
    assert "window.__invitation.stop()" not in page, "the invitation must never be cut off"
    assert "whenFree(cb)" in inv, "Invitation must expose the wait hook"
    assert "window.TTS.stop()" in inv, "the invitation must silence a speaking reply"


def test_fullscreen_button_sits_above_the_entry_overlay(monkeypatch, tmp_path):
    """It rendered under #begin-overlay (z-index 5), so it LOOKED available and
    silently did nothing until 'Sit by the fire' was clicked."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin/fireside").get_data(as_text=True)
    i = body.index('id="fsBtn"')
    style = body[i : i + 400]
    z = int(style.split("z-index:")[1].split(";")[0].split('"')[0])
    assert z > 5, f"fsBtn z-index {z} is under the entry overlay (5)"
