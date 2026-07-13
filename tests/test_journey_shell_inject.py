"""Journey shell after_request injection — pure helpers + wired behavior.
Mirrors the LOG_DB/CONSOLE_SECRET monkeypatch fixture pattern (see test_calendar.py)."""
import pytest
import shell_nav


# ---- pure helpers ----
def test_should_inject_public_html_200():
    assert shell_nav.should_inject("/begin", "text/html; charset=utf-8", 200) is True


@pytest.mark.parametrize("path", ["/console/orders", "/admin/x", "/api/journey",
                                  "/static/shell.js", "/begin/state",
                                  "/begin/fireside", "/begin/fireside/"])
def test_should_not_inject_excluded_paths(path):
    # /begin/fireside is the chrome-less immersive fireside: no shell ribbon.
    assert shell_nav.should_inject(path, "text/html", 200) is False


def test_should_not_inject_non_html():
    assert shell_nav.should_inject("/begin/state", "application/json", 200) is False


def test_should_not_inject_non_200():
    assert shell_nav.should_inject("/begin", "text/html", 302) is False


def test_resolve_mode_member_when_authenticated():
    assert shell_nav.resolve_mode("/begin", True) == "member"


def test_resolve_mode_member_for_member_paths():
    assert shell_nav.resolve_mode("/client-portal", False) == "member"
    assert shell_nav.resolve_mode("/coaching", False) == "member"


def test_resolve_mode_funnel_default():
    assert shell_nav.resolve_mode("/begin/match", False) == "funnel"


def test_inject_adds_assets_before_head_close():
    out = shell_nav.inject_shell_html("<head><title>x</title></head><body></body>", "funnel")
    assert "/static/shell.js" in out and "/static/shell.css" in out
    assert '"mode":"funnel"' in out or "'mode':'funnel'" in out
    assert out.index("shell.js") < out.index("</head>")


def test_inject_theme_adds_controller_scripts():
    out = shell_nav.inject_theme_html("<head></head>")
    assert "/static/sun-engine.js" in out
    assert "/static/theme-mode.js" in out
    assert out.index("sun-engine.js") < out.index("theme-mode.js")


def test_inject_theme_is_idempotent():
    once = shell_nav.inject_theme_html("<head></head>")
    twice = shell_nav.inject_theme_html(once)
    assert twice.count("/static/sun-engine.js") == 1


def test_inject_theme_noop_without_head():
    assert shell_nav.inject_theme_html("<body>no head</body>") == "<body>no head</body>"


def test_theme_controller_before_shell_when_both_applied():
    html = shell_nav.inject_theme_html("<head></head>")
    html = shell_nav.inject_shell_html(html, "funnel")
    assert html.index("theme-mode.js") < html.index("shell.js")
    assert html.index("sun-engine.js") < html.index("theme-mode.js")


def test_inject_is_idempotent():
    once = shell_nav.inject_shell_html("<head></head>", "funnel")
    twice = shell_nav.inject_shell_html(once, "funnel")
    assert twice.count("/static/shell.js") == 1


def test_inject_noop_without_head():
    assert shell_nav.inject_shell_html("<body>no head</body>", "funnel") == "<body>no head</body>"


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "JOURNEY_SHELL_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_shell_injected_on_begin_page(client):
    c, _ = client
    body = c.get("/begin").get_data(as_text=True)
    assert "/static/shell.js" in body
    assert 'window.__SHELL__' in body


def test_shell_not_injected_on_begin_state_json(client):
    c, _ = client
    body = c.get("/begin/state").get_data(as_text=True)
    assert "/static/shell.js" not in body


def test_shell_noop_when_flag_off(client):
    c, appmod = client
    appmod.JOURNEY_SHELL_ENABLED = False
    body = c.get("/begin").get_data(as_text=True)
    assert "/static/shell.js" not in body


def test_quest_disabled_no_quest_assets():
    out = shell_nav.inject_shell_html("<head></head>", "funnel", quest_enabled=False)
    assert "journey-quest.js" not in out
    assert "journey-quest.css" not in out
    assert '"questEnabled":false' in out


def test_quest_enabled_injects_assets():
    out = shell_nav.inject_shell_html("<head></head>", "funnel", quest_enabled=True)
    assert "journey-quest.js" in out
    assert "journey-quest.css" in out
    assert '"questEnabled":true' in out


def test_quest_audio_script_injected_when_enabled():
    out = shell_nav.inject_shell_html("<head></head>", "funnel", quest_enabled=True)
    assert "journey-audio.js" in out
    # audio script must appear before quest script so __JQAUDIO__ is defined first
    assert out.index("journey-audio.js") < out.index("journey-quest.js")


def test_quest_audio_script_absent_when_disabled():
    out = shell_nav.inject_shell_html("<head></head>", "funnel", quest_enabled=False)
    assert "journey-audio.js" not in out
