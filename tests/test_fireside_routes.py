import importlib


def _reload_app(monkeypatch, tmp_path, enabled="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FIRESIDE_ENABLED", enabled)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_fireside_page_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="false")
    r = appmod.app.test_client().get("/begin/fireside")
    assert r.status_code == 404


def test_fireside_page_served_when_on(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    r = appmod.app.test_client().get("/begin/fireside")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "fireside" in body.lower()
    # sets the anonymous session cookie
    assert any("amg_session" in (h or "") for h in r.headers.getlist("Set-Cookie"))


class _FakeStream:
    def __init__(self, toks): self._toks = toks
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def text_stream(self):
        for t in self._toks: yield t

class _FakeMessages:
    def __init__(self, toks, boom=False): self._toks = toks; self.boom = boom; self.calls = 0
    def stream(self, **kw):
        self.calls += 1
        if self.boom: raise RuntimeError("claude down")
        return _FakeStream(self._toks)

class _FakeCl:
    def __init__(self, toks, boom=False): self.messages = _FakeMessages(toks, boom)


import re as _re, json as _json
def _tokens(body):
    return "".join(_json.loads(m)["token"]
                   for m in _re.findall(r'data: (\{"token":.*?\})\n\n', body))


def _post(appmod, message, sess="fixedsess"):
    # Send a fixed amg_session cookie so the route reuses a known session id
    # (otherwise it mints a random uuid and persistence is unreadable).
    # use_cookies=False: Werkzeug 3.x's cookie jar strips Cookie headers added
    # via headers={}; disabling it preserves the header so request.cookies works.
    return appmod.app.test_client(use_cookies=False).post(
        "/begin/fireside/agent", json={"message": message},
        headers={"Cookie": "amg_session=" + sess})


def test_agent_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="false")
    assert _post(appmod, "hi").status_code == 404


def test_agent_empty_message_400(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    assert _post(appmod, "   ").status_code == 400


def test_agent_streams_tokens_and_persists(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_cl", _FakeCl(["I hear ", "you, ", "friend."]))
    body = _post(appmod, "I'm exhausted").get_data(as_text=True)
    assert "I hear you, friend." in _tokens(body)
    assert '"done": true' in body
    assert '"hook": false' in body
    # persisted: one traveler turn + one glendalf turn
    import sqlite3
    from dashboard import fireside_store as fs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        s = fs.get_or_create(cx, "fixedsess")  # same cookie the POST sent
    # transcript should hold both turns under that session
    assert any(t["speaker"] == "glendalf" and "I hear you, friend." == t["text"]
               for t in s["transcript"])


def test_agent_hides_hook_marker_and_flags_when_eligible(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    # Force eligibility regardless of turn count/coverage
    from dashboard import fireside_agent as fa
    monkeypatch.setattr(fa, "hook_eligible", lambda *a, **k: True)
    monkeypatch.setattr(appmod, "_cl",
                        _FakeCl(["Shall we go and find it?", "\n", "⟦HOOK⟧"]))
    body = _post(appmod, "I think I'm ready").get_data(as_text=True)
    assert "⟦HOOK⟧" not in body          # marker never reaches the client
    assert "Shall we go and find it?" in _tokens(body)
    assert '"hook": true' in body


def test_agent_hook_marker_ignored_when_not_eligible(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    from dashboard import fireside_agent as fa
    monkeypatch.setattr(fa, "hook_eligible", lambda *a, **k: False)
    monkeypatch.setattr(appmod, "_cl", _FakeCl(["Too soon.", "⟦HOOK⟧"]))
    body = _post(appmod, "first thing I say").get_data(as_text=True)
    assert "⟦HOOK⟧" not in body
    assert '"hook": false' in body        # server refuses to honor an early close


def test_agent_error_frame_on_model_failure(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_cl", _FakeCl([], boom=True))
    body = _post(appmod, "hello").get_data(as_text=True)
    assert '"error": true' in body


def test_agent_split_hook_marker_does_not_leak(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    from dashboard import fireside_agent as fa
    monkeypatch.setattr(fa, "hook_eligible", lambda *a, **k: True)
    # sentinel arrives SPLIT across tokens, as real streaming does
    monkeypatch.setattr(appmod, "_cl", _FakeCl(["Shall we go find it?", "\n", "⟦HO", "OK⟧"]))
    body = _post(appmod, "ready").get_data(as_text=True)
    assert "⟦" not in body and "HOOK" not in body   # no sentinel fragment leaks
    assert '"hook": true' in body                          # still detected -> hook fires
    assert "Shall we go find it?" in _tokens(body)


def test_agent_rate_limited_returns_429(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_velocity_guard",
                        lambda *a, **k: (appmod.jsonify({"error": "rate_limited"}), 429))
    assert _post(appmod, "hello").status_code == 429


def test_manifest_served(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    r = appmod.app.test_client().get("/static/fireside/fireside-manifest.json")
    assert r.status_code == 200
    data = r.get_json()
    # Assert the contract, not which clip. This previously pinned the literal
    # "intro.mp4" and broke the moment the opening clip changed — twice: #453
    # repointed it at rest-1.mp4 to de-black the intro, then #456 at
    # intro-read.mp4 for the book intro. Which clip opens the fireside is a
    # creative decision; that it names a real, servable video is the invariant.
    intro = data["intro_video"]
    assert intro.startswith("/static/fireside/video/")
    assert intro.endswith(".mp4")
    assert appmod.app.test_client().get(intro).status_code == 200, (
        f"manifest declares {intro} but it is not served"
    )
    assert len(data["fillers"]) >= 5
    assert len(data["interjections"]) >= 3
