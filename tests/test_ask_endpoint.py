import types

import app as appmod


def _open(monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "", raising=False)
    monkeypatch.setattr(appmod, "embed", lambda q: [0.0] * 1536, raising=False)
    return appmod.app.test_client()


def _match(title, source, text):
    return types.SimpleNamespace(metadata={"title": title, "source": source, "text": text})


def test_ask_returns_grounded_answer_with_sources(monkeypatch):
    c = _open(monkeypatch)
    monkeypatch.setattr(appmod._idx, "query",
                        lambda **kw: types.SimpleNamespace(matches=[_match("Clinical tagging", "project_x.md", "definite change = ...")]))
    monkeypatch.setattr(appmod._cl.messages, "create",
                        lambda **kw: types.SimpleNamespace(content=[types.SimpleNamespace(text="It works like X.")]))
    r = c.post("/api/ask", json={"question": "how does definite change work?"})
    d = r.get_json()
    assert r.status_code == 200 and d["ok"] and d["answer"] == "It works like X."
    assert any(s["source"] == "project_x.md" for s in d["sources"])


def test_ask_no_matches_graceful(monkeypatch):
    c = _open(monkeypatch)
    monkeypatch.setattr(appmod._idx, "query", lambda **kw: types.SimpleNamespace(matches=[]))
    r = c.post("/api/ask", json={"question": "unknown thing"})
    d = r.get_json()
    assert r.status_code == 200 and d["ok"] and "isn't in our systems docs" in d["answer"] and d["sources"] == []


def test_ask_empty_question_400(monkeypatch):
    c = _open(monkeypatch)
    r = c.post("/api/ask", json={"question": "  "})
    assert r.status_code == 400


def test_ask_bad_key_401(monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sekret", raising=False)
    r = appmod.app.test_client().post("/api/ask", json={"question": "x"})
    assert r.status_code == 401
