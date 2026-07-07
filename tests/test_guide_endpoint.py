import app as appmod


def _open_client(monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "", raising=False)  # endpoint reads dashboard.CONSOLE_SECRET
    return appmod.app.test_client()


def test_guide_adds_page_tagged_pending_idea(monkeypatch):
    from dashboard import projects as pj
    c = _open_client(monkeypatch)
    before = len(pj.pending_ideas())
    r = c.post("/api/guide", json={"text": "Move the refund button up", "active": "finance", "sub": "money", "url": "/console/money"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    ideas = pj.pending_ideas()
    assert len(ideas) == before + 1
    # the newest idea carries the guidance text + the page tag
    assert any("Move the refund button up" in i["text"] and "finance/money" in i["text"] for i in ideas)


def test_guide_empty_text_400(monkeypatch):
    c = _open_client(monkeypatch)
    r = c.post("/api/guide", json={"text": "   ", "active": "finance"})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_guide_bad_key_401(monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sekret", raising=False)
    r = appmod.app.test_client().post("/api/guide", json={"text": "x", "active": "a"})
    assert r.status_code == 401
