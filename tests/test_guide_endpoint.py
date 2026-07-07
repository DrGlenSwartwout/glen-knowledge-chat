import sqlite3
import app as appmod


def _open_client(monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "", raising=False)  # endpoint reads dashboard.CONSOLE_SECRET
    return appmod.app.test_client()


def test_guide_creates_page_tagged_todo(monkeypatch):
    c = _open_client(monkeypatch)
    r = c.post("/api/guide", json={"text": "Move the refund button up", "active": "finance", "sub": "money", "url": "/console/money"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    cx = sqlite3.connect(str(appmod.LOG_DB))
    row = cx.execute("SELECT category, source, title, body FROM todos WHERE source LIKE 'ask-guide:%' ORDER BY id DESC LIMIT 1").fetchone()
    cx.close()
    assert row is not None
    assert row[0] == "Guidance" and row[1] == "ask-guide:finance/money"
    assert "Move the refund button up" in row[2]          # title
    assert "finance/money" in row[3] and "/console/money" in row[3]   # body carries page + url


def test_guide_empty_text_400(monkeypatch):
    c = _open_client(monkeypatch)
    r = c.post("/api/guide", json={"text": "   ", "active": "finance"})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_guide_bad_key_401(monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "sekret", raising=False)
    r = appmod.app.test_client().post("/api/guide", json={"text": "x", "active": "a"})
    assert r.status_code == 401
