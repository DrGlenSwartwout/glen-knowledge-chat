import importlib
import sys
from pathlib import Path

import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:  # missing env (Pinecone etc.) -> runs in CI only
        pytest.skip(f"app not importable in this env: {e}")


def test_action_route_completes_todo(monkeypatch, tmp_path):
    app_module = _load_app()
    import sqlite3
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY, status TEXT, done_at TEXT)")
    from dashboard import events as E
    E.init_event_tables(cx)
    cx.execute("INSERT INTO todos (id, status) VALUES (3, 'open')")
    cx.commit(); cx.close()

    client = app_module.app.test_client()
    key = app_module.dashboard.CONSOLE_SECRET or ""
    r = client.post("/api/action/tasks.complete_todo",
                    json={"todo_id": 3},
                    headers={"X-Console-Key": key})
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "done"

    r2 = client.get("/api/events", headers={"X-Console-Key": key})
    assert r2.status_code == 200
    assert any(e["action_key"] == "tasks.complete_todo" for e in r2.get_json()["data"])


def test_home_signals_route(monkeypatch, tmp_path):
    app_module = _load_app()
    import sqlite3
    db = str(tmp_path / "h.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    cx = sqlite3.connect(db)
    from dashboard import events as E
    E.init_event_tables(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY, status TEXT, priority TEXT)")
    cx.execute("INSERT INTO todos (status, priority) VALUES ('open','high')")
    cx.commit(); cx.close()

    client = app_module.app.test_client()
    key = app_module.dashboard.CONSOLE_SECRET or ""
    r = client.get("/api/home/signals", headers={"X-Console-Key": key})
    assert r.status_code == 200
    cells = r.get_json()["data"]
    assert len(cells) == 9
    tasks = [c for c in cells if c["module"] == "tasks"][0]
    assert tasks["level"] == "red"  # an open high-priority todo


def test_home_page_served(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "p.db"))
    client = app_module.app.test_client()
    r = client.get("/console/home")
    assert r.status_code == 200
    assert b"Home" in r.data or b"home" in r.data
