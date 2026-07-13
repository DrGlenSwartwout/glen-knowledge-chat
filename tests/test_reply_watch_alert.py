import importlib, sys
from pathlib import Path
import pytest

def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")

def test_token_missing_alerts_once(monkeypatch, tmp_path):
    app_module = _app()
    from dashboard import gmail_token as gt
    db = str(tmp_path / "chat_log.db")
    import sqlite3
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE oauth_tokens (name TEXT PRIMARY KEY, "
                   "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        cx.commit()
    monkeypatch.setattr(app_module, "LOG_DB", db)
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    # process_inbox_replies raises token-missing
    def _raise(*a, **k):
        raise gt.GmailTokenMissing("no token")
    monkeypatch.setattr("reply_watcher.process_inbox_replies", _raise)
    sent = []
    monkeypatch.setattr(app_module, "_send_token_alert",
                        lambda subject, body: sent.append(subject) or True)
    client = app_module.app.test_client()
    r1 = client.post("/api/cron/reply-watch", headers={"X-Cron-Secret": "s3cret"})
    r2 = client.post("/api/cron/reply-watch", headers={"X-Cron-Secret": "s3cret"})
    assert r1.status_code == 500 and r2.status_code == 500
    assert len(sent) == 1  # deduped within the 6h window
