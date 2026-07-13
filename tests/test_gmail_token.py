import json
import sqlite3
import pytest
from pathlib import Path

from dashboard import gmail_token as gt

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def _token_json(access="ya29.access", refresh="1//refresh"):
    return json.dumps({
        "token": access,
        "refresh_token": refresh,
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "cid.apps.googleusercontent.com",
        "client_secret": "secret",
        "scopes": SCOPES,
    })

def _db(tmp_path):
    p = tmp_path / "chat_log.db"
    with sqlite3.connect(p) as cx:
        cx.execute("CREATE TABLE oauth_tokens (name TEXT PRIMARY KEY, "
                   "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        cx.commit()
    return str(p)

def test_loads_from_db_when_present(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO oauth_tokens VALUES (?,?,?)",
                   ("inbox_gmail", _token_json(), "2026-07-13T00:00:00Z"))
        cx.commit()
    loaded = gt.load_gmail_credentials(db, name="inbox_gmail", scopes=SCOPES)
    assert loaded.source == "db"
    assert loaded.name == "inbox_gmail"
    assert loaded.creds.refresh_token == "1//refresh"

def test_falls_back_to_file_and_self_heals_db(tmp_path, monkeypatch):
    db = _db(tmp_path)
    tokfile = tmp_path / "google-token.json"
    tokfile.write_text(_token_json(access="from-file"))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tokfile))
    loaded = gt.load_gmail_credentials(db, name="inbox_gmail", scopes=SCOPES)
    assert loaded.source == "file"
    # self-heal: DB row now exists
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT token_json FROM oauth_tokens WHERE name=?",
                         ("inbox_gmail",)).fetchone()
    assert row is not None
    assert json.loads(row[0])["token"] == "from-file"

def test_raises_when_nowhere(tmp_path, monkeypatch):
    db = _db(tmp_path)
    monkeypatch.delenv("GMAIL_TOKEN_PATH", raising=False)
    monkeypatch.setattr(gt, "_FILE_CANDIDATES", [str(tmp_path / "nope.json")])
    with pytest.raises(gt.GmailTokenMissing):
        gt.load_gmail_credentials(db, name="inbox_gmail", scopes=SCOPES)

def test_persist_writes_only_when_changed(tmp_path):
    db = _db(tmp_path)
    creds = gt._build_creds(_token_json(access="old"), SCOPES)
    baseline = creds.to_json()
    # unchanged -> no write, returns False
    loaded_same = gt.LoadedGmail(creds=creds, source="db",
                                 original_json=baseline, name="inbox_gmail")
    assert gt.persist_refreshed_credentials(db, loaded_same) is False
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM oauth_tokens").fetchone()[0] == 0
    # changed (simulate refresh) -> writes, returns True
    creds.token = "new-access"
    loaded_changed = gt.LoadedGmail(creds=creds, source="db",
                                    original_json=baseline, name="inbox_gmail")
    assert gt.persist_refreshed_credentials(db, loaded_changed) is True
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT token_json FROM oauth_tokens WHERE name=?",
                         ("inbox_gmail",)).fetchone()
    assert json.loads(row[0])["token"] == "new-access"
