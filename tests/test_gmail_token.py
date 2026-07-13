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

def test_alert_dedup_within_window(tmp_path):
    db = _db(tmp_path)
    t0 = "2026-07-13T00:00:00+00:00"
    t_soon = "2026-07-13T02:00:00+00:00"   # +2h, inside 6h window
    t_later = "2026-07-13T07:00:00+00:00"  # +7h, outside window
    # first time: no health row -> should alert
    assert gt.should_send_alert(db, "inbox_gmail", t0) is True
    gt.record_alert(db, "inbox_gmail", t0)
    # inside window -> suppressed
    assert gt.should_send_alert(db, "inbox_gmail", t_soon) is False
    # outside window -> alert again
    assert gt.should_send_alert(db, "inbox_gmail", t_later) is True

def test_record_ok_clears_alert_and_marks_healthy(tmp_path):
    db = _db(tmp_path)
    t0 = "2026-07-13T00:00:00+00:00"
    gt.record_alert(db, "inbox_gmail", t0)
    gt.record_ok(db, "inbox_gmail", now_iso="2026-07-13T01:00:00+00:00")
    raw = gt._read_db_token(db, "inbox_gmail_health")
    state = json.loads(raw)
    assert state["healthy"] is True
    assert state["last_alert"] is None
    # after an OK, a later failure alerts again (window cleared)
    assert gt.should_send_alert(db, "inbox_gmail", "2026-07-13T01:30:00+00:00") is True

def test_alert_boundary_is_inclusive(tmp_path):
    db = _db(tmp_path)
    t0 = "2026-07-13T00:00:00+00:00"
    t_exact = "2026-07-13T06:00:00+00:00"  # exactly window_hours (6h) later
    gt.record_alert(db, "inbox_gmail", t0)
    # at exactly the window boundary the alert re-fires (>= semantics).
    # If the comparator regressed from >= to >, this returns False and the test fails.
    assert gt.should_send_alert(db, "inbox_gmail", t_exact) is True


def test_default_db_path_honors_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert gt.default_db_path() == str(tmp_path / "chat_log.db")


def test_default_db_path_falls_back_to_repo_root(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    expected_root = Path(gt.__file__).resolve().parent.parent
    assert gt.default_db_path() == str(expected_root / "chat_log.db")


def test_read_db_token_missing_table_returns_none(tmp_path):
    # A fresh DB with no oauth_tokens table at all (e.g. brand-new persistent
    # disk) must not raise sqlite3.OperationalError -- it should fall through
    # to the file fallback in load_gmail_credentials.
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE unrelated (id INTEGER)")
        cx.commit()
    assert gt._read_db_token(db, "inbox_gmail") is None


def test_load_falls_through_to_file_when_db_table_missing(tmp_path, monkeypatch):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE unrelated (id INTEGER)")
        cx.commit()
    tokfile = tmp_path / "google-token.json"
    tokfile.write_text(_token_json(access="from-file"))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tokfile))
    loaded = gt.load_gmail_credentials(db, name="inbox_gmail", scopes=SCOPES)
    assert loaded.source == "file"


def test_self_heal_write_failure_is_best_effort(tmp_path, monkeypatch):
    # A transient self-heal write failure must not fail an otherwise-successful
    # load from the file fallback -- the design is best-effort.
    db = _db(tmp_path)
    tokfile = tmp_path / "google-token.json"
    tokfile.write_text(_token_json(access="from-file"))
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tokfile))

    def _boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(gt, "_write_db_token", _boom)
    loaded = gt.load_gmail_credentials(db, name="inbox_gmail", scopes=SCOPES)
    assert loaded.source == "file"
    assert loaded.creds.refresh_token == "1//refresh"
