"""Tests for reply_watcher's wiring onto the durable Gmail token loader.

Covers: process_inbox_replies loads creds via dashboard.gmail_token when no
svc is injected, persists any refresh, and records token health at the end
of a run. When a svc is injected directly (existing callers/tests), the
token loader must not be touched at all.
"""

import sqlite3
import types
import pytest
import reply_watcher as rw
from dashboard import gmail_token as gt


def _db(tmp_path):
    p = tmp_path / "chat_log.db"
    with sqlite3.connect(p) as cx:
        cx.execute("CREATE TABLE oauth_tokens (name TEXT PRIMARY KEY, "
                   "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        cx.commit()
    return str(p)


def test_process_persists_and_records_ok(tmp_path, monkeypatch):
    db = _db(tmp_path)
    # Fake loader returns a fake creds + LoadedGmail; avoid real Gmail/Google.
    fake_creds = types.SimpleNamespace(to_json=lambda: '{"token":"new"}')
    loaded = gt.LoadedGmail(creds=fake_creds, source="db",
                            original_json='{"token":"old"}', name="inbox_gmail")
    monkeypatch.setattr(rw, "_build_service_from_creds", lambda creds: object())
    monkeypatch.setattr(gt, "load_gmail_credentials", lambda *a, **k: loaded)
    persisted = {}
    monkeypatch.setattr(gt, "persist_refreshed_credentials",
                        lambda dbp, l: persisted.setdefault("hit", True))
    recorded = {}
    monkeypatch.setattr(gt, "record_ok",
                        lambda dbp, name: recorded.setdefault("hit", True))
    # Stub out the actual Gmail work: patch the label/list calls to no-op counts.
    monkeypatch.setattr(rw, "_ensure_label", lambda svc, name: "LBL")
    monkeypatch.setattr(rw, "_scan_and_process",
                        lambda svc, db_path, dry_run, max_messages,
                               processed_label_id, nonuser_label_id:
                        {"processed": 0, "skipped_nonuser": 0, "errored": 0, "details": []})
    counts = rw.process_inbox_replies(db_path=db, dry_run=True, max_messages=1)
    assert counts["processed"] == 0
    assert persisted.get("hit") is True
    assert recorded.get("hit") is True


def test_process_uses_injected_svc_without_token_load(tmp_path, monkeypatch):
    called = {"load": False}
    monkeypatch.setattr(gt, "load_gmail_credentials",
                        lambda *a, **k: called.__setitem__("load", True))
    monkeypatch.setattr(rw, "_ensure_label", lambda svc, name: "LBL")
    monkeypatch.setattr(rw, "_scan_and_process",
                        lambda *a, **k: {"processed": 0, "skipped_nonuser": 0,
                                          "errored": 0, "details": []})
    rw.process_inbox_replies(svc=object(), db_path=str(tmp_path / "x.db"))
    assert called["load"] is False  # injected svc bypasses token load
