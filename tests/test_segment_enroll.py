"""Backfill: enroll an opted-in segment into a GHL workflow via the write-queue."""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest


def _app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture
def app_db(monkeypatch, tmp_path):
    app = _app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    app._init_people_table()
    with sqlite3.connect(db) as cx:
        app._bos_ghl_queue.init_ghl_queue_table(cx)
        cx.commit()
    return app, db


def _seed(db, email, tags):
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO people (email, tags, created_at, updated_at) VALUES (?,?,?,?)",
                   (email, json.dumps(tags), "", ""))
        cx.commit()


def _queue(db):
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        return [dict(r) for r in cx.execute(
            "SELECT op,email,payload_json FROM ghl_write_queue WHERE status='pending'").fetchall()]


WF = "abc12345-aaaa-bbbb-cccc-000000000000"


def test_enrolls_only_opted_in_clients(app_db):
    app, db = app_db
    _seed(db, "client@x.com", ["type:client", "consent:opted-in"])
    _seed(db, "cold@x.com", ["type:client", "consent:cold-no-consent"])
    _seed(db, "unsub@x.com", ["type:client", "consent:opted-in", "consent:unsubscribed"])
    s = app.enroll_segment_in_workflow(WF)
    assert s["matched"] == 1 and s["enqueued"] == 1
    q = _queue(db)
    assert len(q) == 1 and q[0]["op"] == "workflow" and q[0]["email"] == "client@x.com"
    assert json.loads(q[0]["payload_json"])["workflow_id"] == WF


def test_dedup_on_rerun(app_db):
    app, db = app_db
    _seed(db, "c@x.com", ["type:client", "consent:opted-in"])
    app.enroll_segment_in_workflow(WF)
    s2 = app.enroll_segment_in_workflow(WF)
    assert s2["enqueued"] == 0 and s2["skipped_already"] == 1
    assert len(_queue(db)) == 1  # no duplicate enqueue


def test_cap_limits(app_db):
    app, db = app_db
    for i in range(4):
        _seed(db, f"c{i}@x.com", ["type:client", "consent:opted-in"])
    s = app.enroll_segment_in_workflow(WF, cap=2)
    assert s["enqueued"] == 2 and len(_queue(db)) == 2


def test_dry_run_writes_nothing(app_db):
    app, db = app_db
    _seed(db, "c@x.com", ["type:client", "consent:opted-in"])
    s = app.enroll_segment_in_workflow(WF, dry_run=True)
    assert s["enqueued"] == 1 and _queue(db) == []


def test_missing_workflow_id_errors(app_db):
    app, db = app_db
    assert app.enroll_segment_in_workflow("").get("error")
