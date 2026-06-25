"""End-to-end: /journal/today and /journal/history read from the local sqlite
store (re-homed from Supabase). Confirms the route wiring + metadata filtering."""
import importlib
import sqlite3
from datetime import datetime, timezone, timedelta

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import journal_blueprint as jb
    importlib.reload(jb)  # rebind LOG_DB to the temp DATA_DIR
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(jb.journal_bp)
    # seed entries directly via the store into the same LOG_DB
    from dashboard import journal_store as js
    now = datetime.now(timezone.utc)

    def rec(dt, **md):
        meta = {"test": False, "entry_type": "journal"}
        meta.update(md)
        return {"user_id": "glen", "recorded_at": dt.isoformat(), "duration_seconds": 5,
                "transcript": "x", "dominant_element": md.get("el", "Fire"),
                "top_emotions": [{"name": "Calmness", "score": 0.5}], "tcm_scores": {},
                "metadata": meta}
    with sqlite3.connect(jb.LOG_DB) as cx:
        js.insert(cx, rec(now - timedelta(hours=2), el="Metal"))
        js.insert(cx, rec(now - timedelta(hours=1), el="Wood"))         # most recent real
        js.insert(cx, rec(now - timedelta(minutes=30), test=True))      # test entry
        js.insert(cx, rec(now - timedelta(days=10), el="Water"))        # in 30d history
    return app.test_client()


def test_today_returns_most_recent_nontest(client):
    r = client.get("/journal/today")
    assert r.status_code == 200
    assert r.get_json()["dominant_element"] == "Wood"  # newest non-test in 24h


def test_today_includes_test_when_asked(client):
    r = client.get("/journal/today?include_test=true")
    assert r.get_json().get("metadata", {}).get("test") is True


def test_history_counts_and_excludes_test_by_default(client):
    j = client.get("/journal/history").get_json()
    assert j["test_count"] == 1
    assert all(not (e.get("metadata") or {}).get("test") for e in j["entries"])
    assert j["count"] == 3  # two recent + one 10-day, test excluded
