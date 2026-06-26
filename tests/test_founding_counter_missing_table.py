"""Regression: the public founding counter must not 500 on a database that has
never had a subscription written (the `subscriptions` table is created lazily on
first write). count_founding treats a missing table as 0 reservations, so the
status route and the reserve open-check stay alive on a fresh prod DB."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

import dashboard.subscriptions as subs


def test_count_founding_returns_zero_when_table_missing():
    cx = sqlite3.connect(":memory:")  # no tables at all
    assert subs.count_founding(cx, "neuro-magnesium") == 0


def test_count_founding_counts_when_table_present():
    cx = sqlite3.connect(":memory:")
    subs.init_subscriptions_table(cx)
    subs.migrate_add_founding_columns(cx)
    assert subs.count_founding(cx, "neuro-magnesium") == 0  # table exists, no rows


def _load_app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_founding_status_route_alive_without_subscriptions_table(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)  # journey only, NO subscriptions table
    c = app_module.app.test_client()
    r = c.get("/begin/founding/status/neuro-magnesium")
    assert r.status_code == 200, r.get_data(as_text=True)[:200]
    body = r.get_json()
    assert body["cap"] == 2500
    assert body["remaining"] == 2500   # 0 reserved -> full cap
    assert body["open"] is True
