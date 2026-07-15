"""POST /api/portal/<token>/life-stress/selection — the client SAVE endpoint
(Task 3). Saves a preference only -- NO order, NO invoice, NO Stripe/QBO.

Mirrors tests/test_life_stress_payload.py's real-portal-token setup.
  - flag off               -> 404
  - flag on, submitted      -> only pool slugs (from _life_stress_for) are kept;
                               response `saved` and the persisted row both
                               reflect the filtered list (server-side pool-filter
                               is the security guard: a client can only prefer
                               what they were actually offered).
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_portal as cp
from dashboard import life_stress_selection as sel

EMAIL = "lsclient@example.com"

POOL = {
    "label": "Life Stress",
    "patterns": [{"emotion": "Fear", "score": 1.0}],
    "items": [
        {"slug": "a-in-terrain-restore", "name": "A",
         "url": "/begin/product/a-in-terrain-restore", "note": "n"},
        {"slug": "b-in-terrain-restore", "name": "B",
         "url": "/begin/product/b-in-terrain-restore", "note": "n"},
    ],
}


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def app_env(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.delenv("LIFE_STRESS_ENABLED", raising=False)
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        token, _pid = cp.upsert_portal(cx, EMAIL, "Client", {})
    client = app.app.test_client()
    return app, client, token


def test_saves_only_pool_slugs(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setattr(app, "_life_stress_enabled", lambda: True)
    monkeypatch.setattr(app, "_life_stress_for", lambda e: dict(POOL))

    resp = client.post(f"/api/portal/{token}/life-stress/selection",
                        json={"slugs": ["a-in-terrain-restore", "evil-not-in-pool"]})
    assert resp.status_code == 200
    j = resp.get_json()
    assert j["ok"] is True
    assert j["saved"] == ["a-in-terrain-restore"]

    with sqlite3.connect(app.LOG_DB) as cx:
        assert sel.get(cx, EMAIL) == ["a-in-terrain-restore"]


def test_flag_off_404(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setattr(app, "_life_stress_enabled", lambda: False)

    resp = client.post(f"/api/portal/{token}/life-stress/selection",
                        json={"slugs": ["a-in-terrain-restore"]})
    assert resp.status_code == 404
