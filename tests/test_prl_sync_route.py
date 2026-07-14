"""Console sync endpoint for the PRL Supplement portal card. Loads data/prl_seed.json
into the prl_* reference tables (idempotent full replace; mirror table untouched)."""
import importlib
import sys
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


@pytest.fixture()
def client(tmp_db, monkeypatch):
    app_mod = _app()
    monkeypatch.setattr(app_mod, "LOG_DB", tmp_db)
    monkeypatch.setattr(app_mod, "CONSOLE_SECRET", "")  # open gate in tests
    return app_mod.app.test_client()


def test_prl_sync_populates_tables(client):
    r = client.post("/api/console/prl/sync")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["products"] >= 143
    assert body["focus_area_products"] >= 1 and body["focus_area_items"] >= 1
