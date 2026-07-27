"""Console sync endpoint for the Fullscript dispensary portal card. Loads
data/fullscript_seed.json into the fullscript_* reference tables (idempotent
full replace; client tables -- pins/review-links/clicks -- untouched).

Mirrors tests/test_prl_sync_route.py's fixture/pattern for the sibling PRL
channel."""
import importlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SEED_PATH = REPO / "data" / "fullscript_seed.json"


def _app():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _load_seed():
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def app_mod(tmp_db, monkeypatch):
    m = _app()
    monkeypatch.setattr(m, "LOG_DB", tmp_db)
    monkeypatch.setattr(m, "CONSOLE_SECRET", "")  # open gate in tests
    return m


@pytest.fixture()
def client(app_mod):
    return app_mod.app.test_client()


def test_fullscript_sync_populates_tables_from_committed_seed(client, app_mod, tmp_db):
    seed = _load_seed()
    r = client.post("/api/console/fullscript/sync")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["products"] == len(seed["products"])
    assert body["focus_area_products"] == len(seed["focus_area_products"])
    assert body["focus_area_items"] == len(seed["focus_area_items"])

    cx = sqlite3.connect(tmp_db)
    cx.row_factory = sqlite3.Row
    rows = cx.execute("SELECT * FROM fullscript_products").fetchall()
    assert len(rows) == len(seed["products"])
    by_name = {row["name"]: row for row in rows}
    assert "Magnesium Taurate" in by_name
    assert by_name["Magnesium Taurate"]["best_ff"] == "Neuro Magnesium"
    cx.close()


def test_fullscript_sync_is_console_gated_and_writes_nothing_when_unauthorized(
        client, app_mod, monkeypatch, tmp_db):
    monkeypatch.setattr(app_mod, "_portal_console_ok", lambda: False)
    r = client.post("/api/console/fullscript/sync")
    assert r.status_code == 401

    # Nothing written: the table shouldn't even exist yet since init_tables()
    # never ran (the handler returns before reaching db.connect at all).
    cx = sqlite3.connect(tmp_db)
    tables = {row[0] for row in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "fullscript_products" not in tables
    cx.close()


def test_fullscript_sync_then_scan_match_renders_card_with_ff_chip(
        client, app_mod, monkeypatch, tmp_db):
    """End-to-end: seed load -> scan match -> card -> best_ff chip renders.

    A client whose scan_recommendations has item_code "ED4" (which the
    committed seed maps to focus_area_id 9, "Nervous System") must surface
    Magnesium Taurate on their Fullscript card, with its ff chip showing
    "Neuro Magnesium" and a resolved (non-null) slug."""
    r = client.post("/api/console/fullscript/sync")
    assert r.status_code == 200

    from dashboard import scan_recommendations as _sr
    email = "e2e-fullscript@example.com"
    scan_date = "2026-07-20"
    with app_mod.db.connect(tmp_db) as cx:
        _sr.init_table(cx)
        n = _sr.replace_scan(
            cx, email, "scan-1", scan_date,
            [{"item_code": "ED4", "priority_rank": 1}])
    assert n == 1

    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")

    card = app_mod._fullscript_for(email, scan_date)
    assert card is not None, "expected a Fullscript card, got None"

    all_products = [p for g in card["groups"] for p in g["products"]]
    names = [p["name"] for p in all_products]
    assert "Magnesium Taurate" in names, f"Magnesium Taurate missing from card products: {names}"

    mag_taurate = next(p for p in all_products if p["name"] == "Magnesium Taurate")
    ff = mag_taurate["ff"]
    assert ff is not None, "Magnesium Taurate should carry a best_ff chip"
    assert ff["name"] == "Neuro Magnesium"
    assert ff["slug"] is not None, "ff chip slug must resolve, not be null"
