"""POST /api/portal/<token>/ff-matches — the client-facing FF-match card (Slice 3b).

Flag-gated (FF_MATCHES_ENABLED). Generate-once via ff_match_drafts.get_or_create:
the same email/scan_date always returns the same items without re-calling the
(Pinecone-backed) generator. Animal clients get the scan's infoceuticals instead
of FF matches — a cat has no FF-product recommendation to review.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_portal as cp
from dashboard import client_species as cs
from dashboard import scan_recommendations as sr

EMAIL = "ffclient@example.com"
ITEMS = [
    {"item_code": "BFA", "priority_rank": 1, "protocol_days": 15,
     "section": "Infoceuticals", "category": "BFA", "label": "Big Field Aligner"},
    {"item_code": "ED6", "priority_rank": 2, "protocol_days": 15,
     "section": "Infoceuticals", "category": "ED", "label": "Heart"},
]


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
    monkeypatch.setenv("SCAN_RECOMMENDATIONS_ENABLED", "1")
    monkeypatch.delenv("FF_MATCHES_ENABLED", raising=False)
    monkeypatch.delenv("ANIMAL_GREETING_ENABLED", raising=False)
    with sqlite3.connect(tmp_db) as cx:
        cp.init_client_portal_table(cx)
        sr.init_table(cx)
        cs.init_table(cx)
        sr.replace_scan(cx, EMAIL, "10", "2026-07-02", ITEMS)
        token, _pid = cp.upsert_portal(cx, EMAIL, "Client", {})
    client = app.app.test_client()
    return app, client, token


def test_ff_matches_flag_off_404(app_env):
    app, client, token = app_env  # flag unset
    assert client.post(f"/api/portal/{token}/ff-matches").status_code == 404


def test_ff_matches_generate_once_and_cached(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    calls = []
    monkeypatch.setattr(
        app, "_make_ff_items_for",
        lambda e, d: (calls.append(1) or
                      [{"name": "X", "slug": "x", "url": "/begin/product/x",
                        "meaning": "m", "score": 0.9, "dosing": "2 caps daily"}]))
    a = client.post(f"/api/portal/{token}/ff-matches").get_json()["ff_matches"]
    b = client.post(f"/api/portal/{token}/ff-matches").get_json()["ff_matches"]
    assert calls == [1]                       # generate-once
    assert a["items"] == b["items"]
    assert all("dosing" not in it for it in a["items"])
    assert a["reviewed"] is False
    assert a["kind"] == "ff"


def test_animal_returns_infoceuticals_not_ff(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    monkeypatch.setenv("ANIMAL_GREETING_ENABLED", "1")
    with sqlite3.connect(app.LOG_DB) as cx:
        cs.upsert(cx, EMAIL, "Cat", "Sasha")   # token's client is an animal
    out = client.post(f"/api/portal/{token}/ff-matches").get_json()["ff_matches"]
    assert out["kind"] == "infoceutical"      # animals get the scan's infoceuticals
    # _scan_recommendations_for returns a DICT ({scan_date, scan_dates, infoceuticals,
    # mihealth}), not a list — the card's items must be flattened to a LIST of
    # {name, url, meaning} dicts, or every animal's card renders "no matches".
    assert isinstance(out["items"], list)
    assert out["items"]                        # scan recs were seeded for EMAIL (BFA, ED6)
    for it in out["items"]:
        assert isinstance(it, dict)
        assert it.get("name")                  # non-empty name


def test_unknown_token_404(app_env):
    app, client, token = app_env
    import os
    os.environ["FF_MATCHES_ENABLED"] = "1"
    try:
        assert client.post("/api/portal/not-a-real-token/ff-matches").status_code == 404
    finally:
        del os.environ["FF_MATCHES_ENABLED"]


def test_not_covered_published_draft_strips_dosing(app_env, monkeypatch):
    """Critical regression: a not-covered viewer of an already-PUBLISHED draft
    must still have dosing stripped. _ff_covered stays the default 3b stub
    (always False) — coverage never enters into it here; the leak was in
    treating "reviewed" alone as sufficient. Seeds the published draft directly
    via ff_match_drafts so get_or_create finds the existing row and does NOT
    regenerate it."""
    app, client, token = app_env
    monkeypatch.setenv("FF_MATCHES_ENABLED", "1")
    from dashboard import ff_match_drafts as ffd
    scan_date = app._current_scan_date_for(EMAIL)
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        ffd.init_table(cx)
        ffd.get_or_create(
            cx, EMAIL, scan_date,
            lambda: [{"name": "X", "slug": "x", "url": "/begin/product/x",
                      "meaning": "m", "score": 0.9, "dosing": "2 caps daily"}])
        ffd.publish(cx, EMAIL, scan_date)
    out = client.post(f"/api/portal/{token}/ff-matches").get_json()["ff_matches"]
    assert out["reviewed"] is True
    assert all("dosing" not in it for it in out["items"])
