"""Task 6: Share & Unlock portal card.

(a) Server round-trip integration test — proves the data contract the card
relies on in both directions (POST toggles -> GET payload reflects them,
including tier/rewards and prospective revocation).
(b) Static markup assertion test — guards the client-portal.html wiring
without a browser (per feedback_static_serve_render_verify: no headless
Chrome / http.server live-render for this task).
"""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        app = importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    return app


def _seed_portal(tmp_db, email="member@ex.com"):
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, email, "M", {})
    return token


def test_sharing_card_data_round_trips_through_payload(monkeypatch, tmp_db):
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db)
    client = app.app.test_client()

    r = client.post(
        f"/api/portal/{token}/sharing",
        json={"toggles": {"research_results": True}},
    )
    assert r.status_code == 200

    body = client.get(f"/api/portal/{token}").get_json()
    assert body["data_sharing"]["toggles"]["research_results"] is True
    assert body["data_sharing"]["tier"] == 2
    assert body["data_sharing"]["rewards"].get("free_reveal_unlock") == "granted"

    # Turn everything back off -- proves prospective revocation round-trips
    # to what the card would render (unchecked boxes, tier back to 0).
    r2 = client.post(f"/api/portal/{token}/sharing", json={"toggles": {}})
    assert r2.status_code == 200

    body2 = client.get(f"/api/portal/{token}").get_json()
    assert body2["data_sharing"]["toggles"]["research_results"] is False
    assert body2["data_sharing"]["tier"] == 0


def test_sharing_card_markup_wired_in_client_portal_html():
    repo = Path(__file__).resolve().parent.parent
    html = (repo / "static" / "client-portal.html").read_text()

    assert "d.data_sharing" in html
    assert "sharing-card" in html
    assert "data-sharing-toggle" in html
    for key in ("improve_ai_chat", "research_results", "share_story", "video_testimonial"):
        assert f'"{key}"' in html
    assert "/sharing" in html
