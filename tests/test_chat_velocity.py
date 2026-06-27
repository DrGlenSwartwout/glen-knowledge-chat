"""Integration tests for per-IP velocity limiting on the three chat endpoints.

We monkeypatch _chat_velocity with a fresh VelocityLimiter and tighten LIMITS
to per_min=2 so a third request from the same IP hits the 429 guard without
needing real Claude/Pinecone calls.

The guard fires before any retrieval/streaming, so an empty or minimal POST
body is fine — the handler returns 429 before it ever calls embed() or Claude.
"""
import importlib
import sys
from pathlib import Path

import pytest

from dashboard.chat_limits import VelocityLimiter


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable in this env: {e}")


_TIGHT_LIMITS = {
    "anonymous":  {"per_min": 2, "per_day": 40,  "monthly_full_words": None, "flag_full_words": None},
    "registered": {"per_min": 2, "per_day": 60,  "monthly_full_words": None, "flag_full_words": None},
    "member":     {"per_min": 2, "per_day": 150, "monthly_full_words": None, "flag_full_words": None},
}


@pytest.fixture
def velocity_app(monkeypatch, tmp_path):
    """App with a fresh velocity limiter and tight per_min=2 limits."""
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    # Fresh limiter — no prior hits
    monkeypatch.setattr(app_module, "_chat_velocity", VelocityLimiter())
    # Tighten the per_min limit so tests don't need to hammer 10+ requests
    monkeypatch.setattr(app_module, "LIMITS", _TIGHT_LIMITS)
    return app_module


def _post_chat(client, path="/chat", body=None):
    return client.post(
        path,
        json=body or {"query": "hi", "mode": "brief"},
        content_type="application/json",
    )


# ── /chat ────────────────────────────────────────────────────────────────────

def test_chat_blocks_third_request_same_ip(velocity_app):
    """Two requests pass, the third hits the 429 guard (per_min=2)."""
    client = velocity_app.app.test_client()

    r1 = _post_chat(client, "/chat")
    r2 = _post_chat(client, "/chat")
    # First two must NOT be 429
    assert r1.status_code != 429, f"First request unexpectedly 429: {r1.get_data(as_text=True)}"
    assert r2.status_code != 429, f"Second request unexpectedly 429: {r2.get_data(as_text=True)}"

    r3 = _post_chat(client, "/chat")
    assert r3.status_code == 429
    body = r3.get_json()
    assert body is not None and body.get("error") == "rate_limited"


def test_chat_under_limit_not_blocked(velocity_app):
    """A single request should never be 429 (well under per_min=2)."""
    client = velocity_app.app.test_client()
    r = _post_chat(client, "/chat")
    assert r.status_code != 429


# ── /begin/match/chat ────────────────────────────────────────────────────────

def test_begin_match_chat_blocks_third_request(velocity_app):
    """/begin/match/chat also rate-limits at per_min=2."""
    client = velocity_app.app.test_client()

    r1 = _post_chat(client, "/begin/match/chat")
    r2 = _post_chat(client, "/begin/match/chat")
    assert r1.status_code != 429
    assert r2.status_code != 429

    r3 = _post_chat(client, "/begin/match/chat")
    assert r3.status_code == 429
    body = r3.get_json()
    assert body is not None and body.get("error") == "rate_limited"


# ── /begin/concierge/chat ────────────────────────────────────────────────────

def test_begin_concierge_chat_blocks_third_request(velocity_app):
    """/begin/concierge/chat also rate-limits at per_min=2."""
    client = velocity_app.app.test_client()

    r1 = _post_chat(client, "/begin/concierge/chat")
    r2 = _post_chat(client, "/begin/concierge/chat")
    assert r1.status_code != 429
    assert r2.status_code != 429

    r3 = _post_chat(client, "/begin/concierge/chat")
    assert r3.status_code == 429
    body = r3.get_json()
    assert body is not None and body.get("error") == "rate_limited"


# ── cross-endpoint isolation (velocity is per-IP across all three) ───────────

def test_velocity_shared_across_endpoints(velocity_app):
    """Hits are shared by IP across all three endpoints — two requests on
    /chat and /begin/match/chat exhaust the per_min=2 budget; a third on
    /begin/concierge/chat is blocked."""
    client = velocity_app.app.test_client()

    _post_chat(client, "/chat")
    _post_chat(client, "/begin/match/chat")

    r = _post_chat(client, "/begin/concierge/chat")
    assert r.status_code == 429
    assert (r.get_json() or {}).get("error") == "rate_limited"
