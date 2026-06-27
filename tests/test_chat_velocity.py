"""Tests for per-IP velocity limiting on the three chat endpoints.

Design:
  Integration tests — prove the guard is wired into each real endpoint: exhaust
  the per-IP budget (per_min=2) via direct _velocity_guard calls inside a
  test_request_context (zero DB I/O), then POST to the real endpoint for the
  blocking case only.  The guard fires before any DB/Pinecone work, so the POST
  returns 429 without touching query_log.

  Unit tests — verify _velocity_guard and _resolve_chat_tier directly using
  test_request_context.  No chat pipeline runs, no tables needed.
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
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    app_module = _load_app()
    # Fresh limiter — no prior hits
    monkeypatch.setattr(app_module, "_chat_velocity", VelocityLimiter())
    # Tighten the per_min limit so tests don't need to hammer 10+ requests
    monkeypatch.setattr(app_module, "LIMITS", _TIGHT_LIMITS)
    return app_module


def _exhaust_budget(velocity_app, ip, n=2, tier="anonymous"):
    """Fire n direct _velocity_guard calls for ip without touching any DB."""
    with velocity_app.app.test_request_context(
        "/chat", headers={"X-Forwarded-For": ip}
    ):
        from flask import request as _rq
        for i in range(n):
            result = velocity_app._velocity_guard(_rq, tier)
            assert result is None, f"Guard unexpectedly blocked hit {i + 1}/{n}"


# ── Integration: over-limit → 429 (guard fires BEFORE handler body) ──────────
# Budget exhausted via direct guard calls; only the (per_min+1)th request is a
# real HTTP POST — it gets 429 before the handler body runs.

def test_chat_blocks_on_over_limit(velocity_app):
    """Guard is wired into /chat: (per_min+1)th request from same IP → 429."""
    ip = "203.0.113.1"
    _exhaust_budget(velocity_app, ip)  # 2 hits consumed
    client = velocity_app.app.test_client()
    r = client.post(
        "/chat",
        json={"query": "hi", "mode": "brief"},
        content_type="application/json",
        headers={"X-Forwarded-For": ip},
    )
    assert r.status_code == 429
    body = r.get_json()
    assert body is not None and body.get("error") == "rate_limited"


def test_begin_match_chat_blocks_on_over_limit(velocity_app):
    """Guard is wired into /begin/match/chat: (per_min+1)th request → 429."""
    ip = "203.0.113.2"
    _exhaust_budget(velocity_app, ip)
    client = velocity_app.app.test_client()
    r = client.post(
        "/begin/match/chat",
        json={"query": "hi", "mode": "brief"},
        content_type="application/json",
        headers={"X-Forwarded-For": ip},
    )
    assert r.status_code == 429
    body = r.get_json()
    assert body is not None and body.get("error") == "rate_limited"


def test_begin_concierge_chat_blocks_on_over_limit(velocity_app):
    """Guard is wired into /begin/concierge/chat: (per_min+1)th request → 429."""
    ip = "203.0.113.3"
    _exhaust_budget(velocity_app, ip)
    client = velocity_app.app.test_client()
    r = client.post(
        "/begin/concierge/chat",
        json={"query": "hi", "mode": "brief"},
        content_type="application/json",
        headers={"X-Forwarded-For": ip},
    )
    assert r.status_code == 429
    body = r.get_json()
    assert body is not None and body.get("error") == "rate_limited"


# ── Unit tests: _velocity_guard via test_request_context (no chat pipeline) ──

def test_velocity_guard_under_limit_returns_none(velocity_app):
    """First request under the per_min budget is not blocked (returns None)."""
    with velocity_app.app.test_request_context(
        "/chat", headers={"X-Forwarded-For": "203.0.113.10"}
    ):
        from flask import request as _rq
        assert velocity_app._velocity_guard(_rq, "anonymous") is None


def test_velocity_guard_over_limit_returns_429_tuple(velocity_app):
    """After per_min hits from the same IP, guard returns a (response, 429) tuple."""
    with velocity_app.app.test_request_context(
        "/chat", headers={"X-Forwarded-For": "203.0.113.11"}
    ):
        from flask import request as _rq
        # per_min=2: first two pass
        assert velocity_app._velocity_guard(_rq, "anonymous") is None
        assert velocity_app._velocity_guard(_rq, "anonymous") is None
        # third is blocked
        result = velocity_app._velocity_guard(_rq, "anonymous")
        assert result is not None, "Guard should block after per_min exceeded"
        assert result[1] == 429


def test_velocity_guard_shared_counter_across_paths(velocity_app):
    """The IP budget is shared regardless of which path the context uses."""
    ip = "203.0.113.12"
    with velocity_app.app.test_request_context(
        "/chat", headers={"X-Forwarded-For": ip}
    ):
        from flask import request as _rq
        assert velocity_app._velocity_guard(_rq, "anonymous") is None  # hit 1

    with velocity_app.app.test_request_context(
        "/begin/match/chat", headers={"X-Forwarded-For": ip}
    ):
        from flask import request as _rq
        assert velocity_app._velocity_guard(_rq, "anonymous") is None  # hit 2

    with velocity_app.app.test_request_context(
        "/begin/concierge/chat", headers={"X-Forwarded-For": ip}
    ):
        from flask import request as _rq
        result = velocity_app._velocity_guard(_rq, "anonymous")  # hit 3 — blocked
        assert result is not None and result[1] == 429


def test_velocity_guard_different_ips_independent(velocity_app):
    """Two different IPs have independent counters; exhausting one doesn't block the other."""
    with velocity_app.app.test_request_context(
        "/chat", headers={"X-Forwarded-For": "203.0.113.20"}
    ):
        from flask import request as _rq
        # Exhaust IP .20
        velocity_app._velocity_guard(_rq, "anonymous")
        velocity_app._velocity_guard(_rq, "anonymous")
        velocity_app._velocity_guard(_rq, "anonymous")  # blocked

    with velocity_app.app.test_request_context(
        "/chat", headers={"X-Forwarded-For": "203.0.113.21"}
    ):
        from flask import request as _rq
        # IP .21 is fresh — should not be blocked
        assert velocity_app._velocity_guard(_rq, "anonymous") is None


# ── Unit test: _resolve_chat_tier fail-open ───────────────────────────────────

def test_resolve_chat_tier_unauthenticated_returns_anonymous(velocity_app):
    """Unauthenticated request resolves to ('anonymous', '') without raising."""
    with velocity_app.app.test_request_context("/chat"):
        from flask import request as _rq
        tier, eff_email = velocity_app._resolve_chat_tier(_rq, "", "")
        assert tier == "anonymous"
        assert eff_email == ""
