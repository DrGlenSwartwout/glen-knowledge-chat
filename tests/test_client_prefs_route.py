"""HTTP-level coverage for /api/console/client-prefs (GET + POST).

Imports app (needs real secrets + writable DATA_DIR), so it's skipped under
plain pytest and runs under the Doppler harness:
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 -m pytest tests/test_client_prefs_route.py

Replaces the hollow test_the_route_still_accepts_an_explicit_false (a
substring-in-source assertion in test_client_prefs.py) with a real request
through app.app.test_client(). That test only checked that the literal text
'bool(body.get("pickup_default"))' appeared somewhere in the route source —
it kept passing even when a reviewer broke the guard to
`if "pickup_default" not in body or not body.get("pickup_default"):`, which
wrongly rejects an explicit false. This file drives the actual route so a
regression like that fails a real assertion (see test_post_false_flips_pickup_off).
"""
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import app
except Exception as _e:  # pragma: no cover - exercised only under plain pytest
    pytest.skip(f"app import requires real secrets: {_e}", allow_module_level=True)


OWNER_KEY = "X-Console-Key"


def _owner_client(monkeypatch, tmp_path):
    """A test client authenticated as OWNER, pointed at a scratch sqlite db."""
    monkeypatch.setattr(app, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(
        app, "_bos_actor",
        lambda: app._bos_rbac.Actor(role=app._bos_rbac.OWNER, name="test-owner"))
    return app.app.test_client()


def _va_client(monkeypatch, tmp_path):
    """A test client authenticated as a non-OWNER (VA) actor."""
    monkeypatch.setattr(app, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(
        app, "_bos_actor",
        lambda: app._bos_rbac.Actor(role=app._bos_rbac.VA, name="test-va"))
    return app.app.test_client()


def test_post_true_sets_pickup_and_get_reflects_it(monkeypatch, tmp_path):
    client = _owner_client(monkeypatch, tmp_path)

    r = client.post("/api/console/client-prefs",
                    json={"email": "bobbi@x.com", "pickup_default": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["pickup_default"] is True

    r = client.get("/api/console/client-prefs?email=bobbi@x.com")
    assert r.status_code == 200
    assert r.get_json()["pickup_default"] is True


def test_post_false_flips_pickup_off(monkeypatch, tmp_path):
    """The assertion the hollow test failed to make: an explicit
    {"pickup_default": false} must actually flip the preference off, not be
    treated as a rejected/missing key."""
    client = _owner_client(monkeypatch, tmp_path)

    r = client.post("/api/console/client-prefs",
                    json={"email": "bobbi@x.com", "pickup_default": True})
    assert r.status_code == 200
    assert r.get_json()["pickup_default"] is True

    r = client.post("/api/console/client-prefs",
                    json={"email": "bobbi@x.com", "pickup_default": False})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["pickup_default"] is False

    r = client.get("/api/console/client-prefs?email=bobbi@x.com")
    assert r.status_code == 200
    assert r.get_json()["pickup_default"] is False


def test_post_missing_pickup_default_key_is_rejected_and_leaves_value_unchanged(monkeypatch, tmp_path):
    client = _owner_client(monkeypatch, tmp_path)

    r = client.post("/api/console/client-prefs",
                    json={"email": "bobbi@x.com", "pickup_default": True})
    assert r.status_code == 200
    assert r.get_json()["pickup_default"] is True

    r = client.post("/api/console/client-prefs", json={"email": "bobbi@x.com"})
    assert r.status_code == 400
    assert "pickup_default" in r.get_json()["error"]

    r = client.get("/api/console/client-prefs?email=bobbi@x.com")
    assert r.status_code == 200
    assert r.get_json()["pickup_default"] is True


def test_post_missing_email_is_rejected(monkeypatch, tmp_path):
    client = _owner_client(monkeypatch, tmp_path)

    r = client.post("/api/console/client-prefs", json={"pickup_default": True})
    assert r.status_code == 400
    assert "email" in r.get_json()["error"]

    r = client.post("/api/console/client-prefs",
                    json={"email": "  ", "pickup_default": True})
    assert r.status_code == 400
    assert "email" in r.get_json()["error"]


def test_get_missing_email_is_rejected(monkeypatch, tmp_path):
    client = _owner_client(monkeypatch, tmp_path)

    r = client.get("/api/console/client-prefs")
    assert r.status_code == 400
    assert "email" in r.get_json()["error"]


def test_non_owner_actor_is_rejected_on_both_verbs(monkeypatch, tmp_path):
    client = _va_client(monkeypatch, tmp_path)

    r = client.post("/api/console/client-prefs",
                    json={"email": "bobbi@x.com", "pickup_default": True})
    assert r.status_code == 401
    assert r.get_json()["ok"] is False

    r = client.get("/api/console/client-prefs?email=bobbi@x.com")
    assert r.status_code == 401
    assert r.get_json()["ok"] is False
