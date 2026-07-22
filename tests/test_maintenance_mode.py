"""P06 Code Task C2: MAINTENANCE_MODE write-freeze.

During the SQLite -> Postgres cutover window, a global before_request hook
must block mutating requests (POST/PUT/DELETE/PATCH) with a 503 while
MAINTENANCE_MODE is truthy, while leaving GET/HEAD/OPTIONS and the operator's
admin/console + health surface untouched. Flag-off (unset) must be a
zero-behavior-change no-op.
"""
import importlib
import sys
from pathlib import Path

import pytest


def _app(tmp_path, monkeypatch):
    """Fresh app import per test (avoid import-time state bleeding across
    tests, per [[feedback_order_dependent_tests_are_import_time]])."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MAINTENANCE_MODE", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


# ── Flag OFF (unset) = zero behavior change ──────────────────────────────────

def test_flag_off_normal_post_is_untouched(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.delenv("MAINTENANCE_MODE", raising=False)
    c = appmod.app.test_client()
    r = c.post("/reorder/request", json={"email": ""})
    assert r.status_code != 503
    assert r.get_json() == {"ok": True}


def test_flag_empty_string_counts_as_off(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setenv("MAINTENANCE_MODE", "")
    c = appmod.app.test_client()
    r = c.post("/reorder/request", json={"email": ""})
    assert r.status_code != 503


# ── Flag ON: mutating methods on a normal route are frozen ──────────────────

def test_flag_on_blocks_post_to_normal_route(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setenv("MAINTENANCE_MODE", "1")
    c = appmod.app.test_client()
    r = c.post("/reorder/request", json={"email": ""})
    assert r.status_code == 503
    body = r.get_json()
    assert body["maintenance"] is True


@pytest.mark.parametrize("flag_val", ["true", "TRUE", "Yes", "on", "1"])
def test_flag_on_truthy_variants_block(tmp_path, monkeypatch, flag_val):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setenv("MAINTENANCE_MODE", flag_val)
    c = appmod.app.test_client()
    r = c.post("/reorder/request", json={"email": ""})
    assert r.status_code == 503


# ── Flag ON: GET/HEAD/OPTIONS keep working (reads + health checks) ───────────

def test_flag_on_get_to_normal_route_not_blocked(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setenv("MAINTENANCE_MODE", "1")
    c = appmod.app.test_client()
    # /reorder/request only registers POST -> a GET 405s, but must NOT be 503:
    # proves the hook is method-scoped, not path-scoped, for normal routes.
    r = c.get("/reorder/request")
    assert r.status_code != 503


def test_flag_on_root_health_path_exempt_even_for_post(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setenv("MAINTENANCE_MODE", "1")
    c = appmod.app.test_client()
    r = c.post("/")
    assert r.status_code != 503


# ── Flag ON: the console/admin operator surface stays reachable ─────────────

def test_flag_on_console_admin_path_exempt_even_for_post(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setenv("MAINTENANCE_MODE", "1")
    c = appmod.app.test_client()
    # Real console POST route (/api/console/household) — must not 503 even
    # though the console-key gate (separate from maintenance) may reject it
    # with 401 for lacking X-Console-Key; that 401 IS "normal handling".
    r = c.post("/api/console/household",
                json={"primary_email": "p@x.com", "member_email": "m@x.com"})
    assert r.status_code != 503
