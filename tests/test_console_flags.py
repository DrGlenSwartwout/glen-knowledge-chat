"""Report every *_ENABLED flag: the value the RUNNING PROCESS holds, and whether the
env var exists at all.

On 2026-07-09 three flags were DELETED from the prod service (a single-key GET returned
404 "not found"), not set false. A deleted var and a deliberate false make the app behave
identically but mean different things — REPERTOIRE_ENABLED missing silently charges paid
members more. That distinction is what `env_present` carries.

Two kinds of flag exist: 34 import-time module constants (can go stale — set the var,
skip the deploy, the app still behaves as if off) and 29 that exist only as call-time
os.environ reads. A globals scan finds none of the 29, so discovery is the UNION of
module globals and env keys.
"""
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


def test_import_flag_reports_global_not_env(monkeypatch):
    """An import-time flag's value is the GLOBAL, not os.environ. That is what makes
    'set true but never redeployed' detectable."""
    appmod = _app()
    monkeypatch.setattr(appmod, "FIRESIDE_ENABLED", False)
    monkeypatch.setenv("FIRESIDE_ENABLED", "true")      # env says on, process says off
    f = appmod._flag_report()["FIRESIDE_ENABLED"]
    assert f["value"] is False, "must report what the process holds"
    assert f["env_present"] is True
    assert f["source"] == "import"


def test_deleted_import_flag_reports_env_absent(monkeypatch):
    appmod = _app()
    monkeypatch.setattr(appmod, "FIRESIDE_ENABLED", False)
    monkeypatch.delenv("FIRESIDE_ENABLED", raising=False)
    f = appmod._flag_report()["FIRESIDE_ENABLED"]
    assert f["value"] is False
    assert f["env_present"] is False, "a deleted var must be distinguishable from a false one"


def test_runtime_flag_tracks_env_without_restart(monkeypatch):
    """SCAN_REQUEST_ENABLED has NO module global (app.py:14808 reads os.environ per
    call). It must still be reported, and its value follows the env immediately."""
    appmod = _app()
    assert not hasattr(appmod, "SCAN_REQUEST_ENABLED")
    monkeypatch.setenv("SCAN_REQUEST_ENABLED", "1")
    f = appmod._flag_report()["SCAN_REQUEST_ENABLED"]
    assert f["value"] is True
    assert f["source"] == "runtime"
    monkeypatch.setenv("SCAN_REQUEST_ENABLED", "off")
    assert appmod._flag_report()["SCAN_REQUEST_ENABLED"]["value"] is False


def test_deleted_runtime_flag_vanishes_from_the_report(monkeypatch):
    """It has no global and no env key, so it cannot appear. check_flags() then reports
    it via the absent-from-response rule — this is the SCAN_REQUEST_ENABLED deletion."""
    appmod = _app()
    monkeypatch.delenv("SCAN_REQUEST_ENABLED", raising=False)
    assert "SCAN_REQUEST_ENABLED" not in appmod._flag_report()


def test_only_enabled_keys_are_returned(monkeypatch):
    """CONSOLE_SECRET must not be able to leak through this endpoint."""
    appmod = _app()
    monkeypatch.setenv("CONSOLE_SECRET", "super-secret")
    monkeypatch.setenv("NOT_A_FLAG", "true")
    report = appmod._flag_report()
    assert "CONSOLE_SECRET" not in report
    assert "NOT_A_FLAG" not in report
    assert all(k.endswith("_ENABLED") for k in report)


def test_values_are_booleans_never_strings(monkeypatch):
    appmod = _app()
    monkeypatch.setenv("SCAN_REQUEST_ENABLED", "yes")
    for info in appmod._flag_report().values():
        assert isinstance(info["value"], bool)
        assert isinstance(info["env_present"], bool)
        assert info["source"] in ("import", "runtime")


def test_a_new_global_is_discovered_without_registration(monkeypatch):
    appmod = _app()
    monkeypatch.setattr(appmod, "BRAND_NEW_ENABLED", True, raising=False)
    assert appmod._flag_report()["BRAND_NEW_ENABLED"]["source"] == "import"


def test_route_requires_the_console_key(monkeypatch):
    appmod = _app()
    monkeypatch.setattr(appmod.dashboard, "CONSOLE_SECRET", "k")
    appmod.app.config["TESTING"] = True
    c = appmod.app.test_client()
    assert c.get("/api/console/flags").status_code == 401
    r = c.get("/api/console/flags", headers={"X-Console-Key": "k"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert "FIRESIDE_ENABLED" in body["data"]["flags"]
