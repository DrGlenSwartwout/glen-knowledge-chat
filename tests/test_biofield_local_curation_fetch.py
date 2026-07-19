"""Unit tests for the local Biofield app's best-effort Life Stress curation fetch.

_fetch_life_stress_curation(email) mirrors the existing /api/people fetch pattern
(biofield_local_app._default_fetch_profile): X-Console-Key header, CONSOLE_SECRET +
PUBLIC_BASE_URL from env, short timeout, whole body wrapped so any failure (missing
secret, non-200, timeout, exception) yields None and never blocks the report.
"""
import json
import urllib.error

import pytest

from biofield_local_app import _fetch_life_stress_curation


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def _console_secret(monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "test-key")


def test_returns_curation_dict_on_success(monkeypatch):
    calls = {}

    def fake_urlopen(req, timeout=None):
        calls["url"] = req.full_url
        calls["headers"] = req.headers
        calls["timeout"] = timeout
        return _FakeResponse({"ok": True, "curation": {"slugs": ["x"], "note": "n"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = _fetch_life_stress_curation("A@B.com")

    assert result == {"slugs": ["x"], "note": "n"}
    assert "life-stress-curation" in calls["url"]
    assert "a%40b.com" in calls["url"].lower()
    assert calls["headers"].get("X-console-key") == "test-key"


def test_returns_none_on_exception(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert _fetch_life_stress_curation("a@b.com") is None


def test_returns_none_on_null_curation(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"ok": True, "curation": None})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert _fetch_life_stress_curation("a@b.com") is None


def test_blank_email_returns_none_without_network_call(monkeypatch):
    called = {"hit": False}

    def fake_urlopen(req, timeout=None):
        called["hit"] = True
        return _FakeResponse({"ok": True, "curation": {"slugs": []}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert _fetch_life_stress_curation("") is None
    assert _fetch_life_stress_curation(None) is None
    assert called["hit"] is False


def test_missing_console_secret_returns_none_without_network_call(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    called = {"hit": False}

    def fake_urlopen(req, timeout=None):
        called["hit"] = True
        return _FakeResponse({"ok": True, "curation": {"slugs": []}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert _fetch_life_stress_curation("a@b.com") is None
    assert called["hit"] is False
