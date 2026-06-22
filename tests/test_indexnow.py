import sys
from pathlib import Path

import pytest


def _ensure_path():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)


def _in():
    _ensure_path()
    try:
        from dashboard import indexnow
        return indexnow
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"indexnow not importable: {e}")


class _FakeHTTP:
    def __init__(self):
        self.calls = []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        return None


def test_submit_noops_without_key(monkeypatch):
    inx = _in()
    monkeypatch.delenv("INDEXNOW_KEY", raising=False)
    http = _FakeHTTP()
    assert inx.submit("https://illtowell.com/learn/x", http=http) is False
    assert http.calls == []


def test_submit_pings_with_key():
    inx = _in()
    http = _FakeHTTP()
    ok = inx.submit("https://illtowell.com/learn/low-energy",
                    base_url="https://illtowell.com", k="ABC123", http=http)
    assert ok is True
    assert len(http.calls) == 1
    url, kw = http.calls[0]
    assert "indexnow.org" in url
    body = kw.get("json") or {}
    assert body.get("host") == "illtowell.com"
    assert body.get("key") == "ABC123"
    assert body.get("urlList") == ["https://illtowell.com/learn/low-energy"]
    assert body.get("keyLocation") == "https://illtowell.com/ABC123.txt"


def test_submit_never_raises_on_http_error():
    inx = _in()

    class Boom:
        def post(self, *a, **k):
            raise RuntimeError("network down")

    # must swallow the error and report failure, never propagate
    assert inx.submit("https://x.test/learn/y", k="K", http=Boom()) is False


def test_submit_skips_empty_url():
    inx = _in()
    http = _FakeHTTP()
    assert inx.submit("", k="K", http=http) is False
    assert http.calls == []
