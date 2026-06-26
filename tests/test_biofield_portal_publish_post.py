import json
import pytest
from dashboard import biofield_portal_publish as bpp

class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)
    def json(self):
        return self._body

def test_publish_posts_with_key_and_send_false_and_returns_json():
    captured = {}
    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _Resp(200, {"ok": True, "url": "https://illtowell.com/portal/abc",
                           "token": "abc"})
    out = bpp.publish_to_portal(
        {"email": "k@example.com", "name": "K", "content": {}, "scan_date": "2026-06-25"},
        base_url="https://illtowell.com", console_key="secret", http_post=fake_post)
    assert out["url"] == "https://illtowell.com/portal/abc"
    assert captured["url"] == "https://illtowell.com/admin/portal/upsert"
    assert captured["headers"]["X-Console-Key"] == "secret"
    assert captured["json"]["send"] is False
    assert captured["json"]["email"] == "k@example.com"

def test_publish_raises_on_non_2xx():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _Resp(401, {"error": "unauthorized"})
    with pytest.raises(RuntimeError):
        bpp.publish_to_portal({"email": "k@example.com"}, base_url="https://x",
                              console_key="bad", http_post=fake_post)
