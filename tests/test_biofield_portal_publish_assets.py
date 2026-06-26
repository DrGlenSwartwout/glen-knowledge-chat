# tests/test_biofield_portal_publish_assets.py
import re
import sqlite3
import pytest
from dashboard import biofield_portal_publish as bpp
from dashboard.biofield_authoring import create_test, add_chain_row

CATALOG = {"vitality": {"name": "Vitality"}}

class _Resp:
    def __init__(self, status, body):
        self.status_code = status; self._b = body
        import json as _j; self.text = _j.dumps(body)
    def json(self): return self._b

def test_asset_name_opaque_and_unique():
    a = bpp._asset_name("mp3"); b = bpp._asset_name("mp3")
    assert re.match(r'^biofield-[0-9a-f]{16}\.mp3$', a)
    assert bpp._asset_name("pdf").endswith(".pdf")
    assert a != b

def test_upload_asset_puts_bytes_with_key_and_returns_url():
    cap = {}
    def fake_put(url, data=None, headers=None, timeout=None):
        cap["url"] = url; cap["data"] = data; cap["headers"] = headers
        return _Resp(200, {"ok": True, "url": "https://h/portal-asset/biofield-x.pdf"})
    out = bpp.upload_asset(b"PDFBYTES", "biofield-x.pdf",
                           base_url="https://h", console_key="secret", http_put=fake_put)
    assert out == "https://h/portal-asset/biofield-x.pdf"
    assert cap["url"] == "https://h/portal-asset/upload?filename=biofield-x.pdf"
    assert cap["data"] == b"PDFBYTES"
    assert cap["headers"]["X-Console-Key"] == "secret"

def test_upload_asset_raises_on_non_2xx():
    def fake_put(url, data=None, headers=None, timeout=None):
        return _Resp(401, {"error": "unauthorized"})
    with pytest.raises(RuntimeError):
        bpp.upload_asset(b"x", "biofield-x.pdf", base_url="https://h",
                         console_key="bad", http_put=fake_put)

def test_build_content_includes_audio_and_pdf_when_urls_given():
    cx = sqlite3.connect(":memory:")
    tid = create_test(cx, "K", "k@example.com", "2026-06-25")
    add_chain_row(cx, f"a{tid}", layer=1, head="ED3", most_affected="C",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    out = bpp.build_portal_content(cx, f"a{tid}", special_price_cents=5000, catalog=CATALOG,
                                   audio_url="https://h/portal-asset/a.mp3",
                                   report_pdf_url="https://h/portal-asset/r.pdf")
    c = out["content"]
    assert c["audio"] == {"url": "https://h/portal-asset/a.mp3", "label": "Listen to your walkthrough"}
    assert c["report_pdf"] == {"url": "https://h/portal-asset/r.pdf"}

def test_build_content_omits_audio_pdf_when_not_given():
    cx = sqlite3.connect(":memory:")
    tid = create_test(cx, "K", "k@example.com", "2026-06-25")
    add_chain_row(cx, f"a{tid}", layer=1, head="ED3", most_affected="C",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    c = bpp.build_portal_content(cx, f"a{tid}", special_price_cents=5000, catalog=CATALOG)["content"]
    assert "audio" not in c and "report_pdf" not in c

def test_publish_send_param_controls_body():
    cap = {}
    def fake_post(url, json=None, headers=None, timeout=None):
        cap["send"] = json.get("send"); return _Resp(200, {"ok": True, "url": "u"})
    bpp.publish_to_portal({"email": "k@example.com"}, base_url="https://h",
                          console_key="s", send=True, http_post=fake_post)
    assert cap["send"] is True
    bpp.publish_to_portal({"email": "k@example.com"}, base_url="https://h",
                          console_key="s", http_post=fake_post)
    assert cap["send"] is False
