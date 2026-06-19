import sqlite3
import pytest
from dashboard import sales_images as si
from dashboard import sales_image_prompts as sip
from dashboard import replicate_client as rc

def _cx(): return sqlite3.connect(":memory:")

class _Resp:
    def __init__(self, js=None, content=b"", status=200): self._js=js; self.content=content; self.status_code=status
    def json(self): return self._js
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError("http %d" % self.status_code)

def test_queue_enqueue_pending_done():
    cx = _cx()
    si.enqueue(cx, "longevity")
    assert si.list_pending(cx) == ["longevity"]
    assert si.queue_state(cx, "longevity") == "pending"
    si.mark_done(cx, "longevity")
    assert si.list_pending(cx) == []
    assert si.queue_state(cx, "longevity") == "done"

def test_enqueue_idempotent_resets_to_pending():
    cx = _cx()
    si.enqueue(cx, "energy"); si.mark_failed(cx, "energy")
    si.enqueue(cx, "energy")
    assert si.queue_state(cx, "energy") == "pending"

def test_record_and_display_first_ready_per_kind():
    cx = _cx()
    si.record_image(cx, "longevity", "botanical", 1, "botanical-1.png")
    si.record_image(cx, "longevity", "botanical", 2, "botanical-2.png")
    si.record_image(cx, "longevity", "mechanism", 1, "mechanism-1.png")
    disp = si.display_images(cx, "longevity")
    assert disp == {"botanical": "botanical-1.png", "mechanism": "mechanism-1.png"}
    assert len(si.get_images(cx, "longevity")) == 3

def test_prompts_two_modes_two_variants_each():
    p = sip.build_image_prompts({"name": "Longevity", "ingredients": [{"name": "Resveratrol"}]})
    assert set(p.keys()) == {"botanical", "mechanism"}
    assert len(p["botanical"]) == 2 and len(p["mechanism"]) == 2
    # variants within a kind are distinct
    assert p["botanical"][0] != p["botanical"][1]

def test_prompts_ground_in_ingredients_and_name():
    p = sip.build_image_prompts({"name": "Longevity", "ingredients": [{"name": "Resveratrol"}, "Quercetin"]})
    joined = " ".join(p["botanical"] + p["mechanism"])
    assert "Resveratrol" in joined and "Quercetin" in joined
    # botanical references the lifestyle scene; mechanism references the protective-field concept
    assert "kitchen" in p["botanical"][0].lower()
    assert "cell" in p["mechanism"][0].lower() or "field" in p["mechanism"][0].lower()

def test_generate_image_returns_bytes(monkeypatch):
    calls = {"post": 0, "get": 0}
    def fake_post(url, **kw):
        calls["post"] += 1
        return _Resp(js={"status": "succeeded", "output": "https://img/x.png", "urls": {"get": "https://api/get"}})
    def fake_get(url, **kw):
        calls["get"] += 1
        return _Resp(content=b"PNGBYTES")
    monkeypatch.setattr(rc.requests, "post", fake_post)
    monkeypatch.setattr(rc.requests, "get", fake_get)
    out = rc.generate_image("a prompt", token="tok")
    assert out == b"PNGBYTES" and calls["post"] == 1

def test_generate_image_raises_on_failed_status(monkeypatch):
    monkeypatch.setattr(rc.requests, "post", lambda url, **kw: _Resp(js={"status": "failed", "urls": {"get": "g"}}))
    with pytest.raises(Exception):
        rc.generate_image("p", token="tok")

def test_generate_image_requires_token(monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    with pytest.raises(Exception):
        rc.generate_image("p")
