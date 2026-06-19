import importlib
import sqlite3
import pathlib
import pytest
from dashboard import sales_images as si
from dashboard import sales_image_prompts as sip
from dashboard import replicate_client as rc


def _reload(monkeypatch, tmp_path, imgs="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_IMAGES", imgs)
    import app as appmod; importlib.reload(appmod); return appmod

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


def test_worker_generates_and_records(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    monkeypatch.setattr(appmod, "_product_card", lambda p: {"ingredients": [{"name": "Resveratrol"}]})
    from dashboard import replicate_client as rc
    monkeypatch.setattr(rc, "generate_image", lambda prompt, **kw: b"PNG")
    from dashboard import sales_images as si
    with sqlite3.connect(appmod.LOG_DB) as cx: si.enqueue(cx, slug)
    appmod._drain_sales_image_queue()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert si.queue_state(cx, slug) == "done"
        assert len(si.get_images(cx, slug)) == 4
    files = list((appmod._SALES_IMG_DIR / slug).glob("*.png"))
    assert len(files) == 4


def test_worker_flag_off_noop(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, imgs="false")
    assert appmod._SALES_AI_IMAGES_ENABLED is False
    from dashboard import sales_images as si
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    with sqlite3.connect(appmod.LOG_DB) as cx: si.enqueue(cx, slug)
    appmod._drain_sales_image_queue()  # flag off → no-op
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert si.queue_state(cx, slug) == "pending"


def test_worker_marks_failed_on_error(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    monkeypatch.setattr(appmod, "_product_card", lambda p: {"ingredients": []})
    from dashboard import replicate_client as rc
    def boom(prompt, **kw): raise RuntimeError("replicate down")
    monkeypatch.setattr(rc, "generate_image", boom)
    from dashboard import sales_images as si
    with sqlite3.connect(appmod.LOG_DB) as cx: si.enqueue(cx, slug)
    appmod._drain_sales_image_queue()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert si.queue_state(cx, slug) == "failed"
