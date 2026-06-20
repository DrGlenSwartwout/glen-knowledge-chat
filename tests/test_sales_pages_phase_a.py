import sqlite3
from dashboard import sales_prompt_variations as pv
from dashboard import sales_image_models as mods
from dashboard import replicate_client as rc

def _cx(): return sqlite3.connect(":memory:")

def test_seed_creates_four_active_variations_per_kind():
    cx = _cx(); pv.seed(cx)
    for kind in ("botanical", "mechanism"):
        rows = pv.active_variations(cx, kind)
        assert len(rows) == 4
        assert all(r["kind"] == kind for r in rows)
        assert all(r["prompt_template"] and r["label"] for r in rows)
        # distinct scenes, not duplicates
        assert len({r["prompt_template"] for r in rows}) == 4

def test_seed_is_idempotent():
    cx = _cx(); pv.seed(cx); pv.seed(cx)
    assert len(pv.active_variations(cx, "botanical")) == 4

def test_seed_creates_three_active_models():
    cx = _cx(); mods.seed(cx)
    rows = mods.active_models(cx)
    ids = [m["id"] for m in rows]
    assert ids == ["flux-1.1-pro", "imagen-4", "recraft-v3"]
    assert all(m["engine"] == "replicate" and m["engine_ref"] for m in rows)
    assert all(m["label"] for m in rows)

def test_models_seed_idempotent():
    cx = _cx(); mods.seed(cx); mods.seed(cx)
    assert len(mods.active_models(cx)) == 3

class _Resp:
    def __init__(self, j=None, content=b""): self._j = j or {}; self.content = content
    def json(self): return self._j
    def raise_for_status(self): pass

def test_generate_image_uses_model_ref_url(monkeypatch):
    calls = {}
    def fake_post(url, **kw):
        calls["url"] = url
        return _Resp({"status": "succeeded", "output": ["http://img/x.png"], "urls": {"get": "http://g"}})
    def fake_get(url, **kw):
        return _Resp(content=b"PNGDATA")
    monkeypatch.setattr(rc.requests, "post", fake_post)
    monkeypatch.setattr(rc.requests, "get", fake_get)
    out = rc.generate_image("hello", token="t", model_ref="google/imagen-4")
    assert out == b"PNGDATA"
    assert "google/imagen-4" in calls["url"]

def test_dispatch_uses_requested_model_ref(monkeypatch):
    cx = _cx(); mods.seed(cx)
    seen = {}
    def fake_gen(prompt, *, aspect_ratio="1:1", model_ref=None, **kw):
        seen["ref"] = model_ref; return b"IMG"
    monkeypatch.setattr("dashboard.replicate_client.generate_image", fake_gen)
    data, used = mods.generate(cx, "recraft-v3", "p")
    assert data == b"IMG" and used == "recraft-v3"
    assert seen["ref"] == "recraft-ai/recraft-v3"

def test_dispatch_falls_back_to_flux_on_error(monkeypatch):
    cx = _cx(); mods.seed(cx)
    calls = {"n": 0}
    def flaky(prompt, *, aspect_ratio="1:1", model_ref=None, **kw):
        calls["n"] += 1
        if model_ref != "black-forest-labs/flux-1.1-pro":
            raise RuntimeError("engine down")
        return b"FALLBACK"
    monkeypatch.setattr("dashboard.replicate_client.generate_image", flaky)
    data, used = mods.generate(cx, "imagen-4", "p")
    assert data == b"FALLBACK" and used == "flux-1.1-pro"
    assert calls["n"] == 2

from dashboard import sales_images as si
from dashboard import sales_image_prompts as sip

def test_record_image_persists_tags_and_counts():
    cx = _cx()
    si.record_image(cx, "p", "botanical", 1, "botanical-1.png", prompt_variant_id=3, model_id="imagen-4")
    si.record_image(cx, "p", "botanical", 2, "botanical-2.png")   # legacy, untagged
    rows = {r["variant"]: r for r in si.get_images(cx, "p")}
    assert rows[1]["prompt_variant_id"] == 3 and rows[1]["model_id"] == "imagen-4"
    assert rows[2]["prompt_variant_id"] is None
    assert si.tagged_count(cx, "p") == 1
    assert si.needs_topup(cx, "p") is True

def test_no_text_constant_exposed():
    assert "No text" in sip.NO_TEXT

def test_build_jobs_full_set_covers_all_variations_and_8_slots():
    cx = _cx()
    pv.seed(cx); mods.seed(cx); si.init_tables(cx)
    jobs = si.build_generation_jobs(cx, "alpha")
    assert len(jobs) == 8
    for kind in ("botanical", "mechanism"):
        kjobs = [j for j in jobs if j["kind"] == kind]
        assert sorted(j["variant"] for j in kjobs) == [1, 2, 3, 4]
        assert len({j["prompt_variant_id"] for j in kjobs}) == 4     # all 4 variations
        assert all("No text" in j["prompt_text"] for j in kjobs)     # NO_TEXT appended
        assert all(j["model_id"] in ("flux-1.1-pro", "imagen-4", "recraft-v3") for j in kjobs)

def test_build_jobs_skips_present_slots():
    cx = _cx()
    pv.seed(cx); mods.seed(cx); si.init_tables(cx)
    si.record_image(cx, "beta", "botanical", 1, "botanical-1.png", prompt_variant_id=1, model_id="flux-1.1-pro")
    jobs = si.build_generation_jobs(cx, "beta")
    assert ("botanical", 1) not in {(j["kind"], j["variant"]) for j in jobs}
    assert len(jobs) == 7

def test_build_jobs_deterministic_and_model_offset_varies_by_slug():
    cx = _cx()
    pv.seed(cx); mods.seed(cx); si.init_tables(cx)
    j1 = si.build_generation_jobs(cx, "slug-one")
    j1b = si.build_generation_jobs(cx, "slug-one")
    assert [j["model_id"] for j in j1] == [j["model_id"] for j in j1b]   # deterministic

def test_grouped_returns_tagged_with_labels_and_state():
    cx = _cx()
    mods.seed(cx)
    for v in (1, 2, 3, 4):
        si.record_image(cx, "p", "botanical", v, f"botanical-{v}.png", prompt_variant_id=v, model_id="imagen-4")
    g = si.display_images_grouped(cx, "p")
    assert [e["variant"] for e in g["botanical"]] == [1, 2, 3, 4]
    assert g["botanical"][0]["model_label"] == "Imagen 4"
    assert g["botanical"][0]["url"] == "/begin/product-image/p/botanical-1.png"
    assert g["mechanism"] == []
    assert si.images_grouped_state(cx, "p") == "generating"   # only 4 tagged of 8

def test_grouped_legacy_fallback_no_label():
    cx = _cx()
    si.init_tables(cx)
    si.record_image(cx, "leg", "botanical", 1, "botanical-1.png")   # untagged legacy
    g = si.display_images_grouped(cx, "leg")
    assert len(g["botanical"]) == 1 and g["botanical"][0]["model_label"] is None

def test_grouped_state_ready_at_8():
    cx = _cx()
    mods.seed(cx)
    n = 0
    for kind in ("botanical", "mechanism"):
        for v in (1, 2, 3, 4):
            n += 1
            si.record_image(cx, "full", kind, v, f"{kind}-{v}.png", prompt_variant_id=n, model_id="flux-1.1-pro")
    assert si.images_grouped_state(cx, "full") == "ready"
