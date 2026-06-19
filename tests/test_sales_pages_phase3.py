import sqlite3
from dashboard import sales_images as si
from dashboard import sales_image_prompts as sip

def _cx(): return sqlite3.connect(":memory:")

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
