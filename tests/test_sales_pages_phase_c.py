import sqlite3
from dashboard import sales_image_exposures as ex
from dashboard import sales_image_leaderboard as lb
from dashboard import sales_images as si
from dashboard import sales_votes as sv

def _cx(): return sqlite3.connect(":memory:")

def test_record_dedups_per_session():
    cx = _cx()
    ex.record(cx, "a", "s1")
    ex.record(cx, "a", "s1")     # same session -> no new row
    ex.record(cx, "a", "s2")     # different session
    ex.record(cx, "b", "s1")     # different product
    assert ex.per_product_counts(cx) == {"a": 2, "b": 1}

def test_record_ignores_empty_session():
    cx = _cx()
    ex.record(cx, "a", "")
    ex.record(cx, "a", None)
    assert ex.per_product_counts(cx) == {}

def test_wilson_lower_rewards_volume_at_equal_rate():
    assert lb.wilson_lower(0, 0) == 0.0
    assert lb.wilson_lower(80, 100) > lb.wilson_lower(8, 10)   # same 0.8 rate, more data -> higher lower bound
    assert 0.0 < lb.wilson_lower(8, 10) < 0.8

def test_leaderboard_model_impressions_use_containing_products():
    cx = _cx()
    # product a contains models flux + imagen; product b contains flux + recraft
    si.record_image(cx, "a", "botanical", 1, "a-b1.png", prompt_variant_id=1, model_id="flux-1.1-pro")
    si.record_image(cx, "a", "mechanism", 1, "a-m1.png", prompt_variant_id=5, model_id="imagen-4")
    si.record_image(cx, "b", "botanical", 1, "b-b1.png", prompt_variant_id=1, model_id="flux-1.1-pro")
    si.record_image(cx, "b", "mechanism", 1, "b-m1.png", prompt_variant_id=5, model_id="recraft-v3")
    for i in range(10): ex.record(cx, "a", f"a{i}")   # a: 10 sessions
    for i in range(5):  ex.record(cx, "b", f"b{i}")   # b: 5 sessions
    # votes (tagged with model_id, Phase-B style)
    for i in range(6): sv.record_pick(cx, "a", "botanical", 1, f"va{i}", model_id="flux-1.1-pro", prompt_variant_id=1)
    for i in range(2): sv.record_pick(cx, "a", "mechanism", 1, f"vm{i}", model_id="imagen-4", prompt_variant_id=5)
    sv.record_pick(cx, "b", "mechanism", 1, "vb0", model_id="recraft-v3", prompt_variant_id=5)
    data = lb.leaderboard(cx, min_volume=8)
    models = {r["key"]: r for r in data["models"]}
    assert models["flux-1.1-pro"]["impressions"] == 15   # a(10) + b(5)
    assert models["imagen-4"]["impressions"] == 10        # a only
    assert models["recraft-v3"]["impressions"] == 5       # b only
    assert models["flux-1.1-pro"]["votes"] == 6
    assert abs(models["flux-1.1-pro"]["rate"] - 6/15) < 1e-9
    assert models["recraft-v3"]["low_volume"] is True     # 5 < 8
    assert data["models"][0]["rank"] == 1                 # ranked, wilson desc
    # variations present too
    keys = {r["key"] for r in data["variations"]}
    assert 1 in keys and 5 in keys
