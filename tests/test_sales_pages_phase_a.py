import sqlite3
from dashboard import sales_prompt_variations as pv
from dashboard import sales_image_models as mods

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
