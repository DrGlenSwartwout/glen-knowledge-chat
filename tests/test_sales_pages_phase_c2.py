import sqlite3
from dashboard import sales_image_models as mods
from dashboard import sales_prompt_variations as pv

def _cx(): return sqlite3.connect(":memory:")

def test_model_candidates_seed_and_setstate():
    cx = _cx(); mods.seed(cx)                 # 3 active
    mods.seed_candidates(cx)                  # + 3 candidate
    cands = {m["id"] for m in mods.candidate_models(cx)}
    assert cands == {"ideogram-v3", "flux-ultra", "sd-3.5-large"}
    assert {m["id"] for m in mods.active_models(cx)} == {"flux-1.1-pro", "imagen-4", "recraft-v3"}
    mods.seed_candidates(cx)                  # idempotent
    assert len(mods.candidate_models(cx)) == 3
    mods.set_state(cx, "ideogram-v3", "active")
    assert "ideogram-v3" in {m["id"] for m in mods.active_models(cx)}

def test_variation_setstate_and_candidates():
    cx = _cx(); pv.seed(cx)
    first = pv.active_variations(cx, "botanical")[0]["id"]
    pv.set_state(cx, first, "candidate")
    assert first in {v["id"] for v in pv.candidate_variations(cx, "botanical")}
    assert first not in {v["id"] for v in pv.active_variations(cx, "botanical")}
