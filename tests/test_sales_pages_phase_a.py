import sqlite3
from dashboard import sales_prompt_variations as pv

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
