import sqlite3
from dashboard import sales_prompt_variations as pv

def _cx(): return sqlite3.connect(":memory:")

def test_insert_and_review_variations():
    cx = _cx()
    vid = pv.insert_variation(cx, "botanical", "lbl", "a fresh herb scene")
    assert isinstance(vid, int)
    revs = pv.review_variations(cx, "botanical")
    assert [r["id"] for r in revs] == [vid]
    assert revs[0]["label"] == "lbl" and revs[0]["prompt_template"] == "a fresh herb scene"
    pv.set_state(cx, vid, "candidate")               # set_state from C2
    assert pv.review_variations(cx, "botanical") == []
    assert vid in {v["id"] for v in pv.candidate_variations(cx, "botanical")}
