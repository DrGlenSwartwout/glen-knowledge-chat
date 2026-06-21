import sqlite3, json
from dashboard import sales_prompt_variations as pv
from dashboard import sales_image_prompt_gen as gen

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

def test_generate_inserts_review_candidates_robust_parse_and_dedupe():
    cx = _cx()
    pv.seed(cx)
    existing_tmpl = pv.active_variations(cx, "botanical")[0]["prompt_template"]
    # fake LLM: JSON wrapped in prose + code fences; one item duplicates an existing template
    fake = ('Sure! Here are the prompts:\n```json\n'
            '[{"label":"new-a","prompt_template":"a brand new sunny herb garden scene"},'
            f'{{"label":"dupe","prompt_template":{json.dumps(existing_tmpl)}}}]\n```')
    out = gen.generate_candidates(cx, "botanical", 2, llm=lambda p: fake)
    assert len(out) == 1 and out[0]["label"] == "new-a"          # dupe skipped
    revs = {r["prompt_template"] for r in pv.review_variations(cx, "botanical")}
    assert "a brand new sunny herb garden scene" in revs

def test_generate_malformed_response_returns_empty():
    cx = _cx(); pv.seed(cx)
    assert gen.generate_candidates(cx, "botanical", 2, llm=lambda p: "no json here") == []
    assert pv.review_variations(cx, "botanical") == []

def test_parse_json_array_slices_and_tolerates():
    assert gen._parse_json_array('prefix [ {"a":1} ] suffix') == [{"a": 1}]
    assert gen._parse_json_array("garbage") == []
    assert gen._parse_json_array("") == []
