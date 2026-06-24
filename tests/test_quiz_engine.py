import quiz_engine


def test_config_loads_and_quiz_present():
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    assert q is not None
    assert q["product_slug"] == "neuro-magnesium"
    assert len(q["questions"]) == 9
    # disclaimer present and DSHEA-correct
    assert "not been evaluated by the Food and Drug Administration" in q["disclaimer"]


def test_segment_of_reads_q1():
    assert quiz_engine.segment_of({"q1": "watch_wait", "q2": "restful"}) == "watch_wait"
    assert quiz_engine.segment_of({}) == "general"


def test_depletion_score_counts_high_signals():
    high = {"q2": "frequent", "q3": "often", "q4": "frequent_fog",
            "q5": "6plus", "q6": "avoid", "q7": "rarely", "q8": "none"}
    assert quiz_engine.depletion_score(high) == 7
    low = {"q2": "restful", "q3": "rarely", "q4": "sharp",
           "q5": "under2", "q6": "comfortable", "q7": "yes", "q8": "both"}
    assert quiz_engine.depletion_score(low) == 0


def test_result_barrier_band_for_watch_wait():
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    r = quiz_engine.result_for(q, {"q1": "watch_wait", "q8": "eye_formula"})
    assert r["band"] == "barrier"
    assert "barrier" in r["reasoning"].lower()
    assert r["segment"] == "watch_wait"
    assert isinstance(r["bullets"], list) and r["bullets"]


def test_result_calm_band_for_stress():
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    r = quiz_engine.result_for(q, {"q1": "general", "q2": "frequent", "q3": "often"})
    assert r["band"] == "calm"


def test_result_always_has_no_disease_nouns_or_emdash():
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    banned = ["macular", "amd", "glaucoma", "cataract", "alzheimer", "dementia", "—"]
    for answers in ({"q1": "watch_wait"}, {"q1": "general", "q4": "frequent_fog"},
                    {"q1": "family", "q8": "both"}, {"q1": "supplement_gap", "q5": "6plus"}):
        r = quiz_engine.result_for(q, answers)
        blob = (r["headline"] + " " + r["reasoning"] + " " + " ".join(r["bullets"])).lower()
        for b in banned:
            assert b not in blob, f"banned token {b!r} in result for {answers}"
