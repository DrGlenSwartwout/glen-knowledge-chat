import sqlite3
from dashboard import sales_image_models as mods
from dashboard import sales_prompt_variations as pv
from dashboard import sales_image_leaderboard as lb
from dashboard import sales_image_evolution as ev
from dashboard import sales_images as si
from dashboard import sales_votes as sv
from dashboard import sales_image_exposures as ex

def _cx(): return sqlite3.connect(":memory:")

def _seed_model_field(cx, *, loser_votes, winner_votes, impressions_each):
    # two products so flux & recraft each appear; give exposures + lopsided votes
    si.record_image(cx, "p1", "botanical", 1, "p1b.png", prompt_variant_id=1, model_id="flux-1.1-pro")
    si.record_image(cx, "p1", "mechanism", 1, "p1m.png", prompt_variant_id=5, model_id="recraft-v3")
    for i in range(impressions_each): ex.record(cx, "p1", f"s{i}")
    for i in range(winner_votes): sv.record_pick(cx, "p1", "botanical", 1, f"w{i}", model_id="flux-1.1-pro", prompt_variant_id=1)
    for i in range(loser_votes):  sv.record_pick(cx, "p1", "mechanism", 1, f"l{i}", model_id="recraft-v3", prompt_variant_id=5)

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

def test_wilson_upper_brackets_rate():
    assert lb.wilson_upper(0, 0) == 0.0
    lo, hi = lb.wilson_lower(5, 10), lb.wilson_upper(5, 10)
    assert lo < 0.5 < hi                       # interval brackets the 0.5 rate
    assert lb.wilson_upper(5, 10) > lb.wilson_upper(50, 100)   # less data -> wider/higher upper

def test_propose_fires_on_confident_loser_with_candidate():
    cx = _cx()
    mods.seed(cx); mods.seed_candidates(cx)
    # recraft is the confident loser (0/60), flux the winner (55/60); imagen has no data
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    props = ev.propose(cx, min_impressions=20)
    model_props = [p for p in props if p["axis"] == "model"]
    assert any(p["retire_key"] == "recraft-v3" for p in model_props)
    p = next(p for p in model_props if p["retire_key"] == "recraft-v3")
    assert p["promote_key"] in {"ideogram-v3", "flux-ultra", "sd-3.5-large"}
    # persisted as pending, and idempotent (no duplicate pending)
    n1 = len(ev.pending_proposals(cx)); ev.propose(cx, min_impressions=20)
    assert len(ev.pending_proposals(cx)) == n1

def test_propose_no_fire_when_intervals_overlap():
    cx = _cx()
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=28, winner_votes=32, impressions_each=60)  # 28/60 vs 32/60 -> overlap
    props = [p for p in ev.propose(cx, min_impressions=20) if p["axis"] == "model"]
    assert props == []

def test_decide_approve_swaps_and_keeps_count():
    cx = _cx()
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    ev.propose(cx, min_impressions=20)
    pid = next(p["id"] for p in ev.pending_proposals(cx) if p["axis"] == "model")
    before = len(mods.active_models(cx))
    res = ev.decide(cx, pid, "approve", actor="t")
    assert res["ok"] and res["applied"]
    active = {m["id"] for m in mods.active_models(cx)}
    assert "recraft-v3" not in active                  # retired
    assert len(active) == before                       # set-size preserved
    assert not ev.pending_proposals(cx)                # proposal consumed

def test_decide_reject_no_state_change():
    cx = _cx()
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    ev.propose(cx, min_impressions=20)
    pid = next(p["id"] for p in ev.pending_proposals(cx) if p["axis"] == "model")
    ev.decide(cx, pid, "reject", actor="t")
    assert "recraft-v3" in {m["id"] for m in mods.active_models(cx)}
    assert not ev.pending_proposals(cx)

def test_trial_and_undo():
    cx = _cx()
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    res = ev.trial(cx, "model", "", "ideogram-v3", actor="t")
    assert res["ok"]
    assert "ideogram-v3" in {m["id"] for m in mods.active_models(cx)}
    log_id = res["log_id"]
    ev.undo(cx, log_id, actor="t")
    assert "ideogram-v3" not in {m["id"] for m in mods.active_models(cx)}   # back to candidate

def test_console_section_html_lists_proposals_and_candidates():
    cx = _cx()
    mods.seed(cx); mods.seed_candidates(cx)
    _seed_model_field(cx, loser_votes=0, winner_votes=55, impressions_each=60)
    ev.propose(cx, min_impressions=20)
    html = ev.console_section_html(cx)
    assert "recraft-v3" in html              # the proposed retire
    assert "Approve" in html and "Reject" in html
    assert "ideogram-v3" in html             # a benched candidate
    assert "Trial" in html
