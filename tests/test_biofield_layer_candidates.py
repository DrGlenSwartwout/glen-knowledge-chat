"""layer_candidates: per-layer ranked remedy pick-list (augments the set-cover)."""
import sqlite3

from dashboard.biofield_stress import (
    init_stress_tables, seed_from_scan, save_remedy_set, layer_candidates)

# ED1 covered by two heart remedies (real alternatives); ES3 by lymph flow;
# MB5 covered by nothing -> its layer is "blank" and must fall back to functional.
_FIND = [{"code": "ED1", "name": "Membrane"},
         {"code": "ES3", "name": "Lymph"},
         {"code": "MB5", "name": "Calm"}]
_COV = {"Heart Health": {"ED1"}, "Cardio Plus": {"ED1"}, "Lymph Flow": {"ES3"}}
# One chain row per layer. Layer 3 has no remedy AND its head matches MB5's label,
# so MB5 assigns to it by head -> a real blank layer (has a code, no coverer).
_CHAIN = [{"layer": 1, "head": "Membrane", "remedy": "Heart Health"},
          {"layer": 2, "head": "Lymph", "remedy": "Lymph Flow"},
          {"layer": 3, "head": "Calm", "remedy": ""}]


def _seed(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    seed_from_scan(cx, "a5", _FIND, _COV)
    return cx


def _layer(lc, n):
    return next(L for L in lc if L["n"] == n)


def test_covering_alternatives_ranked_with_default_marked(tmp_path):
    cx = _seed(tmp_path)
    L1 = _layer(layer_candidates(cx, "a5", _CHAIN), 1)
    names = {c["remedy"].lower() for c in L1["candidates"]}
    assert {"heart health", "cardio plus"} <= names          # both cover ED1
    assert any(c.get("is_default") and c["remedy"].lower() == "heart health"
               for c in L1["candidates"])                    # current pick flagged
    assert all(c["source"] == "coverage" for c in L1["candidates"])


def test_learned_boost_lifts_a_prior_pick(tmp_path):
    cx = _seed(tmp_path)
    save_remedy_set(cx, "a5", ["Cardio Plus"])               # Glen's prior choice
    L1 = _layer(layer_candidates(cx, "a5", _CHAIN), 1)
    top = L1["candidates"][0]
    assert top["remedy"].lower() == "cardio plus" and top["used_before"] is True


def test_blank_layer_falls_back_to_functional(tmp_path):
    cx = _seed(tmp_path)
    lc = layer_candidates(cx, "a5", _CHAIN, fallback_by_code={"MB5": ["Emotional Stress Release"]})
    L3 = _layer(lc, 3)
    assert L3["codes"] == ["MB5"]
    assert L3["candidates"], "blank layer must still offer candidates"
    assert L3["candidates"][0]["source"] == "functional"
    assert L3["candidates"][0]["remedy"] == "Emotional Stress Release"


def test_candidates_capped_at_n(tmp_path):
    cx = _seed(tmp_path)
    for i in range(8):
        cx.execute("INSERT INTO biofield_auth_remedy_coverage(test_id,remedy,code) VALUES(5,?,?)",
                   (f"Opt {i}", "ED1"))
    cx.commit()
    L1 = _layer(layer_candidates(cx, "a5", _CHAIN, n=5), 1)
    assert len(L1["candidates"]) <= 5
    assert any(c.get("is_default") for c in L1["candidates"])   # default survives the cap
