"""Proposal engine: aggregates similar codes' formulations into ranked {name, score}."""
from dashboard.formulation_map_propose import propose_for_code

# item matches: metadata carries a comma-joined `formulations` list per similar code
_MATCHES = [
    {"metadata": {"code": "ED5", "formulations": "Blood Cleanse, Chelation"}, "score": 0.99},
    {"metadata": {"code": "ED6", "formulations": "Heart Health, Rhythm Restore"}, "score": 0.92},
    {"metadata": {"code": "ED11", "formulations": "Chelation, Liver Support"}, "score": 0.80},
]


def _deps(matches):
    return {
        "fetch": lambda ids: {"vectors": {ids[0]: {"values": [0.1, 0.2]}}},
        "query": lambda vec, top_k: {"matches": matches},
    }


def test_aggregates_and_ranks_by_best_score():
    out = propose_for_code("ED5", top_k=6, deps=_deps(_MATCHES))
    names = [o["name"] for o in out]
    # Chelation appears at 0.99 and 0.80 -> keeps the best (0.99), ranks near top
    assert "Blood Cleanse" in names and "Heart Health" in names and "Chelation" in names
    assert dict((o["name"], o["score"]) for o in out)["Chelation"] == 0.99
    assert out[0]["score"] >= out[-1]["score"]              # sorted desc


def test_exclude_drops_already_mapped():
    out = propose_for_code("ED5", exclude=["Blood Cleanse", "chelation"], deps=_deps(_MATCHES))
    names = {o["name"].lower() for o in out}
    assert "blood cleanse" not in names and "chelation" not in names
    assert "heart health" in names


def test_missing_vector_and_errors_return_empty():
    assert propose_for_code("ZZ9", deps={"fetch": lambda ids: {"vectors": {}}, "query": lambda v, k: {}}) == []

    def boom(ids):
        raise RuntimeError("pinecone down")
    assert propose_for_code("ED5", deps={"fetch": boom, "query": lambda v, k: {}}) == []
