from dashboard.ff_matcher import generate_ff_matches

SCAN = [
    {"label": "Adrenal / stress axis", "category": "ES"},
    {"label": "Liver detox pathway", "category": "ED"},
]

def _query(text, top_k):
    # verifies the generator built a query from the scan labels
    assert "Adrenal" in text and "Liver" in text
    return [
        {"id": "a", "score": 0.91, "metadata": {"name": "Adrenal Restore"}},
        {"id": "b", "score": 0.88, "metadata": {"name": "Liver Support"}},
        {"id": "c", "score": 0.80, "metadata": {"name": "Adrenal Restore"}},  # dup name
        {"id": "d", "score": 0.70, "metadata": {"name": "Unresolvable Thing"}},
    ][:top_k]

def _resolve(name):
    return {"Adrenal Restore": "adrenal-restore", "Liver Support": "liver-support"}.get(name)

def _dest(slug):
    return f"/begin/product/{slug}"

def test_generate_ranks_dedupes_resolves_and_carries_no_dosing():
    out = generate_ff_matches(SCAN, query_matches=_query, resolve_slug=_resolve,
                              destination=_dest, top_k=4)
    # unresolvable dropped, dup slug collapsed, ordered by score
    assert [m["slug"] for m in out] == ["adrenal-restore", "liver-support"]
    assert out[0]["url"] == "/begin/product/adrenal-restore"
    assert all("dosing" not in m for m in out)
    assert out[0]["score"] >= out[1]["score"]

def test_empty_candidates_returns_empty_not_error():
    assert generate_ff_matches(SCAN, query_matches=lambda t, k: [],
                               resolve_slug=_resolve, destination=_dest) == []
