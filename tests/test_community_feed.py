# tests/test_community_feed.py
from dashboard import community_feed as _f


def test_cosine_bounds():
    assert _f.cosine([1, 0], [1, 0]) == 1.0
    assert _f.cosine([1, 0], [0, 1]) == 0.0
    assert _f.cosine([], [1, 0]) == 0.0
    assert _f.cosine([0, 0], [1, 0]) == 0.0


def test_build_interest_text_empty_when_no_signal():
    assert _f.build_interest_text([], [], []) == ""


def test_build_interest_text_concatenates():
    t = _f.build_interest_text(["slept poorly"], ["sleep"], ["asked about melatonin"])
    assert "slept poorly" in t and "sleep" in t and "melatonin" in t


def _cand(id, tags, pub, rc=0):
    return {"id": id, "interest_tags": tags, "published_at": pub, "reaction_count": rc}


def test_rank_filters_blocked_topics():
    cands = [_cand(1, ["sleep"], "2026-01-01"), _cand(2, ["adrenals"], "2026-01-02")]
    vecs = {1: [1, 0], 2: [0, 1]}
    out = _f.rank(cands, [1, 0], vecs, liked_topics=[], blocked_topics=["adrenals"])
    assert [i["id"] for i in out] == [1]  # blocked item removed


def test_rank_liked_boost_changes_order():
    # item 2 is a weaker cosine match but carries a liked topic → boosted above item 1
    cands = [_cand(1, ["x"], "2026-01-01"), _cand(2, ["sleep"], "2026-01-02")]
    vecs = {1: [1, 0.0], 2: [0.9, 0.1]}
    member = [1, 0]
    no_boost = _f.rank(cands, member, vecs, liked_topics=[], blocked_topics=[])
    assert no_boost[0]["id"] == 1
    boosted = _f.rank(cands, member, vecs, liked_topics=["sleep"], blocked_topics=[])
    assert boosted[0]["id"] == 2
    assert "sleep" in boosted[0]["reason"]


def test_rank_cold_start_newest_then_reactions():
    cands = [_cand(1, [], "2026-01-01", rc=5), _cand(2, [], "2026-02-01", rc=0),
             _cand(3, [], "2026-01-01", rc=9)]
    out = _f.rank(cands, member_vec=[], content_vecs={}, liked_topics=[], blocked_topics=[])
    assert [i["id"] for i in out] == [2, 3, 1]  # newest first, then by reactions
    assert out[0]["reason"]  # non-empty cold-start reason


def test_reason_for_branches():
    item = {"interest_tags": ["sleep"]}
    assert "sleep" in _f.reason_for(item, ["sleep"], has_vec=True, cold_start=False)
    assert _f.reason_for({"interest_tags": ["x"]}, [], has_vec=True, cold_start=False) \
        == "Related to your recent reflections"
    assert _f.reason_for({"interest_tags": []}, [], has_vec=False, cold_start=True) \
        == "New in the community"
