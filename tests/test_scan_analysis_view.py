from dashboard import scan_analysis_view as v

ART = {
    "scan_count": 12, "date_range": ["2024-01-01", "2026-06-01"],
    "generated_at": "2026-06-22T00:00:00Z", "narrative": "Over time you...",
    "top_patterns": [{"code": "MR2"}, {"code": "ED11"}, {"code": "EI1"}, {"code": "ES3"}],
    "clusters": [{"structure": "Nervous System", "codes": ["MR2", "ED4"]}],
    "functional_relation": [{"structure": "Detoxification", "weight": 9, "is_functional": True}],
}


def test_paid_sees_full_even_when_flag_off():
    p = v.gated_payload(ART, tier=v.PAID, free_enabled=False)
    assert p["access"] == "full" and p["upsell"] is False
    assert p["narrative"] == "Over time you..."
    assert len(p["top_patterns"]) == 4
    assert p["clusters"] and p["functional_relation"]


def test_free_gets_teaser_when_flag_off():
    p = v.gated_payload(ART, tier=v.FREE, free_enabled=False)
    assert p["access"] == "teaser" and p["upsell"] is True
    assert len(p["top_patterns"]) == 3          # top 3 only
    assert p["narrative"] == ""                  # depth withheld
    assert p["clusters"] == [] and p["functional_relation"] == []
    assert p["scan_count"] == 12                 # non-sensitive context still shown


def test_free_sees_full_when_flag_on():
    p = v.gated_payload(ART, tier=v.FREE, free_enabled=True)
    assert p["access"] == "full" and p["upsell"] is False
    assert len(p["top_patterns"]) == 4 and p["clusters"]


def test_none_is_locked_regardless_of_flag():
    for flag in (True, False):
        p = v.gated_payload(ART, tier=v.NONE, free_enabled=flag)
        assert p["access"] == "locked" and p["upsell"] is True
        assert p["top_patterns"] == [] and p["narrative"] == ""


def test_empty_artifact_is_safe():
    p = v.gated_payload(None, tier=v.PAID, free_enabled=False)
    assert p["access"] == "full" and p["scan_count"] == 0
    assert p["top_patterns"] == [] and p["date_range"] == [None, None]


def test_resolve_tier():
    assert v.resolve_tier(is_paid=True, has_tos=True) == v.PAID
    assert v.resolve_tier(is_paid=False, has_tos=True) == v.FREE
    assert v.resolve_tier(is_paid=False, has_tos=False) == v.NONE


def test_format_facts_includes_patterns_clusters_functional():
    f = v.format_facts(ART)
    assert "MR2" in f and "Calm Mind" not in f  # ART top_patterns have no name; code present
    assert "Nervous System" in f                 # cluster structure + codes
    assert "ED4" in f
    assert "Detoxification" in f                  # functional pattern
    assert "Over time you..." in f               # narrative folded in


def test_chat_context_full_is_grounded():
    c = v.chat_context(ART, access="full")
    assert c["grounded"] is True and c["upsell"] is False
    assert "Nervous System" in c["facts"]


def test_chat_context_teaser_and_locked_are_not_grounded():
    for acc in ("teaser", "locked"):
        c = v.chat_context(ART, access=acc)
        assert c["grounded"] is False and c["upsell"] is True
        assert c["facts"] == ""
