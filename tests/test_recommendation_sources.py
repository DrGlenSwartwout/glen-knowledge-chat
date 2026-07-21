from dashboard import recommendation_sources as rs


def test_registry_has_the_ten_sources_with_shapes():
    for key in ["biofield", "intake", "scan", "chat", "self",
                "email", "newsletter", "ads", "social", "purchased"]:
        s = rs.get_source(key)
        assert s is not None, key
        assert s["label"] and s["icon"]
        assert s["kind"] in ("clinical", "engagement")


def test_clinical_vs_engagement_kinds():
    # biofield counts client ACTIONS on a reveal (clicked-to-learn / ordered), so it is engagement.
    assert rs.get_source("biofield")["kind"] == "engagement"
    assert rs.get_source("intake")["kind"] == "clinical"
    assert rs.get_source("scan")["kind"] == "engagement"
    assert rs.get_source("purchased")["kind"] == "engagement"


def test_known_source():
    assert rs.known_source("biofield") is True
    assert rs.known_source("nope") is False
