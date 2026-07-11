from dashboard import related_products as rp

_PRODUCTS = {
    "iop-syntropy": {"name": "IOP Syntropy"},
    "immune-modulation": {"name": "Immune Modulation"},
    "wholomega": {"name": "WholOmega"},
    "neuroprotect": {"name": "Neuroprotect"},
    "book-healing-glaucoma": {"name": "Healing Glaucoma (book)"},
    "denas-scenar": {"name": "DENAS PCM Pro"},
    "old-thing": {"name": "Old", "inactive": True},
    "water-ionizer-9plate": {"name": "9-Plate Water Ionizer (Living Water)"},
}

def test_manual_first_then_one_auto_in_featured():
    out = rp.resolve_related(
        "iop-syntropy",
        manual=["immune-modulation"],
        harvested=["wholomega", "neuroprotect"],
        semantic=["book-healing-glaucoma"],
        products=_PRODUCTS)
    assert out["featured"] == ["immune-modulation", "wholomega"]
    assert out["more"] == ["neuroprotect", "book-healing-glaucoma"]

def test_auto_drops_self_inactive_and_do_not_recommend():
    out = rp.resolve_related(
        "iop-syntropy",
        manual=[],
        harvested=["iop-syntropy", "old-thing", "water-ionizer-9plate", "wholomega"],
        semantic=[],
        products=_PRODUCTS)
    assert out["featured"] == ["wholomega"]
    assert out["more"] == []

def test_manual_bypasses_guardrail_but_dedups_from_auto():
    out = rp.resolve_related(
        "iop-syntropy",
        manual=["water-ionizer-9plate", "wholomega"],
        harvested=["wholomega", "neuroprotect"],
        semantic=[],
        products=_PRODUCTS)
    # manual keeps the do-not-recommend pick; wholomega not repeated in auto
    assert out["featured"] == ["water-ionizer-9plate", "wholomega", "neuroprotect"]
    assert out["more"] == []

def test_auto_capped(monkeypatch):
    prods = {f"p{i}": {"name": str(i)} for i in range(20)}
    prods["base"] = {"name": "base"}
    out = rp.resolve_related("base", manual=[], harvested=[f"p{i}" for i in range(20)],
                             semantic=[], products=prods, cap=12)
    assert len(out["featured"]) + len(out["more"]) == 12

def test_empty_when_nothing_related():
    out = rp.resolve_related("iop-syntropy", manual=[], harvested=[], semantic=[], products=_PRODUCTS)
    assert out["featured"] == [] and out["more"] == []

_SLUGS = {"immune-modulation", "book-healing-glaucoma", "denas-scenar", "water-ionizer-9plate"}
_ALIASES = {
    "healing-glaucoma-book": "book-healing-glaucoma",
    "denas-microcurrent-system-for-eye-healing": "denas-scenar",
    "living-water-ionizer-9-plate": "water-ionizer-9plate",
}

def test_map_exact_remedies_url():
    assert rp.map_storefront_slug(
        "https://remedymatch.com/remedies/syntropy/56-immune-modulation",
        _SLUGS, _ALIASES) == "immune-modulation"

def test_map_resources_via_alias():
    assert rp.map_storefront_slug(
        "https://remedymatch.com/resources/50-healing-glaucoma-book",
        _SLUGS, _ALIASES) == "book-healing-glaucoma"

def test_map_unknown_returns_none():
    assert rp.map_storefront_slug(
        "https://remedymatch.com/resources/999-mystery-widget",
        _SLUGS, _ALIASES) is None

def test_map_rejects_nonremedymatch_domain():
    assert rp.map_storefront_slug(
        "https://evil.example.com/resources/50-healing-glaucoma-book",
        _SLUGS, _ALIASES) is None

def test_map_tolerates_query_string():
    assert rp.map_storefront_slug(
        "https://remedymatch.com/remedies/syntropy/56-immune-modulation?ref=email",
        _SLUGS, _ALIASES) == "immune-modulation"
