from dashboard import portal_recommendations as pr


def _resolve(slug):
    return {"name": {"neuro-magnesium": "Neuro Magnesium"}.get(slug, slug), "url": "/begin/buy/" + slug}


def test_product_appears_in_each_source_section_sorted():
    product_sources = [
        {"product_key": "neuro-magnesium", "hidden": False, "sources": [
            {"source": "self", "count": 1, "first_touch": "2026-06-01", "last_touch": "2026-06-01"},
            {"source": "biofield", "count": 3, "first_touch": "2026-07-01", "last_touch": "2026-07-20"}]},
        {"product_key": "immune-modulation", "hidden": False, "sources": [
            {"source": "biofield", "count": 1, "first_touch": "2026-07-05", "last_touch": "2026-07-05"}]},
        {"product_key": "hidden-one", "hidden": True, "sources": [
            {"source": "biofield", "count": 9, "first_touch": "2026-07-01", "last_touch": "2026-07-01"}]},
    ]
    secs = pr.build_sections(product_sources, {}, {}, _resolve)
    by = {s["source"]: s for s in secs}
    # biofield section: neuro-magnesium (count 3) before immune-modulation (count 1); hidden excluded
    assert [p["product_key"] for p in by["biofield"]["products"]] == ["neuro-magnesium", "immune-modulation"]
    # self section present with neuro-magnesium
    assert [p["product_key"] for p in by["self"]["products"]] == ["neuro-magnesium"]
    # neuro-magnesium's icon row carries BOTH its sources with counts
    icons = {i["source"]: i["count"] for i in by["biofield"]["products"][0]["icons"]}
    assert icons == {"self": 1, "biofield": 3}
    # section order follows the registry (biofield before self)
    assert [s["source"] for s in secs].index("biofield") < [s["source"] for s in secs].index("self")


def test_notes_and_collapse_attached_top_n():
    product_sources = [{"product_key": f"p{i}", "hidden": False,
                        "sources": [{"source": "purchased", "count": 10 - i,
                                     "first_touch": "d", "last_touch": "d"}]} for i in range(7)]
    notes = {"p0": {"operator_note": "take daily", "client_note": "works"}}
    secs = pr.build_sections(product_sources, notes, {"purchased": True}, lambda s: {"name": s, "url": ""}, top_n=5)
    sec = next(s for s in secs if s["source"] == "purchased")
    assert sec["collapsed"] is True
    assert sec["total"] == 7 and sec["shown"] == 5 and len(sec["products"]) == sec["total"]
    assert sec["products"][0]["product_key"] == "p0"
    assert sec["products"][0]["operator_note"] == "take daily" and sec["products"][0]["client_note"] == "works"
