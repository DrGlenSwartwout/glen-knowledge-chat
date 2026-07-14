from dashboard import entity_refs as er


def test_clip_keeps_n_sentences_and_caps():
    assert er.clip("One. Two. Three.", sentences=2) == "One. Two."
    long = "x" * 400
    out = er.clip(long, sentences=1, cap=280)
    assert len(out) == 281 and out.endswith("…")


def test_pattern_ref_is_popup_only():
    r = er.pattern_ref("Heavy Metals", "Accumulated metals stress detoxification. More detail. And more.")
    assert r["name"] == "Heavy Metals"
    assert r["info"].startswith("Accumulated metals stress")
    assert r["href"] is None


def test_pattern_ref_blank_description_gives_empty_info():
    assert er.pattern_ref("ER Stress", "")["info"] == ""


class _FakeCx:
    """Minimal stand-in; entity_refs only calls the wrapped module functions,
    which we monkeypatch, so this object is never queried directly."""


def test_remedy_ref_href_only_when_product_exists(monkeypatch):
    monkeypatch.setattr(er._ba, "resolve_remedy_name", lambda cx, s: "Terrain Restore")
    monkeypatch.setattr(er._bm, "get_map", lambda cx: {"terrain-restore": "Rebuilds terrain. Second sentence. Third."})
    r = er.remedy_ref(_FakeCx(), "terain restore", product_exists=lambda slug: slug == "terrain-restore")
    assert r["href"] == "/begin/product/terrain-restore"
    assert r["info"] == "Rebuilds terrain. Second sentence."


def test_remedy_ref_no_product_exists_is_popup_only(monkeypatch):
    monkeypatch.setattr(er._ba, "resolve_remedy_name", lambda cx, s: "Mystery Remedy")
    monkeypatch.setattr(er._bm, "get_map", lambda cx: {"mystery-remedy": "A meaning."})
    r = er.remedy_ref(_FakeCx(), "mystery remedy")  # product_exists=None
    assert r["href"] is None
    assert r["info"] == "A meaning."


def test_remedy_ref_blank_name():
    assert er.remedy_ref(_FakeCx(), "")["href"] is None


def test_function_ref_uses_topic_page_and_links(monkeypatch):
    page = {"slug": "detoxification", "kind": "function", "state": "approved",
            "content_json": {"summary": "How the body clears toxins. Extra sentence here."}}
    monkeypatch.setattr(er._tp, "get_page", lambda cx, slug: page if slug == "detoxification" else None)
    r = er.function_ref(_FakeCx(), "Detoxification")
    assert r["href"] == "/learn/detoxification"
    assert r["info"].startswith("How the body clears toxins")


def test_function_ref_unapproved_or_missing_is_plain(monkeypatch):
    monkeypatch.setattr(er._tp, "get_page", lambda cx, slug: None)
    r = er.function_ref(_FakeCx(), "Nonexistent")
    assert r["href"] is None and r["info"] == ""


def test_ingredient_ref_pulls_info_and_links(monkeypatch):
    getter = lambda slug: {"content_json": {"what": "A magnesium form that crosses into the brain."}}
    r = er.ingredient_ref(_FakeCx(), "Magnesium L-Threonate", "magnesium-l-threonate", page_getter=getter)
    assert r["href"] == "/begin/ingredient/magnesium-l-threonate"
    assert r["info"].startswith("A magnesium form")


def test_ingredient_ref_no_page_is_plain():
    r = er.ingredient_ref(_FakeCx(), "Obscure Herb", "obscure-herb", page_getter=lambda slug: None)
    assert r["href"] is None and r["info"] == ""
