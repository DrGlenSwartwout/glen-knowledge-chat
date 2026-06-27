from dashboard.portal_concierge import build_context, system_prompt

SAMPLE_CONTENT = {
    "layers": [{"n": 3, "title": "Liver terrain", "meaning": "detox load", "remedy": "Terrain Restore"}],
    "findings": [{"code": "EI8", "name": "stress", "rank": 1}],
    "reorder_items": [{"slug": "terrain-restore", "qty": 1}],
}
SAMPLE_ORDERS = [{"items": [{"name": "Neuro-Magnesium", "qty": 1}], "status": "shipped"}]

def test_build_context_collects_grounding_facts():
    ctx = build_context(SAMPLE_CONTENT, SAMPLE_ORDERS)
    assert ctx["has_data"] is True
    assert any("stress" in f.get("name", "") for f in ctx["findings"])
    assert any("Terrain Restore" in (l.get("remedy") or "") for l in ctx["layers"])
    assert "Neuro-Magnesium" in ctx["owned"]

def test_build_context_no_data_degrades():
    ctx = build_context({}, [])
    assert ctx["has_data"] is False
    assert ctx["owned"] == [] and ctx["findings"] == [] and ctx["layers"] == []

def test_system_prompt_embeds_facts_and_is_ongoing():
    p = system_prompt(build_context(SAMPLE_CONTENT, SAMPLE_ORDERS))
    assert "Terrain Restore" in p and "Neuro-Magnesium" in p
    assert "post-purchase" not in p.lower()      # widened away from the one-purchase framing
    assert "ongoing" in p.lower() or "your scan" in p.lower()

def test_system_prompt_no_data_still_valid():
    p = system_prompt(build_context({}, []))
    assert isinstance(p, str) and len(p) > 100   # generic-but-valid
