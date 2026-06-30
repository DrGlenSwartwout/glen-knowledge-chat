import dashboard.practitioner_chat as pc


def _capture(monkeypatch):
    """Monkeypatch _llm_json to capture the system prompt and return a fixed dict."""
    seen = {}
    def fake(system, messages):
        seen["system"] = system
        return {"reply": "ok", "suggested_slugs": []}
    monkeypatch.setattr(pc, "_llm_json", fake)
    return seen


def test_overlay_injected_after_system_before_catalog(monkeypatch):
    seen = _capture(monkeypatch)
    cat = [{"slug": "neuro-magnesium", "name": "Neuro Magnesium", "description": "x"}]
    pc.scoped_reply("hi", [], cat, overlay="OVERLAY-TEXT")
    sys = seen["system"]
    assert "OVERLAY-TEXT" in sys
    # positioned after _SYSTEM and before the catalog line
    assert sys.index(pc._SYSTEM) < sys.index("OVERLAY-TEXT") < sys.index("neuro-magnesium")


def test_no_overlay_is_backward_compatible(monkeypatch):
    seen = _capture(monkeypatch)
    cat = [{"slug": "neuro-magnesium", "name": "Neuro Magnesium", "description": "x"}]
    pc.scoped_reply("hi", [], cat)  # 3-arg legacy call
    sys = seen["system"]
    assert "OVERLAY" not in sys
    # byte-identical to the legacy _SYSTEM + cat_txt assembly
    cat_txt = "\n".join(f"- {c['slug']}: {c.get('name','')} — {c.get('description','')}" for c in cat)
    assert sys == pc._SYSTEM + cat_txt


def test_return_shape_unchanged(monkeypatch):
    _capture(monkeypatch)
    out = pc.scoped_reply("hi", [], [], overlay="X")
    assert set(out.keys()) == {"reply", "suggested_slugs"}
