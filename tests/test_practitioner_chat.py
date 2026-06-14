# tests/test_practitioner_chat.py
from dashboard import practitioner_chat as pc

CATALOG = [
    {"slug": "brain-boost", "name": "Brain Boost", "description": "nootropic support"},
    {"slug": "bone-builder", "name": "Bone Builder", "description": "bone density"},
]

def test_scoped_reply_returns_reply_and_validated_slugs(monkeypatch):
    # stub the LLM to return a reply + a suggested slug that IS in the catalog, plus a
    # hallucinated slug that is NOT — only the valid one survives.
    monkeypatch.setattr(pc, "_llm_json", lambda system, messages: {
        "reply": "For focus, Brain Boost is a good fit.",
        "suggested_slugs": ["brain-boost", "not-a-real-slug"]})
    out = pc.scoped_reply("something for focus", [], CATALOG)
    assert "Brain Boost" in out["reply"]
    assert out["suggested_slugs"] == ["brain-boost"]      # hallucination dropped

def test_scoped_reply_strips_external_links(monkeypatch):
    # even if the model leaks a URL, the reply is scrubbed of links/store mentions.
    monkeypatch.setattr(pc, "_llm_json", lambda system, messages: {
        "reply": "Try it here: https://remedymatch.com/x or truly.vip/y", "suggested_slugs": []})
    out = pc.scoped_reply("hi", [], CATALOG)
    assert "http" not in out["reply"] and "truly.vip" not in out["reply"] and "remedymatch" not in out["reply"].lower()

def test_empty_catalog_safe(monkeypatch):
    monkeypatch.setattr(pc, "_llm_json", lambda system, messages: {"reply": "ok", "suggested_slugs": []})
    out = pc.scoped_reply("hi", [], [])
    assert out["suggested_slugs"] == []

def test_all_four_banned_brands_scrubbed(monkeypatch):
    """illtowell.com and truly.so are also scrubbed; valid slug survives."""
    monkeypatch.setattr(pc, "_llm_json", lambda system, messages: {
        "reply": (
            "Check illtowell.com for details and also truly.so/shop "
            "or remedymatch.com or truly.vip/promo"
        ),
        "suggested_slugs": ["bone-builder", "  brain-boost  "]   # whitespace slug won't match
    })
    out = pc.scoped_reply("what do you recommend?", [], CATALOG)
    reply = out["reply"].lower()
    assert "illtowell" not in reply
    assert "truly.so" not in reply
    assert "remedymatch" not in reply
    assert "truly.vip" not in reply
    assert "http" not in reply
    # exact-match slug with whitespace is NOT in catalog → dropped
    assert out["suggested_slugs"] == ["bone-builder"]
