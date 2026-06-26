"""Regression: the voice-journal analysis intermittently returned no analysis
because Haiku emitted JSON with UNESCAPED double-quotes inside free-text fields
(e.g. self_contradictions: ["X" vs. "Y"]) -> strict json.loads rejected it ->
_haiku_analyze raised "non-JSON" -> analyze() degraded to empty analysis.

Fix: force structured output via tool-use, so the API returns already-valid
structured data (a dict) instead of model-emitted text JSON that can be
mis-escaped. These tests pin that contract."""
import json
import types
import journal_blueprint as jb


class _Resp:
    def __init__(self, body):
        self._body = body
        self.ok = True
        self.status_code = 200
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _tool_use_body(analysis):
    return {"content": [{"type": "tool_use", "name": "emit_analysis", "input": analysis}],
            "stop_reason": "tool_use"}


def test_haiku_analyze_returns_tool_use_input(monkeypatch):
    # The model's analysis contains a contradiction phrased with literal quotes —
    # exactly what broke text-JSON parsing. Via tool-use it arrives as a dict.
    analysis = {
        "emotions": {"Calmness": 0.7, "Anxiety": 0.2},
        "elements": {"Wood": 10, "Fire": 60, "Earth": 20, "Metal": 5, "Water": 5},
        "treasures": {"Jing": 40, "Qi": 55, "Shen": 60},
        "treasure_confidence": {"Jing": 0.5, "Qi": 0.8, "Shen": 0.7},
        "polyvagal_state": {"ventral_vagal": 60, "sympathetic": 30, "dorsal_vagal": 10},
        "congruence": {"score": 0.6,
                       "self_contradictions": ['"felt the weight of everything" vs. "also hope"'],
                       "notes": 'tension between burden and hope'},
        "top_themes": ["resilient hope amid adversity", "rumination and worry"],
    }
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(jb.requests, "post", lambda *a, **k: _Resp(_tool_use_body(analysis)))
    out = jb._haiku_analyze("today was heavy but hopeful", {"word_count": 6})
    assert isinstance(out, dict)
    assert isinstance(out["emotions"], dict) and out["emotions"]["Calmness"] == 0.7
    assert out["elements"]["Fire"] == 60
    # the quote-bearing free text survives intact (would have crashed text parsing)
    assert "self_contradictions" in out["congruence"]


def test_haiku_analyze_forces_the_tool(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _Resp(_tool_use_body({"emotions": {}, "elements": {}, "treasures": {}}))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(jb.requests, "post", fake_post)
    jb._haiku_analyze("hello", {"word_count": 1})
    p = captured["payload"]
    assert p.get("tools"), "must declare the emit_analysis tool"
    assert any(t.get("name") == "emit_analysis" for t in p["tools"])
    assert p.get("tool_choice", {}).get("name") == "emit_analysis", "must force the tool"


def test_haiku_analyze_falls_back_to_text_json(monkeypatch):
    # Defensive: if no tool_use block is present, fall back to tolerant text parsing.
    body = {"content": [{"type": "text", "text": '```json\n{"emotions": {"Joy": 1.0}, '
                         '"elements": {}, "treasures": {}}\n```'}], "stop_reason": "end_turn"}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(jb.requests, "post", lambda *a, **k: _Resp(body))
    out = jb._haiku_analyze("hi", {"word_count": 1})
    assert out["emotions"]["Joy"] == 1.0
