import json
from dashboard import tcm_analysis as tcm


class _Resp:
    def __init__(self, body):
        self._body = body
        self.ok = True
        self.status_code = 200
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _tool_body(analysis):
    return {"content": [{"type": "tool_use", "name": "emit_analysis", "input": analysis}],
            "stop_reason": "tool_use"}


def test_haiku_analyze_returns_parsed_elements(monkeypatch):
    analysis = {"emotions": {"Calmness": 0.7},
                "elements": {"Wood": 10, "Fire": 60, "Earth": 20, "Metal": 5, "Water": 5},
                "treasures": {"Qi": 55}}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(tcm.requests, "post", lambda *a, **k: _Resp(_tool_body(analysis)))
    out = tcm._haiku_analyze("today was heavy but hopeful", {"word_count": 6})
    assert out["elements"]["Fire"] == 60
    assert set(out["elements"]) == {"Wood", "Fire", "Earth", "Metal", "Water"}


def test_haiku_analyze_forces_the_tool(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _Resp(_tool_body({"emotions": {}, "elements": {}, "treasures": {}}))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(tcm.requests, "post", fake_post)
    tcm._haiku_analyze("hello", {"word_count": 1})
    assert captured["payload"]["tool_choice"]["name"] == "emit_analysis"
