import json

from dashboard import homework_analysis as ha


class _FakeContent:
    def __init__(self, text): self.text = text
class _FakeResp:
    def __init__(self, text): self.content = [_FakeContent(text)]
class _FakeMessages:
    def __init__(self, text): self._t = text
    def create(self, **k): return _FakeResp(self._t)
class _FakeClient:
    def __init__(self, text): self.messages = _FakeMessages(text)


class _CapturingMessages:
    """Stub messages.create() that records the prompt it was sent and returns
    a canned valid-JSON response."""
    def __init__(self, captured):
        self._captured = captured
    def create(self, **k):
        self._captured["system"] = k.get("system")
        self._captured["user"] = k["messages"][0]["content"]
        return _FakeResp('{"rating": "Solid", "feedback": "Keep going."}')
class _CapturingClient:
    def __init__(self, captured):
        self.messages = _CapturingMessages(captured)


def test_analyze_parses_rating_and_feedback(monkeypatch):
    monkeypatch.setattr(ha, "_client",
                        lambda: _FakeClient('{"rating": "Solid", "feedback": "Go deeper on X."}'))
    out = ha.analyze("02-body", "Reflect on your body.", "I feel tension in my shoulders.")
    assert out == {"rating": "Solid", "feedback": "Go deeper on X."}


def test_analyze_survives_bad_json(monkeypatch):
    monkeypatch.setattr(ha, "_client", lambda: _FakeClient("not json at all"))
    out = ha.analyze("02-body", "Reflect.", "something")
    assert out["rating"] == "" and isinstance(out["feedback"], str)  # graceful fallback, no raise


def test_analyze_never_raises_on_client_error(monkeypatch):
    def boom(): raise RuntimeError("no api key")
    monkeypatch.setattr(ha, "_client", boom)
    assert ha.analyze("02-body", "Reflect.", "x") == {"rating": "", "feedback": ""}


def test_empty_submission_skips_model(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(ha, "_client", lambda: called.__setitem__("n", called["n"] + 1) or _FakeClient("{}"))
    out = ha.analyze("02-body", "Reflect.", "   ")
    assert called["n"] == 0 and out["rating"] == ""  # no model call for empty submission
    assert out == {"rating": "", "feedback": ""}


def test_bodymap_payload_sent_as_readable(monkeypatch):
    import bodymap_store
    zone = bodymap_store.zone_ids("organs")[0]
    payload = json.dumps({
        "system": "organs",
        "marks": [{"zone": zone, "note": "tight"}],
        "note": "stressed",
    })
    captured = {}
    monkeypatch.setattr(ha, "_client", lambda: _CapturingClient(captured))
    out = ha.analyze("02-body", "Reflect on your body.", payload)
    assert out == {"rating": "Solid", "feedback": "Keep going."}
    sent = captured["user"]
    assert "Organs" in sent
    assert "tight" in sent
    assert "stressed" in sent
    # the raw JSON/zone-id shape must not leak into the prompt
    assert "organ-" not in sent
    assert '"marks"' not in sent


def test_free_text_submission_unchanged(monkeypatch):
    captured = {}
    monkeypatch.setattr(ha, "_client", lambda: _CapturingClient(captured))
    text = "I feel tension in my shoulders and jaw."
    ha.analyze("02-body", "Reflect on your body.", text)
    assert text in captured["user"]
