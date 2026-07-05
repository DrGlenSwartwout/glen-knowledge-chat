import json
from dashboard import community_catalog as _cat


class _FakeTranscription:
    text = "hello world this is the talk"
    segments = [{"start": 0.0, "end": 2.0, "text": "hello world"},
                {"start": 2.0, "end": 5.0, "text": "this is the talk"}]


class _FakeClient:
    """Mimics the openai client surface used by the module."""
    def __init__(self, chat_json):
        self._chat_json = chat_json
        self.audio = self
        self.transcriptions = self
        self.chat = self
        self.completions = self

    def create(self, **kw):
        if kw.get("model") == "whisper-1":
            return _FakeTranscription()
        # chat completion
        class _M:
            def __init__(s, c): s.message = type("x", (), {"content": c})
        return type("R", (), {"choices": [_M(self._chat_json)]})


def test_transcribe_returns_text_and_segments():
    client = _FakeClient("{}")
    out = _cat.transcribe("/tmp/whatever.mp4", client=client)
    assert out["text"].startswith("hello world")
    assert out["segments"][0]["end"] == 2.0


def test_suggest_catalog_parses_json():
    payload = json.dumps({"title": "Sleep and Adrenals",
                          "interest_tags": ["sleep", "adrenals"],
                          "outtakes": [{"start": 2.0, "end": 5.0, "title": "The adrenal tip",
                                        "reason": "punchy standalone tip"}]})
    client = _FakeClient(payload)
    out = _cat.suggest_catalog("hello world this is the talk", client=client)
    assert out["title"] == "Sleep and Adrenals"
    assert out["interest_tags"] == ["sleep", "adrenals"]
    assert out["outtakes"][0]["title"] == "The adrenal tip"


def test_suggest_catalog_degrades_on_bad_json():
    client = _FakeClient("not json")
    out = _cat.suggest_catalog("x", client=client)
    assert out == {"title": "", "interest_tags": [], "outtakes": []}
