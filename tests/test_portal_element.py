import sqlite3
from dashboard import portal_element as pe
from dashboard import portal_chat, member_element_state as mes


def _cx():
    cx = sqlite3.connect(":memory:")
    portal_chat.init_table(cx)
    mes.init_table(cx)
    return cx


def _seed(cx, email, texts):
    for t in texts:
        portal_chat.add_message(cx, email, "client", t, author="You")


def test_refresh_writes_element_state_from_recent_client_msgs(monkeypatch):
    cx = _cx()
    _seed(cx, "j@x.com", [
        "I keep waking at 3am full of dread and I can't get warm.",
        "Everything feels like too much and I am exhausted to my bones.",
    ])
    monkeypatch.setattr(pe, "_haiku_analyze", lambda transcript, lexical: {
        "emotions": {}, "treasures": {},
        "elements": {"Wood": 40, "Fire": 30, "Earth": 20, "Metal": 25, "Water": 5},
    })
    row = pe.refresh(cx, "j@x.com")
    assert row is not None
    assert row["deficient_element"] == "Water"
    assert mes.get(cx, "j@x.com")["deficient_element"] == "Water"


def test_refresh_skips_when_too_little_text(monkeypatch):
    cx = _cx()
    _seed(cx, "j@x.com", ["ok"])
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("should not analyze")

    monkeypatch.setattr(pe, "_haiku_analyze", _boom)
    assert pe.refresh(cx, "j@x.com") is None
    assert called["n"] == 0


def test_refresh_returns_none_when_no_elements(monkeypatch):
    cx = _cx()
    _seed(cx, "j@x.com", ["I keep waking at 3am full of dread and cannot get warm at all."])
    monkeypatch.setattr(pe, "_haiku_analyze", lambda t, l: {"elements": {}})
    assert pe.refresh(cx, "j@x.com") is None


def test_refresh_only_uses_client_messages(monkeypatch):
    cx = _cx()
    _seed(cx, "j@x.com", ["I keep waking at 3am full of dread and cannot get warm at all."])
    portal_chat.add_message(cx, "j@x.com", "assistant", "IGNORE ME " * 20, author="Ask Dr. Glen")
    seen = {}
    monkeypatch.setattr(pe, "_haiku_analyze",
                        lambda transcript, lexical: seen.update(t=transcript) or {"elements": {"Water": 5, "Wood": 9}})
    pe.refresh(cx, "j@x.com")
    assert "IGNORE ME" not in seen["t"]


def test_analyze_returns_scores_without_writing(monkeypatch):
    cx = _cx()
    _seed(cx, "j@x.com", ["I keep waking at 3am full of dread and cannot get warm at all."])
    monkeypatch.setattr(pe, "_haiku_analyze",
                        lambda t, l: {"elements": {"Wood": 40, "Water": 5}})
    scores = pe.analyze(cx, "j@x.com")
    assert scores == {"Wood": 40, "Water": 5}
    assert mes.get(cx, "j@x.com") is None   # analyze must not persist
