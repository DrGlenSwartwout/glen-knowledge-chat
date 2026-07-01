"""Triage store + classifier for portal-chat messages needing Dr. Glen."""
import json
import sqlite3

from dashboard import portal_triage as pt


def _complete(payload):
    return lambda system, user: json.dumps(payload)


def test_classify_flags_and_normalizes():
    res = pt.classify(_complete({"needs_attention": True, "category": "health-concern",
                                 "urgency": "high", "summary": "growing armpit lesion",
                                 "recommendation": "review photo, urge assessment"}),
                      "I have a growing red lump under my arm, is it cancer?")
    assert res["category"] == "health-concern" and res["urgency"] == "high"
    assert "armpit" in res["summary"]


def test_classify_returns_none_when_no_attention():
    assert pt.classify(_complete({"needs_attention": False}), "thanks, that helps!") is None
    assert pt.classify(_complete({"garbage": 1}), "hello") is None      # bad json shape -> None
    assert pt.classify(_complete({"needs_attention": True}), "   ") is None  # empty query


def test_classify_defaults_bad_enum_values():
    res = pt.classify(_complete({"needs_attention": True, "category": "banana",
                                 "urgency": "urgent", "summary": "", "recommendation": ""}),
                      "please help")
    assert res["category"] == "other" and res["urgency"] == "medium"
    assert res["summary"] == "(no summary)" and res["recommendation"] == "(no recommendation)"


def test_store_add_list_count_resolve():
    cx = sqlite3.connect(":memory:")
    i1 = pt.add_item(cx, "A@X.com", "Agnes", "request", "high", "wants consult",
                     "set one up", "I'd like a consultation")
    pt.add_item(cx, "b@x.com", "Bob", "bug", "low", "portal link broken", "check token", "link 404s")
    assert pt.open_count(cx) == 2
    items = pt.list_open(cx)
    assert items[0]["email"] == "b@x.com"          # most recent first
    assert any(it["name"] == "Agnes" for it in items)
    pt.resolve(cx, i1)
    assert pt.open_count(cx) == 1
