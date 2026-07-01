"""Persistent portal chat thread + Dr-Glen replies."""
import sqlite3

from dashboard import portal_chat as pc


def test_add_and_list_in_order():
    cx = sqlite3.connect(":memory:")
    pc.add_message(cx, "A@X.com", pc.CLIENT, "my tooth hurts", author="Agnes")
    pc.add_message(cx, "a@x.com", pc.ASSISTANT, "let's look at that", author="Ask Dr. Glen")
    pc.add_message(cx, "a@x.com", pc.PRACTITIONER, "Aloha Agnes, here is my review", author="Dr. Glen")
    msgs = pc.list_messages(cx, "a@x.com")
    assert [m["role"] for m in msgs] == [pc.CLIENT, pc.ASSISTANT, pc.PRACTITIONER]
    assert msgs[0]["content"] == "my tooth hurts"          # email is case-insensitive
    assert msgs[2]["author"] == "Dr. Glen"


def test_empty_is_noop():
    cx = sqlite3.connect(":memory:")
    assert pc.add_message(cx, "a@x.com", pc.CLIENT, "   ") is None
    assert pc.add_message(cx, "", pc.CLIENT, "hi") is None
    assert pc.list_messages(cx, "a@x.com") == []


def test_record_exchange_persists_both_sides():
    cx = sqlite3.connect(":memory:")
    pc.record_exchange(cx, "a@x.com", "what remedy?", "Nerve Pulse", client_name="Agnes")
    msgs = pc.list_messages(cx, "a@x.com")
    assert [m["role"] for m in msgs] == [pc.CLIENT, pc.ASSISTANT]
    assert msgs[0]["author"] == "Agnes" and msgs[1]["author"] == "Ask Dr. Glen"


def test_list_limit_returns_recent_in_order():
    cx = sqlite3.connect(":memory:")
    for i in range(10):
        pc.add_message(cx, "a@x.com", pc.CLIENT, f"m{i}")
    msgs = pc.list_messages(cx, "a@x.com", limit=3)
    assert [m["content"] for m in msgs] == ["m7", "m8", "m9"]   # last 3, oldest-first
