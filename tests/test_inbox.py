"""Tests for dashboard.inbox — Gmail decoding + reply building (no live API calls)."""

import base64
import pytest


def b64url(s: str) -> str:
    """Encode string the way Gmail does: URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")


# ── _decode_b64url ────────────────────────────────────────────────────────────

def test_decode_b64url_handles_unpadded():
    from dashboard.inbox import _decode_b64url
    assert _decode_b64url(b64url("hello world")) == "hello world"

def test_decode_b64url_handles_unicode():
    from dashboard.inbox import _decode_b64url
    s = "Aloha — testing ñ"
    assert _decode_b64url(b64url(s)) == s

def test_decode_b64url_empty_returns_empty():
    from dashboard.inbox import _decode_b64url
    assert _decode_b64url("") == ""


# ── _extract_body ────────────────────────────────────────────────────────────

def test_extract_body_prefers_plain_over_html():
    from dashboard.inbox import _extract_body
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": b64url("PLAIN VERSION")}},
            {"mimeType": "text/html",  "body": {"data": b64url("<p>HTML</p>")}},
        ],
    }
    body = _extract_body(payload)
    assert body["plain"] == "PLAIN VERSION"
    assert "HTML" in body["html"]

def test_extract_body_falls_back_to_html_stripped_when_no_plain():
    from dashboard.inbox import _extract_body
    payload = {
        "mimeType": "text/html",
        "body": {"data": b64url("<p>Hello <b>world</b></p>")},
    }
    body = _extract_body(payload)
    assert body["html"] == "<p>Hello <b>world</b></p>"
    assert "Hello world" in body["plain"]

def test_extract_body_walks_nested_multipart():
    from dashboard.inbox import _extract_body
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {"mimeType": "multipart/alternative", "parts": [
                {"mimeType": "text/plain", "body": {"data": b64url("nested plain")}},
            ]},
            {"mimeType": "application/pdf", "body": {"data": ""}},
        ],
    }
    assert _extract_body(payload)["plain"] == "nested plain"


# ── _header ──────────────────────────────────────────────────────────────────

def test_header_lookup_case_insensitive():
    from dashboard.inbox import _header
    headers = [{"name": "Subject", "value": "Hi"}, {"name": "from", "value": "a@b.com"}]
    assert _header(headers, "subject") == "Hi"
    assert _header(headers, "FROM") == "a@b.com"
    assert _header(headers, "missing") == ""


# ── _summarize_thread ────────────────────────────────────────────────────────

def test_summarize_thread_extracts_subject_and_unread_state():
    from dashboard.inbox import _summarize_thread
    t = {
        "id": "abc",
        "messages": [
            {"payload": {"headers": [
                {"name": "Subject", "value": "Test thread"},
                {"name": "From", "value": "alice@example.com"},
            ]}, "labelIds": ["INBOX"], "snippet": "first msg", "internalDate": "1700000000000"},
            {"payload": {"headers": [
                {"name": "From", "value": "bob@example.com"},
                {"name": "Date", "value": "Mon, 28 Apr 2026"},
            ]}, "labelIds": ["INBOX", "UNREAD"], "snippet": "latest msg", "internalDate": "1714000000000"},
        ],
    }
    s = _summarize_thread(t)
    assert s["id"] == "abc"
    assert s["subject"] == "Test thread"
    assert s["sender"] == "bob@example.com"   # last message's sender
    assert s["snippet"] == "latest msg"
    assert s["msg_count"] == 2
    assert s["unread"] is True
    assert "INBOX" in s["labels"] and "UNREAD" in s["labels"]


def test_summarize_thread_handles_missing_subject():
    from dashboard.inbox import _summarize_thread
    t = {"id": "x", "messages": [{"payload": {"headers": []}, "labelIds": []}]}
    assert _summarize_thread(t)["subject"] == "(no subject)"


def test_summarize_thread_handles_empty_messages():
    from dashboard.inbox import _summarize_thread
    s = _summarize_thread({"id": "x", "messages": []})
    assert s["subject"] == ""
    assert s["msg_count"] == 0


# ── _build_reply_message ─────────────────────────────────────────────────────

def test_build_reply_uses_reply_to_when_present():
    from dashboard.inbox import _build_reply_message
    thread = {"id": "T1", "messages": [{
        "payload": {"headers": [
            {"name": "From", "value": "noreply@platform.io"},
            {"name": "Reply-To", "value": "real-human@platform.io"},
            {"name": "Subject", "value": "Your invoice"},
            {"name": "Message-Id", "value": "<abc@platform.io>"},
        ]},
    }]}
    payload = _build_reply_message(thread, "Thanks!")
    assert payload["threadId"] == "T1"
    raw = base64.urlsafe_b64decode(payload["raw"] + "===").decode("utf-8")
    assert "To: real-human@platform.io" in raw
    assert "Subject: Re: Your invoice" in raw
    assert "In-Reply-To: <abc@platform.io>" in raw
    assert "References: <abc@platform.io>" in raw
    # MIMEText base64-encodes the body; decode it to verify
    assert base64.b64encode(b"Thanks!").decode() in raw

def test_build_reply_falls_back_to_from_when_no_reply_to():
    from dashboard.inbox import _build_reply_message
    thread = {"id": "T2", "messages": [{
        "payload": {"headers": [
            {"name": "From", "value": "alice@example.com"},
            {"name": "Subject", "value": "Re: existing"},
        ]},
    }]}
    payload = _build_reply_message(thread, "ok")
    raw = base64.urlsafe_b64decode(payload["raw"] + "===").decode("utf-8")
    assert "To: alice@example.com" in raw
    # Subject already starts with Re: — no double prefix
    assert "Subject: Re: existing" in raw
    assert "Subject: Re: Re: existing" not in raw

def test_build_reply_override_to_wins():
    from dashboard.inbox import _build_reply_message
    thread = {"id": "T3", "messages": [{
        "payload": {"headers": [{"name": "From", "value": "auto@bot.io"}]},
    }]}
    payload = _build_reply_message(thread, "hi", override_to="real@person.com")
    raw = base64.urlsafe_b64decode(payload["raw"] + "===").decode("utf-8")
    assert "To: real@person.com" in raw

def test_build_reply_empty_thread_raises():
    from dashboard.inbox import _build_reply_message
    with pytest.raises(ValueError):
        _build_reply_message({"id": "X", "messages": []}, "body")


# ── categorize ────────────────────────────────────────────────────────────────

def test_categorize_promotions():
    from dashboard.inbox import categorize
    assert categorize(["INBOX", "CATEGORY_PROMOTIONS", "UNREAD"]) == "promotions"

def test_categorize_important_overrides_category():
    from dashboard.inbox import categorize
    assert categorize(["INBOX", "CATEGORY_PROMOTIONS", "IMPORTANT"]) == "important"

def test_categorize_starred_counts_as_important():
    from dashboard.inbox import categorize
    assert categorize(["INBOX", "STARRED"]) == "important"

def test_categorize_default_inbox():
    from dashboard.inbox import categorize
    assert categorize(["INBOX", "CATEGORY_PERSONAL"]) == "inbox"

def test_categorize_social_updates_forums():
    from dashboard.inbox import categorize
    assert categorize(["INBOX", "CATEGORY_SOCIAL"]) == "social"
    assert categorize(["INBOX", "CATEGORY_UPDATES"]) == "updates"
    assert categorize(["INBOX", "CATEGORY_FORUMS"]) == "forums"


# ── clean_body ────────────────────────────────────────────────────────────────

def test_clean_body_strips_html_tags():
    from dashboard.inbox import clean_body
    assert "Hello world" in clean_body("<p>Hello <b>world</b></p>")
    assert "<" not in clean_body("<p>Hello</p>")

def test_clean_body_strips_quote_history_on_marker():
    from dashboard.inbox import clean_body
    raw = (
        "Aloha Glen,\n\n"
        "I want to ask about the protocol.\n\n"
        "On Mon, Apr 28, 2026 at 12:11 PM Practice Better <noreply@pb.io> wrote:\n"
        "> Earlier message body\n"
        "> More quoted text\n"
    )
    out = clean_body(raw)
    assert "I want to ask about the protocol." in out
    assert "Practice Better" not in out
    assert "Earlier message" not in out

def test_clean_body_strips_outlook_header_block():
    from dashboard.inbox import clean_body
    raw = (
        "My response here.\n\n"
        "From: alice@example.com\n"
        "Sent: Monday\n"
        "To: glen@x.com\n"
        "Subject: thing\n\n"
        "old message"
    )
    out = clean_body(raw)
    assert "My response here." in out
    assert "alice@example.com" not in out

def test_clean_body_strips_signature_delimiter():
    from dashboard.inbox import clean_body
    raw = "The actual message.\n\n--\nGlen Swartwout\n+1 808-555-0100"
    out = clean_body(raw)
    assert "The actual message." in out
    assert "808-555-0100" not in out

def test_clean_body_strips_sent_from_iphone():
    from dashboard.inbox import clean_body
    out = clean_body("Quick note.\n\nSent from my iPhone")
    assert "Quick note." in out
    assert "iPhone" not in out

def test_clean_body_collapses_blank_lines_and_html_entities():
    from dashboard.inbox import clean_body
    out = clean_body("hi&nbsp;there\n\n\n\n\nnext line")
    assert "hi there" in out
    assert "&nbsp;" not in out

def test_clean_body_handles_empty():
    from dashboard.inbox import clean_body
    assert clean_body("") == ""
    assert clean_body(None) == ""


# ── summarize_thread now exposes category ────────────────────────────────────

def test_summarize_thread_includes_category():
    from dashboard.inbox import _summarize_thread
    t = {"id": "x", "messages": [{
        "payload": {"headers": [{"name": "Subject", "value": "Hi"}]},
        "labelIds": ["INBOX", "CATEGORY_PROMOTIONS"],
        "snippet": "promo", "internalDate": "0",
    }]}
    s = _summarize_thread(t)
    assert s["category"] == "promotions"
