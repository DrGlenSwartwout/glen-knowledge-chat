import sqlite3
from dashboard import email_click_tokens as ect


def _cx():
    cx = sqlite3.connect(":memory:")
    ect.init_email_click_tokens(cx)
    return cx


def test_token_is_opaque_and_contains_no_email():
    cx = _cx()
    t = ect.token_for(cx, "Alice@Example.com")
    assert t and isinstance(t, str) and len(t) >= 16
    assert "alice" not in t.lower() and "@" not in t and "example" not in t.lower()


def test_token_is_stable_and_idempotent_per_email():
    cx = _cx()
    t1 = ect.token_for(cx, "alice@example.com")
    t2 = ect.token_for(cx, "  Alice@example.com  ")   # normalized to same email
    assert t1 == t2
    n = cx.execute("SELECT COUNT(*) FROM email_click_tokens WHERE email=?",
                   ("alice@example.com",)).fetchone()[0]
    assert n == 1


def test_distinct_emails_get_distinct_tokens():
    cx = _cx()
    assert ect.token_for(cx, "a@x.com") != ect.token_for(cx, "b@x.com")


def test_email_for_resolves_and_normalizes():
    cx = _cx()
    t = ect.token_for(cx, "Bob@Example.com")
    assert ect.email_for(cx, t) == "bob@example.com"


def test_email_for_unknown_or_blank_is_none():
    cx = _cx()
    assert ect.email_for(cx, "nope") is None
    assert ect.email_for(cx, "") is None
    assert ect.email_for(cx, None) is None
