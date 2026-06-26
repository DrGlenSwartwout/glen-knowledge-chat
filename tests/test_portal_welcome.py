import sqlite3
from dashboard import portal_welcome as pw


def _cx():
    return sqlite3.connect(":memory:")


def test_mark_welcome_sent_idempotent_case_insensitive():
    cx = _cx()
    assert pw.mark_welcome_sent(cx, "Member@X.com") is True   # first time → send
    assert pw.mark_welcome_sent(cx, "member@x.com") is False  # already sent (case-insensitive)
    assert pw.mark_welcome_sent(cx, "MEMBER@X.COM") is False
    # a different member is independent
    assert pw.mark_welcome_sent(cx, "other@x.com") is True


def test_mark_welcome_sent_blank_email():
    cx = _cx()
    assert pw.mark_welcome_sent(cx, "") is False
    assert pw.mark_welcome_sent(cx, None) is False
    assert pw.mark_welcome_sent(cx, "   ") is False


def test_content_includes_first_name_and_login_url():
    subject, text, html = pw.welcome_email_content("Karin Takahashi", "https://illtowell.com/portal/login")
    assert subject
    assert "Aloha Karin," in text
    assert "Aloha Karin," in html
    assert "https://illtowell.com/portal/login" in text
    assert "https://illtowell.com/portal/login" in html


def test_content_falls_back_when_no_name_or_email_as_name():
    # blank name → generic greeting
    _, text, _ = pw.welcome_email_content("", "https://x/portal/login")
    assert "Aloha," in text
    # an email-shaped "name" must NOT leak into the greeting
    _, text2, html2 = pw.welcome_email_content("this.elf@gmail.com", "https://x/portal/login")
    assert "Aloha," in text2
    assert "this.elf" not in text2
    assert "this.elf" not in html2


def test_content_has_no_token_or_secret():
    _, text, html = pw.welcome_email_content("Bob", "https://x/portal/login")
    # self-serve link only — no embedded token query string
    assert "token=" not in text
    assert "token=" not in html
