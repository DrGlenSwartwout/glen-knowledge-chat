import sqlite3
from dashboard import portal_view as pv

QUIZ = "https://healing.scoreapp.com"
BASE = "https://illtowell.com"

def _cx_with_signups():
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE affiliate_signups (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, name TEXT,
        email TEXT UNIQUE, slug TEXT UNIQUE, token TEXT, status TEXT DEFAULT 'approved',
        organization TEXT DEFAULT '', short_url TEXT DEFAULT '', referred_by TEXT DEFAULT '')""")
    return cx

def test_enrolled_returns_links():
    cx = _cx_with_signups()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) "
               "VALUES ('t','Amy','amy@example.com','amy7','tok','approved')")
    b = pv._ambassador_block(cx, "amy@example.com", QUIZ, BASE)
    assert b["status"] == "enrolled"
    assert b["slug"] == "amy7"
    assert b["referral_url"] == "https://healing.scoreapp.com?utm_source=amy7&utm_medium=affiliate&utm_campaign=scoreapp-quiz"
    assert b["recruit_url"] == "https://illtowell.com/affiliate?ref=amy7"

def test_pending_status():
    cx = _cx_with_signups()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) "
               "VALUES ('t','Pat','pat@example.com','pat3','tok2','pending')")
    assert pv._ambassador_block(cx, "pat@example.com", QUIZ, BASE) == {"status": "pending"}

def test_not_enrolled_returns_signup_url():
    cx = _cx_with_signups()
    b = pv._ambassador_block(cx, "nobody@example.com", QUIZ, BASE)
    assert b == {"status": "none", "signup_url": "https://illtowell.com/affiliate/apply-form"}

def test_email_lowercased():
    cx = _cx_with_signups()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) "
               "VALUES ('t','Amy','amy@example.com','amy7','tok','approved')")
    assert pv._ambassador_block(cx, "AMY@Example.COM", QUIZ, BASE)["status"] == "enrolled"

def test_missing_table_is_none():
    cx = sqlite3.connect(":memory:")  # no affiliate_signups table
    assert pv._ambassador_block(cx, "x@example.com", QUIZ, BASE)["status"] == "none"

def test_enrolled_includes_dashboard():
    cx = _cx_with_signups()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) "
               "VALUES ('2026-01-01','Amy','amy@example.com','amy7','tok','approved')")
    b = pv._ambassador_block(cx, "amy@example.com", QUIZ, BASE)
    assert b["status"] == "enrolled"
    assert isinstance(b.get("dashboard"), dict)
    assert b["dashboard"].get("slug") == "amy7"

def test_not_enrolled_has_no_dashboard():
    cx = _cx_with_signups()
    b = pv._ambassador_block(cx, "nobody@example.com", QUIZ, BASE)
    assert "dashboard" not in b
