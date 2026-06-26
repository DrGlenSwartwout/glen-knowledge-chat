import sqlite3
from dashboard import affiliate_dashboard as ad

QUIZ = "https://healing.scoreapp.com"
BASE = "https://illtowell.com"

def _cx():
    cx = sqlite3.connect(":memory:")
    cx.executescript("""
      CREATE TABLE affiliate_signups (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
        name TEXT, email TEXT, organization TEXT DEFAULT '', slug TEXT, token TEXT,
        status TEXT DEFAULT 'approved', short_url TEXT DEFAULT '', referred_by TEXT DEFAULT '');
      CREATE TABLE referral_events (id INTEGER PRIMARY KEY AUTOINCREMENT, utm_source TEXT,
        received_at TEXT, first_name TEXT, last_name TEXT, quiz_score INTEGER);
      CREATE TABLE affiliate_conversions (id INTEGER PRIMARY KEY AUTOINCREMENT, affiliate_slug TEXT);
      CREATE TABLE affiliate_offers (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
        description TEXT, url_template TEXT, instructions TEXT, active INTEGER DEFAULT 1, sort_order INTEGER DEFAULT 0);
      CREATE TABLE affiliate_social_links (id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT,
        url TEXT, points INTEGER, views INTEGER, likes INTEGER, shares INTEGER, ts TEXT);
    """)
    return cx

def _seed(cx):
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,organization,slug,token,status) "
               "VALUES ('2026-01-01','Amy','amy@x.com','AmyCo','amy7','tok','approved')")
    cx.execute("INSERT INTO referral_events (utm_source,received_at,first_name,last_name,quiz_score) "
               "VALUES ('amy7','2026-06-01','Mary','Johnson',80)")
    cx.execute("INSERT INTO referral_events (utm_source,received_at,first_name,last_name,quiz_score) "
               "VALUES ('amy7','2026-06-10','Bob','Lee',55)")
    cx.execute("INSERT INTO affiliate_conversions (affiliate_slug) VALUES ('amy7')")
    cx.execute("INSERT INTO affiliate_offers (name,description,url_template,instructions,active,sort_order) "
               "VALUES ('Quiz','Take the quiz','https://q/{slug}','Share it',1,0)")
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status,referred_by) "
               "VALUES ('2026-02-01','Rec','r@x.com','rec1','tok2','approved','amy7')")
    cx.commit()

def test_build_dashboard_full():
    cx = _cx(); _seed(cx)
    d = ad.build_dashboard(cx, "amy7", quiz_url=QUIZ, public_base_url=BASE)
    assert d["name"] == "Amy" and d["organization"] == "AmyCo" and d["slug"] == "amy7"
    assert d["tracking_url"] == "https://healing.scoreapp.com?utm_source=amy7&utm_medium=affiliate&utm_campaign=scoreapp-quiz"
    assert d["recruit_url"] == "https://illtowell.com/affiliate?ref=amy7"
    assert d["total_leads"] == 2
    assert d["last_lead"] == "2026-06-10"
    assert d["recruited_count"] == 1
    assert d["conversions_count"] == 1
    assert d["member_since"] == "2026-01-01"
    assert d["offers"][0]["url"] == "https://q/amy7"
    names = [r["name"] for r in d["recent"]]
    assert "Mary J." in names and "Bob L." in names

def test_short_url_wins():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status,short_url) "
               "VALUES ('2026-01-01','Amy','a@x.com','amy7','tok','approved','https://sho.rt/x')")
    cx.commit()
    assert ad.build_dashboard(cx, "amy7", quiz_url=QUIZ, public_base_url=BASE)["tracking_url"] == "https://sho.rt/x"

def test_unknown_slug_empty():
    cx = _cx()
    assert ad.build_dashboard(cx, "nope", quiz_url=QUIZ, public_base_url=BASE) == {}

def test_mask_lead_name():
    assert ad._mask_lead_name("Mary", "Johnson") == "Mary J."
    assert ad._mask_lead_name("Mary", "") == "Mary"
    assert ad._mask_lead_name(None, None) == ""
