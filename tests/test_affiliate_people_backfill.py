import sqlite3
from dashboard import affiliate_dashboard as ad

def _cx():
    cx = sqlite3.connect(":memory:")
    cx.executescript("""
      CREATE TABLE affiliate_signups (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
        name TEXT, email TEXT, slug TEXT, token TEXT, status TEXT DEFAULT 'approved');
      CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        first_name TEXT DEFAULT '', last_name TEXT DEFAULT '', name TEXT DEFAULT '',
        phone TEXT DEFAULT '', source TEXT DEFAULT '', created_at TEXT, updated_at TEXT);
    """)
    return cx

def test_creates_people_only_for_approved_missing():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) VALUES "
               "('t','Has People','has@x.com','s1','t1','approved')")
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) VALUES "
               "('t','Needs People','need@x.com','s2','t2','approved')")
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) VALUES "
               "('t','Pending Guy','pend@x.com','s3','t3','pending')")
    cx.execute("INSERT INTO people (email, name, created_at, updated_at) VALUES ('has@x.com','Has People','t','t')")
    n = ad.backfill_affiliate_people(cx)
    assert n == 1
    emails = {r[0] for r in cx.execute("SELECT email FROM people").fetchall()}
    assert "need@x.com" in emails          # approved + was missing -> created
    assert "pend@x.com" not in emails      # pending -> skipped
    created = cx.execute("SELECT name FROM people WHERE email='need@x.com'").fetchone()
    assert created[0] == "Needs People"    # name carried from the affiliate row

def test_idempotent():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) VALUES "
               "('t','A','a@x.com','s1','t1','approved')")
    assert ad.backfill_affiliate_people(cx) == 1
    assert ad.backfill_affiliate_people(cx) == 0
