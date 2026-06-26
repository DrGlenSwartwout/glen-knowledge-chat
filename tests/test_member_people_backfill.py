import sqlite3
from dashboard import subscriptions as subs

def _cx():
    cx = sqlite3.connect(":memory:")
    cx.executescript("""
      CREATE TABLE subscriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT,
        kind TEXT, status TEXT);
      CREATE TABLE memberships (id TEXT PRIMARY KEY, email TEXT, expires_at TEXT);
      CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        name TEXT DEFAULT '', phone TEXT DEFAULT '', source TEXT DEFAULT '',
        created_at TEXT, updated_at TEXT);
    """)
    return cx

def test_backfill_covers_active_members_missing_people():
    cx = _cx()
    cx.execute("INSERT INTO subscriptions (email,kind,status) VALUES ('paid@x.com','membership','active')")
    cx.execute("INSERT INTO subscriptions (email,kind,status) VALUES ('cancel@x.com','membership','cancelled')")
    cx.execute("INSERT INTO subscriptions (email,kind,status) VALUES ('remedy@x.com','subscription','active')")
    cx.execute("INSERT INTO memberships (id,email,expires_at) VALUES ('m1','grant@x.com','2999-01-01T00:00:00Z')")
    cx.execute("INSERT INTO memberships (id,email,expires_at) VALUES ('m2','expired@x.com','2000-01-01T00:00:00Z')")
    cx.execute("INSERT INTO people (email) VALUES ('paid@x.com')")  # already has people
    n = subs.backfill_member_people(cx)
    emails = {r[0] for r in cx.execute("SELECT email FROM people").fetchall()}
    assert "grant@x.com" in emails        # unexpired grant -> created
    assert "cancel@x.com" not in emails   # cancelled membership -> skipped
    assert "remedy@x.com" not in emails   # non-membership subscription -> skipped
    assert "expired@x.com" not in emails  # expired grant -> skipped
    assert n == 1                         # only grant@ was a member missing people (paid@ already had one)

def test_idempotent():
    cx = _cx()
    cx.execute("INSERT INTO subscriptions (email,kind,status) VALUES ('a@x.com','membership','active')")
    assert subs.backfill_member_people(cx) == 1
    assert subs.backfill_member_people(cx) == 0
