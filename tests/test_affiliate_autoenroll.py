import sqlite3
from dashboard import affiliate_dashboard as ad

def _db():
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE affiliate_signups (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE, organization TEXT DEFAULT '', website TEXT DEFAULT '',
        promo_method TEXT DEFAULT '', slug TEXT NOT NULL UNIQUE, token TEXT NOT NULL UNIQUE,
        status TEXT DEFAULT 'approved', notes TEXT DEFAULT '', referred_by TEXT DEFAULT '',
        short_url TEXT DEFAULT '', gifting_activated_at TEXT)""")
    cx.execute("""CREATE TABLE referral_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, name TEXT NOT NULL,
        slug TEXT NOT NULL UNIQUE, description TEXT DEFAULT '', utm_source TEXT NOT NULL,
        utm_medium TEXT DEFAULT 'referral', utm_campaign TEXT DEFAULT '', active INTEGER DEFAULT 1)""")
    return cx

def test_ensure_affiliate_creates_approved_row_once():
    cx = _db()
    r1 = ad.ensure_affiliate(cx, "Jo@Example.com", name="Jo Rae")
    assert r1["status"] == "approved"
    assert r1["slug"] and r1["short_url"] == ""
    assert r1["email"] == "jo@example.com"
    r2 = ad.ensure_affiliate(cx, "jo@example.com", name="Jo Rae")
    assert r2["slug"] == r1["slug"]
    assert cx.execute("SELECT COUNT(*) FROM affiliate_signups").fetchone()[0] == 1
    assert cx.execute("SELECT COUNT(*) FROM referral_sources WHERE slug=?", (r1["slug"],)).fetchone()[0] == 1

def test_ensure_affiliate_empty_email_returns_none():
    cx = _db()
    assert ad.ensure_affiliate(cx, "", name="x") is None

def test_ensure_affiliate_slug_collision_suffixes():
    cx = _db()
    cx.execute("INSERT INTO affiliate_signups (created_at,name,email,slug,token,status) "
               "VALUES ('t','x','x@x.com','jo-rae','tok0','approved')")
    r = ad.ensure_affiliate(cx, "jo2@example.com", name="Jo Rae")
    assert r["slug"] != "jo-rae" and r["slug"].startswith("jo-rae-")

def test_ensure_affiliate_no_name_uses_email_localpart():
    cx = _db()
    r = ad.ensure_affiliate(cx, "solo@example.com")
    assert r["slug"] == "solo"

def test_autoenroll_flag(monkeypatch):
    monkeypatch.delenv("AFFILIATE_AUTOENROLL_ENABLED", raising=False)
    assert ad.autoenroll_enabled() is False
    monkeypatch.setenv("AFFILIATE_AUTOENROLL_ENABLED", "true")
    assert ad.autoenroll_enabled() is True

def test_ensure_affiliate_persists_across_connections(tmp_path):
    import sqlite3
    dbp = str(tmp_path / "aff.db")
    cx = sqlite3.connect(dbp)
    cx.execute("""CREATE TABLE affiliate_signups (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE, organization TEXT DEFAULT '', website TEXT DEFAULT '',
        promo_method TEXT DEFAULT '', slug TEXT NOT NULL UNIQUE, token TEXT NOT NULL UNIQUE,
        status TEXT DEFAULT 'approved', notes TEXT DEFAULT '', referred_by TEXT DEFAULT '',
        short_url TEXT DEFAULT '', gifting_activated_at TEXT)""")
    cx.execute("""CREATE TABLE referral_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, name TEXT NOT NULL,
        slug TEXT NOT NULL UNIQUE, description TEXT DEFAULT '', utm_source TEXT NOT NULL,
        utm_medium TEXT DEFAULT 'referral', utm_campaign TEXT DEFAULT '', active INTEGER DEFAULT 1)""")
    cx.commit()
    ad.ensure_affiliate(cx, "p@x.com", name="P")
    cx2 = sqlite3.connect(dbp)  # separate connection: only sees COMMITTED rows
    assert cx2.execute("SELECT COUNT(*) FROM affiliate_signups WHERE email='p@x.com'").fetchone()[0] == 1

from dashboard import portal_view as pv

def test_ambassador_block_autoenrolls_when_flag_on(monkeypatch):
    cx = _db()
    monkeypatch.setenv("AFFILIATE_AUTOENROLL_ENABLED", "true")
    block = pv._ambassador_block(cx, "new@example.com", "https://q.example/quiz", "https://illtowell.com")
    assert block["status"] == "enrolled"
    assert block["slug"]
    assert block["referral_url"].startswith("https://q.example/quiz?utm_source=")
    assert block["recruit_url"] == f"https://illtowell.com/affiliate?ref={block['slug']}"
    assert cx.execute("SELECT COUNT(*) FROM affiliate_signups WHERE lower(email)='new@example.com'").fetchone()[0] == 1

def test_ambassador_block_shows_cta_when_flag_off(monkeypatch):
    cx = _db()
    monkeypatch.delenv("AFFILIATE_AUTOENROLL_ENABLED", raising=False)
    block = pv._ambassador_block(cx, "new@example.com", "https://q.example/quiz", "https://illtowell.com")
    assert block["status"] == "none"
    assert block["signup_url"] == "https://illtowell.com/affiliate/apply-form"
    assert cx.execute("SELECT COUNT(*) FROM affiliate_signups").fetchone()[0] == 0

def test_backfill_from_people_and_portals_idempotent():
    cx = _db()
    cx.execute("CREATE TABLE client_portals (id INTEGER PRIMARY KEY, email TEXT, name TEXT)")
    cx.execute("INSERT INTO client_portals (email, name) VALUES ('a@x.com','A'),('b@x.com','B'),('a@x.com','A2')")
    n1 = ad.backfill_affiliates_from_people(cx)
    assert n1 == 2
    n2 = ad.backfill_affiliates_from_people(cx)
    assert n2 == 0
    assert cx.execute("SELECT COUNT(*) FROM affiliate_signups").fetchone()[0] == 2
