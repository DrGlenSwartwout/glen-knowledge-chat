import datetime, sqlite3, uuid
from dashboard import membership_products as mp

def _mk_db(tmp_path):
    cx = sqlite3.connect(tmp_path / "t.db")
    cx.execute("""CREATE TABLE memberships (id TEXT PRIMARY KEY, email TEXT NOT NULL,
        granted_at TEXT NOT NULL, expires_at TEXT, granted_by TEXT, source TEXT,
        truly_vip_ref TEXT, notes TEXT, last_reminder_at TEXT)""")
    cx.commit()
    return cx

def _grant(cx, email, source, days):
    now = datetime.datetime.utcnow()
    exp = (now + datetime.timedelta(days=days)).isoformat()
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,granted_by,source) "
               "VALUES (?,?,?,?,?,?)",
               (uuid.uuid4().hex, email, now.isoformat(), exp, source, source))
    cx.commit()

def test_membership_grant_owns_group(tmp_path):
    cx = _mk_db(tmp_path)
    _grant(cx, "a@x.com", "membership_month", 34)
    assert mp.owns_group(cx, "a@x.com") is True

def test_prepay_grant_does_not_own_group(tmp_path):
    cx = _mk_db(tmp_path)
    _grant(cx, "b@x.com", "prepay_12mo", 369)  # different namespace
    assert mp.owns_group(cx, "b@x.com") is False

def test_expired_membership_grant_does_not_own(tmp_path):
    cx = _mk_db(tmp_path)
    _grant(cx, "c@x.com", "membership_year_prepay", -1)  # already expired
    assert mp.owns_group(cx, "c@x.com") is False
