"""Household / caregiver→member links + within-household scan reassignment.

A caregiver (primary_email) links to member scan accounts (member_email). The
primary may VIEW any linked member's portal; the owner may REASSIGN a mis-attributed
portal report among the members of one household. Lives in LOG_DB (SQLite). Every
scan-subject already has its own E4L email, so members are ordinary email accounts."""
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(e):
    return (e or "").strip().lower()


def init_household_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS household_members (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_email TEXT NOT NULL,
            member_email  TEXT NOT NULL,
            label         TEXT,
            relationship  TEXT,
            created_at    TEXT,
            UNIQUE(primary_email, member_email)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_hm_primary ON household_members(primary_email)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_hm_member ON household_members(member_email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS scan_reassignments (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date  TEXT,
            from_email TEXT,
            to_email   TEXT,
            by         TEXT,
            at         TEXT
        )
    """)
    cx.commit()


def add_member(cx, primary_email, member_email, label="", relationship=""):
    p, m = _norm(primary_email), _norm(member_email)
    if not p or not m or p == m:
        return False
    cx.execute(
        "INSERT OR IGNORE INTO household_members "
        "(primary_email, member_email, label, relationship, created_at) VALUES (?,?,?,?,?)",
        (p, m, label or "", relationship or "", _now()))
    cx.commit()
    return True


def remove_member(cx, primary_email, member_email):
    cx.execute("DELETE FROM household_members WHERE primary_email=? AND member_email=?",
               (_norm(primary_email), _norm(member_email)))
    cx.commit()


def members_for(cx, primary_email):
    rows = cx.execute(
        "SELECT member_email, label, relationship FROM household_members "
        "WHERE primary_email=? ORDER BY created_at, id", (_norm(primary_email),)).fetchall()
    return [{"email": r[0], "label": r[1] or "", "relationship": r[2] or ""} for r in rows]


def can_view(cx, viewer_email, target_email):
    v, t = _norm(viewer_email), _norm(target_email)
    if not v or not t:
        return False
    if v == t:
        return True
    return cx.execute(
        "SELECT 1 FROM household_members WHERE primary_email=? AND member_email=? LIMIT 1",
        (v, t)).fetchone() is not None


def same_household(cx, a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if cx.execute(
        "SELECT 1 FROM household_members WHERE (primary_email=? AND member_email=?) "
        "OR (primary_email=? AND member_email=?) LIMIT 1", (a, b, b, a)).fetchone():
        return True
    return cx.execute(
        "SELECT 1 FROM household_members h1 JOIN household_members h2 "
        "ON h1.primary_email=h2.primary_email "
        "WHERE h1.member_email=? AND h2.member_email=? LIMIT 1", (a, b)).fetchone() is not None
