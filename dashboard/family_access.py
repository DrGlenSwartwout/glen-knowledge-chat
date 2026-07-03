"""Family accounts + per-scan unlock gate (Portal Access V2).

Each member keeps their own account/email and reports; this module only links
members under a primary and decides per-scan access. Pure sqlite; "now" is passed
in so logic is deterministic under test.
"""
import datetime


def _now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _norm(email):
    return (email or "").strip().lower()


def init_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS family_members (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_email TEXT NOT NULL,
            member_email  TEXT NOT NULL,
            member_label  TEXT,
            member_type   TEXT DEFAULT 'human',
            display_order INTEGER DEFAULT 0,
            created_at    TEXT,
            UNIQUE(primary_email, member_email)
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_family_primary ON family_members(primary_email)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_family_member ON family_members(member_email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS scan_unlocks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            member_email TEXT NOT NULL,
            scan_id      TEXT NOT NULL,
            scan_date    TEXT,
            unlocked_at  TEXT NOT NULL,
            source       TEXT NOT NULL,
            UNIQUE(member_email, scan_id)
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_unlock_member ON scan_unlocks(member_email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS family_memberships (
            primary_email TEXT PRIMARY KEY,
            active        INTEGER NOT NULL DEFAULT 0,
            updated_at    TEXT
        )""")
    cx.commit()


def upsert_member(cx, primary_email, member_email, label=None, member_type="human", display_order=0):
    p, m = _norm(primary_email), _norm(member_email)
    cx.execute(
        "INSERT INTO family_members (primary_email, member_email, member_label, member_type, display_order, created_at) "
        "VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(primary_email, member_email) DO UPDATE SET "
        "member_label=excluded.member_label, member_type=excluded.member_type, display_order=excluded.display_order",
        (p, m, label, member_type, int(display_order or 0), _now_iso()))
    cx.commit()


def remove_member(cx, primary_email, member_email):
    cx.execute("DELETE FROM family_members WHERE primary_email=? AND member_email=?",
               (_norm(primary_email), _norm(member_email)))
    cx.commit()


def list_members(cx, primary_email):
    rows = cx.execute(
        "SELECT member_email, member_label, member_type, display_order FROM family_members "
        "WHERE primary_email=? ORDER BY display_order, member_email", (_norm(primary_email),)).fetchall()
    return [{"member_email": r[0], "member_label": r[1], "member_type": r[2], "display_order": r[3]} for r in rows]


def primary_for(cx, member_email):
    r = cx.execute("SELECT primary_email FROM family_members WHERE member_email=? LIMIT 1",
                   (_norm(member_email),)).fetchone()
    return r[0] if r else None


def is_primary(cx, email):
    r = cx.execute("SELECT 1 FROM family_members WHERE primary_email=? LIMIT 1", (_norm(email),)).fetchone()
    return bool(r)


def has_unlock(cx, member_email, scan_id):
    r = cx.execute("SELECT 1 FROM scan_unlocks WHERE member_email=? AND scan_id=?",
                   (_norm(member_email), str(scan_id))).fetchone()
    return bool(r)


def record_unlock(cx, member_email, scan_id, scan_date, source, now_iso):
    cur = cx.execute(
        "INSERT OR IGNORE INTO scan_unlocks (member_email, scan_id, scan_date, unlocked_at, source) "
        "VALUES (?,?,?,?,?)", (_norm(member_email), str(scan_id), scan_date, now_iso, source))
    cx.commit()
    return cur.rowcount == 1


def free_unlock_used_this_month(cx, member_email, year_month):
    r = cx.execute(
        "SELECT 1 FROM scan_unlocks WHERE member_email=? AND source='free_monthly' "
        "AND substr(unlocked_at,1,7)=? LIMIT 1", (_norm(member_email), year_month)).fetchone()
    return bool(r)


def grant_free_monthly(cx, member_email, scan_id, scan_date, now_iso):
    if has_unlock(cx, member_email, scan_id):
        return True, "already"
    year_month = now_iso[:7]  # "YYYY-MM"
    if free_unlock_used_this_month(cx, member_email, year_month):
        return False, "cap"
    record_unlock(cx, member_email, scan_id, scan_date, "free_monthly", now_iso)
    return True, ""


def set_family_membership(cx, primary_email, active, now_iso):
    p = _norm(primary_email)
    cx.execute(
        "INSERT INTO family_memberships (primary_email, active, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(primary_email) DO UPDATE SET active=excluded.active, updated_at=excluded.updated_at",
        (p, 1 if active else 0, now_iso))
    cx.commit()


def family_is_paid(cx, member_email):
    m = _norm(member_email)
    primary = primary_for(cx, m) or (m if is_primary(cx, m) else None)
    if not primary:
        return False
    r = cx.execute("SELECT active FROM family_memberships WHERE primary_email=?", (primary,)).fetchone()
    return bool(r and r[0])
