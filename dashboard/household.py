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


DEPENDENT_RELATIONSHIPS = {"child", "pet", "dependent", "charge", "caregiving-client"}


def is_dependent(relationship):
    return (relationship or "").strip().lower() in DEPENDENT_RELATIONSHIPS


def default_cc_for(relationship):
    return 1 if is_dependent(relationship) else 0


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

    # v1 sharing/cc columns (additive). share_consent defaults 1 (member shared,
    # revocable). cc_enabled default 0 at the column level, but a brand-new column
    # is backfilled from relationship (dependents → 1) exactly once.
    try:
        cx.execute("ALTER TABLE household_members ADD COLUMN share_consent INTEGER DEFAULT 1")
    except Exception:
        pass
    try:
        cx.execute("ALTER TABLE household_members ADD COLUMN cc_enabled INTEGER DEFAULT 0")
        # column is brand new (ALTER succeeded) → backfill dependents once
        cx.execute(
            "UPDATE household_members SET cc_enabled=1 WHERE lower(coalesce(relationship,'')) IN (%s)"
            % ",".join("?" * len(DEPENDENT_RELATIONSHIPS)), tuple(sorted(DEPENDENT_RELATIONSHIPS)))
    except Exception:
        pass

    for _col, _default in (("consent_basis", "''"), ("consent_recorded_by", "''"),
                           ("consent_confirmed_at", "''")):
        try:
            cx.execute(f"ALTER TABLE household_members ADD COLUMN {_col} TEXT DEFAULT {_default}")
        except Exception:
            pass
    cx.commit()

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


def add_member(cx, primary_email, member_email, label="", relationship="",
               consent_basis=None, consent_by=""):
    p, m = _norm(primary_email), _norm(member_email)
    if not p or not m or p == m:
        return False
    rel = (relationship or "").strip().lower()
    if is_dependent(rel):
        share, basis = 1, "caregiver-authority"
    elif not rel:
        share, basis = 1, ""        # legacy generic link: shared by default (revocable)
    elif consent_basis in ("verbal", "written"):
        share, basis = 1, consent_basis
    else:
        share, basis = 0, ""        # explicit operational word: dark until consent
    cx.execute(
        "INSERT OR IGNORE INTO household_members "
        "(primary_email, member_email, label, relationship, created_at, share_consent, "
        " cc_enabled, consent_basis, consent_recorded_by) VALUES (?,?,?,?,?,?,?,?,?)",
        (p, m, label or "", relationship or "", _now(), share,
         default_cc_for(relationship), basis, consent_by or ""))
    cx.commit()
    return True


def remove_member(cx, primary_email, member_email):
    cx.execute("DELETE FROM household_members WHERE primary_email=? AND member_email=?",
               (_norm(primary_email), _norm(member_email)))
    cx.commit()


def members_for(cx, primary_email):
    rows = cx.execute(
        "SELECT member_email, label, relationship, share_consent, cc_enabled FROM household_members "
        "WHERE primary_email=? ORDER BY created_at, id", (_norm(primary_email),)).fetchall()
    return [{"email": r[0], "label": r[1] or "", "relationship": r[2] or "",
             "share_consent": int(r[3] if r[3] is not None else 1),
             "cc_enabled": int(r[4] if r[4] is not None else 0)} for r in rows]


def can_view(cx, viewer_email, target_email):
    v, t = _norm(viewer_email), _norm(target_email)
    if not v or not t:
        return False
    if v == t:
        return True
    return cx.execute(
        "SELECT 1 FROM household_members WHERE primary_email=? AND member_email=? "
        "AND share_consent=1 LIMIT 1", (v, t)).fetchone() is not None


def set_share_consent(cx, primary_email, member_email, consent):
    cx.execute("UPDATE household_members SET share_consent=? WHERE primary_email=? AND member_email=?",
               (1 if consent else 0, _norm(primary_email), _norm(member_email)))
    cx.commit()


def set_cc_enabled(cx, primary_email, member_email, enabled):
    cx.execute("UPDATE household_members SET cc_enabled=? WHERE primary_email=? AND member_email=?",
               (1 if enabled else 0, _norm(primary_email), _norm(member_email)))
    cx.commit()


def viewable_members_for(cx, primary_email):
    rows = cx.execute(
        "SELECT member_email, label, relationship FROM household_members "
        "WHERE primary_email=? AND share_consent=1 ORDER BY created_at, id",
        (_norm(primary_email),)).fetchall()
    return [{"email": r[0], "label": r[1] or "", "relationship": r[2] or ""} for r in rows]


def cc_recipients_for(cx, member_email):
    rows = cx.execute(
        "SELECT primary_email FROM household_members "
        "WHERE member_email=? AND share_consent=1 AND cc_enabled=1", (_norm(member_email),)).fetchall()
    return [r[0] for r in rows]


def caregivers_for(cx, member_email):
    rows = cx.execute(
        "SELECT primary_email, share_consent, relationship FROM household_members "
        "WHERE member_email=? ORDER BY created_at, id", (_norm(member_email),)).fetchall()
    return [{"primary_email": r[0], "share_consent": int(r[1] if r[1] is not None else 1),
             "relationship": r[2] or ""} for r in rows]


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


def reassign_report(cx, scan_date, from_email, to_email, *, by="console"):
    """Move a portal_biofield_reports row from one household member to another.
    Refuses cross-household moves and refuses to overwrite an existing report on the
    target for that date. Logs to scan_reassignments. Returns {"ok", "error"}."""
    f, t = _norm(from_email), _norm(to_email)
    sd = (scan_date or "").strip()
    if not f or not t or not sd:
        return {"ok": False, "error": "missing scan_date/from/to"}
    if f == t:
        return {"ok": False, "error": "from and to are the same account"}
    if not same_household(cx, f, t):
        return {"ok": False, "error": "from and to are not in the same household"}
    if not cx.execute("SELECT 1 FROM portal_biofield_reports WHERE email=? AND scan_date=? LIMIT 1",
                      (f, sd)).fetchone():
        return {"ok": False, "error": "no report for that account/date"}
    if cx.execute("SELECT 1 FROM portal_biofield_reports WHERE email=? AND scan_date=? LIMIT 1",
                  (t, sd)).fetchone():
        return {"ok": False, "error": "target already has a report for that date"}
    cx.execute("UPDATE portal_biofield_reports SET email=?, updated_at=? WHERE email=? AND scan_date=?",
               (t, _now(), f, sd))
    cx.execute("INSERT INTO scan_reassignments (scan_date, from_email, to_email, by, at) "
               "VALUES (?,?,?,?,?)", (sd, f, t, by, _now()))
    cx.commit()
    return {"ok": True, "error": None}


def list_reassignments(cx, limit=100):
    rows = cx.execute(
        "SELECT scan_date, from_email, to_email, by, at FROM scan_reassignments "
        "ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [{"scan_date": r[0], "from_email": r[1], "to_email": r[2], "by": r[3], "at": r[4]}
            for r in rows]


def consent_state(cx, primary_email, member_email):
    r = cx.execute(
        "SELECT share_consent, consent_basis, consent_confirmed_at FROM household_members "
        "WHERE primary_email=? AND member_email=?",
        (_norm(primary_email), _norm(member_email))).fetchone()
    if not r:
        return {"share_consent": 0, "consent_basis": "", "consent_confirmed_at": ""}
    return {"share_consent": int(r[0] if r[0] is not None else 0),
            "consent_basis": r[1] or "", "consent_confirmed_at": r[2] or ""}


def record_consent(cx, primary_email, member_email, basis, by=""):
    """Record an operator-captured consent basis (verbal/written) on an EXISTING
    link, activating sharing. Never overwrites a dependent's caregiver-authority
    basis. Returns False if the link doesn't exist."""
    p, m = _norm(primary_email), _norm(member_email)
    row = cx.execute("SELECT consent_basis FROM household_members "
                     "WHERE primary_email=? AND member_email=?", (p, m)).fetchone()
    if not row:
        return False
    if (row[0] or "") == "caregiver-authority":
        return True   # dependent link already consented via caregiver authority
    cx.execute("UPDATE household_members SET share_consent=1, consent_basis=?, "
               "consent_recorded_by=? WHERE primary_email=? AND member_email=?",
               (basis, by or "", p, m))
    cx.commit()
    return True


def confirm_consent(cx, primary_email, member_email):
    p, m = _norm(primary_email), _norm(member_email)
    row = cx.execute("SELECT consent_basis, consent_confirmed_at FROM household_members "
                     "WHERE primary_email=? AND member_email=?", (p, m)).fetchone()
    if not row:
        return False
    if (row[0] or "") == "caregiver-authority":
        return True   # dependent link already consented; don't overwrite
    if row[1]:  # already hard-confirmed — idempotent, never downgrade
        return True
    cx.execute("UPDATE household_members SET share_consent=1, consent_basis='portal-confirmed', "
               "consent_confirmed_at=? WHERE primary_email=? AND member_email=?", (_now(), p, m))
    cx.commit()
    return True
