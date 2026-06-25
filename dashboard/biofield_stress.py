"""Per-test master stress list + remedy<->stress coverage map for the local
Biofield Intake balancing loop (B1). Pure sqlite; the caller passes a connection.
Balanced state is DERIVED at read time, never stored (see list_stresses)."""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(tid):
    return int(str(tid).lstrip("a") or 0)


def init_stress_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_stress(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, code TEXT, label TEXT,
        source TEXT NOT NULL DEFAULT 'scan', balance TEXT NOT NULL DEFAULT 'optional',
        manual_balanced INTEGER NOT NULL DEFAULT 0, created_at TEXT, updated_at TEXT,
        UNIQUE(test_id, source, code))""")
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_remedy_coverage(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, remedy TEXT, code TEXT,
        UNIQUE(test_id, remedy, code))""")
    cx.commit()


def seed_from_scan(cx, tid, findings, coverage):
    init_stress_tables(cx)
    t = _num(tid)
    covered = set()
    for codes in (coverage or {}).values():
        covered |= set(codes)
    now = _now()
    req = 0
    for f in findings or []:
        code = (f.get("code") or "").strip()
        if not code:
            continue
        balance = "required" if code in covered else "optional"
        if balance == "required":
            req += 1
        cx.execute(
            "INSERT INTO biofield_auth_stress(test_id,code,label,source,balance,"
            "manual_balanced,created_at,updated_at) VALUES(?,?,?,'scan',?,0,?,?) "
            "ON CONFLICT(test_id,source,code) DO UPDATE SET "
            "label=excluded.label, balance=excluded.balance, updated_at=excluded.updated_at",
            (t, code, (f.get("name") or code).strip(), balance, now, now))
    cx.execute("DELETE FROM biofield_auth_remedy_coverage WHERE test_id=?", (t,))
    for remedy, codes in (coverage or {}).items():
        for code in codes:
            cx.execute("INSERT OR IGNORE INTO biofield_auth_remedy_coverage(test_id,remedy,code) "
                       "VALUES(?,?,?)", (t, (remedy or "").strip().lower(), code))
    cx.commit()
    n = cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=? AND source='scan'", (t,)).fetchone()[0]
    c = cx.execute("SELECT COUNT(*) FROM biofield_auth_remedy_coverage WHERE test_id=?", (t,)).fetchone()[0]
    return {"stresses": n, "required": req, "coverage": c}
