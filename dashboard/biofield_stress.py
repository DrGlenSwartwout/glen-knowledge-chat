"""Per-test master stress list + remedy<->stress coverage map for the local
Biofield Intake balancing loop (B1). Pure sqlite; the caller passes a connection.
Balanced state is DERIVED at read time, never stored (see list_stresses)."""
import re
import sqlite3
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


def covered_codes(cx, tid, remedy_names):
    t = _num(tid)
    names = [(n or "").strip().lower() for n in (remedy_names or []) if (n or "").strip()]
    if not names:
        return set()
    ph = ",".join("?" for _ in names)
    rows = cx.execute(
        f"SELECT DISTINCT code FROM biofield_auth_remedy_coverage "
        f"WHERE test_id=? AND remedy IN ({ph})", (t, *names)).fetchall()
    return {r[0] for r in rows}


def _coverers(cx, tid, code, remedy_names):
    t = _num(tid)
    names = [(n or "").strip().lower() for n in (remedy_names or []) if (n or "").strip()]
    if not names:
        return []
    ph = ",".join("?" for _ in names)
    rows = cx.execute(
        f"SELECT remedy FROM biofield_auth_remedy_coverage "
        f"WHERE test_id=? AND code=? AND remedy IN ({ph})", (t, code, *names)).fetchall()
    return [r[0] for r in rows]


def _norm(s):
    """Normalize a stress label for dedup/label-match: lowercase, collapse internal
    whitespace, strip surrounding non-word characters."""
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return re.sub(r"^[^\w]+|[^\w]+$", "", s)


def _chain_parts(chain_rows):
    """Split a mixed chain-rows list into (remedy_names, [(norm_head, remedy), ...]).
    Accepts plain remedy-name strings (no head) and {"head","remedy"} dicts."""
    names, heads = [], []
    for r in chain_rows or []:
        if isinstance(r, str):
            if r.strip():
                names.append(r)
        elif isinstance(r, dict):
            rem = (r.get("remedy") or "").strip()
            if rem:
                names.append(rem)
                h = _norm(r.get("head") or "")
                if h:
                    heads.append((h, rem))
    return names, heads


def list_stresses(cx, tid, chain_rows):
    init_stress_tables(cx)
    cx.row_factory = sqlite3.Row
    t = _num(tid)
    remedy_names, head_pairs = _chain_parts(chain_rows)
    covered = covered_codes(cx, tid, remedy_names)
    head_map = {}
    for h, rem in head_pairs:
        head_map.setdefault(h, rem)
    rows = cx.execute(
        "SELECT id, code, label, source, balance, manual_balanced "
        "FROM biofield_auth_stress WHERE test_id=? ORDER BY "
        "CASE balance WHEN 'required' THEN 0 ELSE 1 END, id", (t,)).fetchall()
    active, balanced = [], []
    for r in rows:
        is_cov = r["code"] in covered
        lbl_rem = head_map.get(_norm(r["label"]))
        is_bal = bool(r["manual_balanced"]) or is_cov or (lbl_rem is not None)
        if is_cov:
            cvs = _coverers(cx, tid, r["code"], remedy_names)
            by = cvs[0] if cvs else ""
        elif lbl_rem is not None:
            by = lbl_rem
        elif r["manual_balanced"]:
            by = "manual"
        else:
            by = ""
        item = {"id": r["id"], "code": r["code"], "label": r["label"],
                "source": r["source"], "balance": r["balance"],
                "balanced": is_bal, "balanced_by": by}
        (balanced if is_bal else active).append(item)
    return {"active": active, "balanced": balanced}


def set_manual_balanced(cx, tid, stress_id, value):
    cx.execute("UPDATE biofield_auth_stress SET manual_balanced=?, updated_at=? "
               "WHERE id=? AND test_id=?",
               (1 if value else 0, _now(), stress_id, _num(tid)))
    cx.commit()


def add_voice_stress(cx, tid, label):
    """Add a voice-captured stress (required) unless its normalized label already
    exists for this test (any source) -> merge. Returns True if inserted."""
    init_stress_tables(cx)
    t = _num(tid)
    n = _norm(label)
    if not n:
        return False
    existing = cx.execute("SELECT label FROM biofield_auth_stress WHERE test_id=?", (t,)).fetchall()
    if any(_norm(r[0]) == n for r in existing):
        return False
    now = _now()
    cx.execute(
        "INSERT INTO biofield_auth_stress(test_id,code,label,source,balance,"
        "manual_balanced,created_at,updated_at) VALUES(?,?,?,'voice','required',0,?,?)",
        (t, n, (label or "").strip(), now, now))
    cx.commit()
    return True
