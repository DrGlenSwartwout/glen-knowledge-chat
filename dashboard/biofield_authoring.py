"""Increment 4a: native authoring store for the local Biofield Analysis tool.

Lets Glen author a biofield test in the app instead of FileMaker. A test is a
header + causal-chain rows entered directly (streamlined vs FMP's stress->promote
flow). `authored_report` returns the SAME shape as `biofield_report.causal_chain_report`
so the schedule, narrative, and your-voice audio all work on authored tests unchanged.

Authored test ids are prefixed "a" (e.g. "a7") so the viewer can tell them apart
from the numeric FMP-snapshot ids. Local + writable; PHI stays on the Mac.
"""
import datetime
import sqlite3

from dashboard.biofield_schedule import build_schedule


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _num(tid):
    return int(str(tid).lstrip("a") or 0)


def init_auth_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_tests(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT,
        date_test TEXT, created_at TEXT, updated_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_chain(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, layer INTEGER,
        head TEXT, most_affected TEXT, remedy TEXT, dosage TEXT, frequency TEXT,
        timing TEXT, sort_seq INTEGER, created_at TEXT)""")
    cx.commit()


def create_test(cx, name, email, date):
    init_auth_tables(cx)
    cur = cx.execute(
        "INSERT INTO biofield_auth_tests(name,email,date_test,created_at,updated_at) "
        "VALUES(?,?,?,?,?)",
        ((name or "").strip(), (email or "").strip().lower(), (date or "").strip(),
         _now(), _now()))
    cx.commit()
    return "a" + str(cur.lastrowid)


def update_header(cx, tid, name=None, email=None, date=None):
    init_auth_tables(cx)
    sets, vals = [], []
    if name is not None:
        sets.append("name=?"); vals.append((name or "").strip())
    if email is not None:
        sets.append("email=?"); vals.append((email or "").strip().lower())
    if date is not None:
        sets.append("date_test=?"); vals.append((date or "").strip())
    if not sets:
        return
    sets.append("updated_at=?"); vals.append(_now())
    vals.append(_num(tid))
    cx.execute(f"UPDATE biofield_auth_tests SET {','.join(sets)} WHERE id=?", vals)
    cx.commit()


def add_chain_row(cx, tid, layer, head, most_affected, remedy,
                  dosage="", frequency="", timing=""):
    init_auth_tables(cx)
    cur = cx.execute(
        "INSERT INTO biofield_auth_chain(test_id,layer,head,most_affected,remedy,"
        "dosage,frequency,timing,sort_seq,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (_num(tid), layer, (head or "").strip(), (most_affected or "").strip(),
         (remedy or "").strip(), dosage or "", frequency or "", timing or "", 0, _now()))
    cx.commit()
    return cur.lastrowid


def update_chain_row(cx, rid, **fields):
    cols = ("layer", "head", "most_affected", "remedy", "dosage", "frequency", "timing")
    sets, vals = [], []
    for k in cols:
        if k in fields:
            sets.append(f"{k}=?"); vals.append(fields[k])
    if not sets:
        return
    vals.append(rid)
    cx.execute(f"UPDATE biofield_auth_chain SET {','.join(sets)} WHERE id=?", vals)
    cx.commit()


def delete_chain_row(cx, rid):
    cx.execute("DELETE FROM biofield_auth_chain WHERE id=?", (rid,))
    cx.commit()


def list_authored(cx):
    init_auth_tables(cx)
    cx.row_factory = sqlite3.Row
    rows = cx.execute("""
        SELECT t.id, t.name, t.email, t.date_test,
          (SELECT COUNT(*) FROM biofield_auth_chain c
             WHERE c.test_id=t.id AND TRIM(COALESCE(c.remedy,''))<>'') AS lc
        FROM biofield_auth_tests t ORDER BY t.id DESC""").fetchall()
    return [{"test_id": "a" + str(r["id"]), "name": r["name"] or "(unnamed)",
             "email": r["email"] or "", "date": r["date_test"] or "",
             "layer_count": r["lc"], "authored": True} for r in rows]


def _has(cx, table):
    return cx.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                      (table,)).fetchone() is not None


def remedy_catalog(cx, q="", limit=20):
    """Search the product catalog (from the snapshot) for the remedy picker."""
    if not _has(cx, "fmp_snap_products"):
        return []
    cx.row_factory = sqlite3.Row
    like = f"%{(q or '').strip()}%"
    rows = cx.execute(
        "SELECT p.product_name AS name, p.dosage AS dosage, p.dosage_freq AS frequency, "
        "p.dosage_timing AS timing, "
        "(SELECT text FROM fmp_snap_products_phases ph WHERE ph.id_fk_product=p.id_pk LIMIT 1) AS phase, "
        "(SELECT text FROM fmp_snap_products_systems sy WHERE sy.id_fk_product=p.id_pk LIMIT 1) AS system "
        "FROM fmp_snap_products p "
        "WHERE TRIM(COALESCE(p.product_name,''))<>'' AND p.product_name LIKE ? "
        "ORDER BY p.product_name LIMIT ?", (like, limit)).fetchall()
    return [{k: (r[k] or "") for k in ("name", "dosage", "frequency", "timing", "phase", "system")}
            for r in rows]


def remedy_dosing(cx, name):
    """Default dosing for a product name, to auto-fill a chain remedy."""
    blank = {"dosage": "", "frequency": "", "timing": ""}
    if not _has(cx, "fmp_snap_products"):
        return blank
    cx.row_factory = sqlite3.Row
    r = cx.execute(
        "SELECT dosage, dosage_freq AS frequency, dosage_timing AS timing "
        "FROM fmp_snap_products WHERE LOWER(TRIM(product_name))=LOWER(TRIM(?)) LIMIT 1",
        (name or "",)).fetchone()
    return {k: (r[k] or "") for k in blank} if r else blank


def stress_vocab(cx, q="", limit=20):
    """Distinct stress-factor terms Glen has actually used (autocomplete)."""
    if not _has(cx, "fmp_snap_client_active_main_stress"):
        return []
    like = f"%{(q or '').strip()}%"
    rows = cx.execute(
        "SELECT DISTINCT main_stress FROM fmp_snap_client_active_main_stress "
        "WHERE TRIM(COALESCE(main_stress,''))<>'' AND main_stress LIKE ? "
        "ORDER BY main_stress LIMIT ?", (like, limit)).fetchall()
    return [r[0] for r in rows]


def stress_suggestions(cx, stress, limit=8):
    """Remedies historically used for a given stress factor, most-used first."""
    if not (_has(cx, "fmp_snap_client_remedy") and _has(cx, "fmp_snap_client_causal_chain")
            and _has(cx, "fmp_snap_client_active_main_stress")):
        return []
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT r.remedy AS remedy, COUNT(*) AS n "
        "FROM fmp_snap_client_remedy r "
        "JOIN fmp_snap_client_causal_chain cc ON cc.id_pk=r.id_fk_causal_chain "
        "JOIN fmp_snap_client_active_main_stress ams ON ams.id_pk=cc.id_fk_active_stress "
        "WHERE LOWER(TRIM(ams.main_stress))=LOWER(TRIM(?)) AND TRIM(COALESCE(r.remedy,''))<>'' "
        "GROUP BY r.remedy ORDER BY n DESC, r.remedy LIMIT ?", (stress or "", limit)).fetchall()
    return [{"remedy": r["remedy"], "count": r["n"]} for r in rows]


def authored_report(cx, tid):
    init_auth_tables(cx)
    cx.row_factory = sqlite3.Row
    t = cx.execute("SELECT * FROM biofield_auth_tests WHERE id=?", (_num(tid),)).fetchone()
    rows = cx.execute("""
        SELECT id, layer, head, most_affected, remedy, dosage, frequency, timing
        FROM biofield_auth_chain
        WHERE test_id=? AND TRIM(COALESCE(remedy,''))<>''
        ORDER BY (layer IS NULL), layer, id""", (_num(tid),)).fetchall()
    layers = [{"layer": r["layer"], "head": r["head"] or "",
               "most_affected": r["most_affected"] or "", "remedy": r["remedy"] or "",
               "dosage": r["dosage"] or "", "frequency": r["frequency"] or "",
               "timing": r["timing"] or "", "rid": r["id"]} for r in rows]
    schedule = build_schedule([
        {"name": l["remedy"], "dosage": l["dosage"],
         "frequency": l["frequency"], "timing": l["timing"]} for l in layers])
    return {"test_id": str(tid),
            "client": {"name": (t["name"] if t else "") or "",
                       "email": (t["email"] if t else "") or ""},
            "date": (t["date_test"] if t else "") or "",
            "layers": layers, "schedule": schedule}
