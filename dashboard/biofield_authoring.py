"""Increment 4a: native authoring store for the local Biofield Analysis tool.

Lets Glen author a biofield test in the app instead of FileMaker. A test is a
header + causal-chain rows entered directly (streamlined vs FMP's stress->promote
flow). `authored_report` returns the SAME shape as `biofield_report.causal_chain_report`
so the schedule, narrative, and your-voice audio all work on authored tests unchanged.

Authored test ids are prefixed "a" (e.g. "a7") so the viewer can tell them apart
from the numeric FMP-snapshot ids. Local + writable; PHI stays on the Mac.
"""
import datetime
import difflib
import re
import sqlite3

from dashboard.biofield_schedule import build_schedule
from dashboard.biofield_dimensions import DEPTH_KEY, depth_label, depth_match, get_tag


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _num(tid):
    return int(str(tid).lstrip("a") or 0)


def _clean_product_name(name):
    """FMP product names carry a trailing '*' as Glen's internal 'intending to
    discontinue' marker — the product is still active and sellable. Drop it so the
    picker name matches the sellable catalog and the stress-coverage map, which
    both store the clean name. Mirrors scripts/fmp_catalog_import.clean_name."""
    return (name or "").strip().rstrip("*").strip()


def _is_discontinue_intent(name):
    """True when a product carries the trailing-'*' discontinue-intent marker."""
    return (name or "").strip().endswith("*")


def init_auth_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_tests(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT,
        date_test TEXT, created_at TEXT, updated_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS biofield_auth_chain(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, layer INTEGER,
        head TEXT, most_affected TEXT, remedy TEXT, dosage TEXT, frequency TEXT,
        timing TEXT, sort_seq INTEGER, created_at TEXT, confirmed INTEGER DEFAULT 1,
        origin TEXT NOT NULL DEFAULT 'live')""")
    try:
        cx.execute("ALTER TABLE biofield_auth_chain ADD COLUMN confirmed INTEGER DEFAULT 1")
    except Exception:
        pass
    try:
        cx.execute("ALTER TABLE biofield_auth_chain ADD COLUMN origin TEXT NOT NULL DEFAULT 'live'")
    except Exception:
        pass
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
                  dosage="", frequency="", timing="", confirmed=1, origin="live"):
    init_auth_tables(cx)
    cur = cx.execute(
        "INSERT INTO biofield_auth_chain(test_id,layer,head,most_affected,remedy,"
        "dosage,frequency,timing,sort_seq,created_at,confirmed,origin) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (_num(tid), layer, (head or "").strip(), (most_affected or "").strip(),
         (remedy or "").strip(), dosage or "", frequency or "", timing or "", 0, _now(),
         1 if confirmed else 0, (origin or "live")))
    cx.commit()
    return cur.lastrowid


def confirm_row(cx, rid):
    cx.execute("UPDATE biofield_auth_chain SET confirmed=1 WHERE id=?", (rid,))
    cx.commit()


def confirm_all(cx, tid):
    cx.execute("UPDATE biofield_auth_chain SET confirmed=1 WHERE test_id=?", (_num(tid),))
    cx.commit()


def delete_test(cx, tid):
    init_auth_tables(cx)
    cx.execute("DELETE FROM biofield_auth_chain WHERE test_id=?", (_num(tid),))
    cx.execute("DELETE FROM biofield_auth_tests WHERE id=?", (_num(tid),))
    for t in ("biofield_notes", "biofield_narratives", "biofield_video_scripts"):
        try:
            cx.execute(f"DELETE FROM {t} WHERE test_id=?", (str(tid),))
        except Exception:
            pass
    cx.commit()


_SMALL_WORDS = {"in", "of", "the", "a", "an", "and", "or", "with", "for", "to", "by", "on"}


def _title_case_name(s):
    """Title-case a free-text name without mangling product codes or small words.
    'reverse age' -> 'Reverse Age', 'head and tail' -> 'Head and Tail', and a token
    carrying a digit or an internal capital (e.g. 'MB5', 'B12', 'pH') is left as-is."""
    s = (s or "").strip()
    if not s:
        return s
    words = s.split()
    out = []
    for i, w in enumerate(words):
        if any(c.isdigit() for c in w) or any(c.isupper() for c in w[1:]):
            out.append(w)                      # preserve codes / intentional casing
        elif i > 0 and w.lower() in _SMALL_WORDS:
            out.append(w.lower())              # keep small connector words lowercase
        else:                                  # capitalize each hyphen/slash chunk
            out.append(re.sub(r"[A-Za-z]+",
                              lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), w))
    return " ".join(out)


def _best_match(spoken, names, cutoff):
    """Case-insensitive closest match: ASR lowercases, so we compare on lowercase and
    map back to the canonical-cased name. Returns None when nothing is close enough."""
    by_low = {}
    for n in names:
        by_low.setdefault(n.lower(), n)        # first canonical spelling wins
    hit = difflib.get_close_matches((spoken or "").lower(), list(by_low), n=1, cutoff=cutoff)
    return by_low[hit[0]] if hit else None


def _token_match(spoken, names, cutoff):
    """Match a distinctive spoken token (e.g. 'Sobopla') to the catalog product whose
    name CONTAINS it as a word — for cases where the clinician says only the unique
    part of a long product name. Returns the canonical name only when exactly ONE
    product qualifies, so a common shared word ('Essence') stays ambiguous and is
    left to the Title-Case fallback. Single distinctive token only."""
    sp = (spoken or "").strip().lower()
    if len(sp) < 5 or " " in sp:               # too short / multi-word -> not a distinctive token
        return None
    hits = set()
    for n in names:
        for t in re.findall(r"[A-Za-z0-9]+", n.lower()):
            if t == sp or (len(t) >= 5 and difflib.SequenceMatcher(None, sp, t).ratio() >= cutoff):
                hits.add(n)
                break
    return next(iter(hits)) if len(hits) == 1 else None


def resolve_remedy_name(cx, spoken, cutoff=0.82):
    """Best-effort auto-correct a (possibly ASR-mangled) remedy name to the closest
    catalog product (case-insensitive). Preserves an ' in Terrain Restore' suffix.
    Falls back to Title Case of the spoken name when there's no close catalog match."""
    spoken = (spoken or "").strip()
    if not spoken:
        return spoken
    suffix = ""
    core = spoken
    low = spoken.lower()
    if low.endswith("in terrain restore"):
        core = spoken[: low.rfind("in terrain restore")].strip()
        suffix = " in Terrain Restore"
    if _has(cx, "fmp_snap_products"):
        names = [r[0] for r in cx.execute(
            "SELECT DISTINCT product_name FROM fmp_snap_products "
            "WHERE TRIM(COALESCE(product_name,''))<>''").fetchall()]
        # whole-string fuzzy first, then a distinctive-token match for long names.
        match = _best_match(core, names, cutoff) or _token_match(core, names, cutoff)
        if match:
            match = _clean_product_name(match)   # drop discontinue-intent '*'
            # Don't double the suffix when the matched name already carries it.
            if suffix and match.lower().endswith("in terrain restore"):
                return match
            return match + suffix
    return _title_case_name(core) + suffix


def resolve_stress_name(cx, spoken, cutoff=0.82):
    """Auto-correct a spoken stress / head-of-chain name to the closest stress term
    Glen has used before (case-insensitive), else Title Case the spoken name so stress
    names are always capitalized."""
    spoken = (spoken or "").strip()
    if not spoken:
        return spoken
    if _has_col(cx, "fmp_snap_client_active_main_stress", "main_stress"):
        names = [r[0] for r in cx.execute(
            "SELECT DISTINCT main_stress FROM fmp_snap_client_active_main_stress "
            "WHERE TRIM(COALESCE(main_stress,''))<>''").fetchall()]
        match = _best_match(spoken, names, cutoff)
        if match:
            return match
    return _title_case_name(spoken)


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


def _has_col(cx, table, col):
    """True only when `table` exists AND has column `col` (snapshot schemas vary)."""
    if not _has(cx, table):
        return False
    return any(r[1] == col for r in cx.execute(f"PRAGMA table_info({table})").fetchall())


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
    out = []
    for r in rows:
        d = {k: (r[k] or "") for k in ("name", "dosage", "frequency", "timing", "phase", "system")}
        # Surface the discontinue-intent marker as a flag, but hand the UI (and,
        # once picked, the chain + invoice + coverage) the clean name.
        d["discontinue_intent"] = _is_discontinue_intent(d["name"])
        d["name"] = _clean_product_name(d["name"])
        out.append(d)
    return out


def remedy_dosing(cx, name):
    """Default dosing for a product name, to auto-fill a chain remedy."""
    blank = {"dosage": "", "frequency": "", "timing": ""}
    if not _has(cx, "fmp_snap_products"):
        return blank
    cx.row_factory = sqlite3.Row
    r = cx.execute(
        "SELECT dosage, dosage_freq AS frequency, dosage_timing AS timing "
        "FROM fmp_snap_products "
        "WHERE LOWER(TRIM(RTRIM(product_name,'* ')))=LOWER(TRIM(?)) LIMIT 1",
        (_clean_product_name(name),)).fetchone()
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


def ordered_chain(cx, tid):
    """Remedy-bearing chain rows in display order with two-zone numbering.
    Top zone = live + confirmed rows (manual order); bottom zone = unbalanced
    scan rows (origin='scan' AND confirmed=0), trailing. Display `layer` = 1..k."""
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT id, layer, head, most_affected, remedy, dosage, frequency, timing, "
        "confirmed, origin FROM biofield_auth_chain "
        "WHERE test_id=? AND TRIM(COALESCE(remedy,''))<>''", (_num(tid),)).fetchall()

    def unbalanced_scan(r):
        return (r["origin"] == "scan") and (r["confirmed"] == 0)

    key = lambda r: (r["layer"] is None, r["layer"] if r["layer"] is not None else 0, r["id"])
    top = sorted([r for r in rows if not unbalanced_scan(r)], key=key)
    bottom = sorted([r for r in rows if unbalanced_scan(r)], key=key)
    out = []
    for i, r in enumerate(top + bottom, 1):
        out.append({"id": r["id"], "layer": i, "head": r["head"] or "",
                    "most_affected": r["most_affected"] or "", "remedy": r["remedy"] or "",
                    "dosage": r["dosage"] or "", "frequency": r["frequency"] or "",
                    "timing": r["timing"] or "",
                    "confirmed": 0 if r["confirmed"] == 0 else 1,
                    "origin": r["origin"] or "live",
                    "zone": "bottom" if unbalanced_scan(r) else "top"})
    return out


def reorder_chain(cx, tid, rid, new_layer):
    """Move top-zone row `rid` to position `new_layer` and renumber the top zone
    contiguously. Unbalanced scan rows (bottom zone) are left untouched."""
    top = [l for l in ordered_chain(cx, tid) if l["zone"] == "top"]
    ids = [l["id"] for l in top]
    if rid not in ids:
        return
    ids.remove(rid)
    pos = max(1, min(int(new_layer or 1), len(ids) + 1)) - 1
    ids.insert(pos, rid)
    for i, _id in enumerate(ids, 1):
        cx.execute("UPDATE biofield_auth_chain SET layer=? WHERE id=?", (i, _id))
    cx.commit()


def set_layer_order(cx, tid, groups):
    """groups = ordered list of layer groups, each a list of chain-row ids. Assign
    stored layer = group position (1-based) to every row in that group so ordered_chain
    presents the groups (and their remedies) in this order.

    Deliberately placing a card also confirms its rows (confirmed=1) so an unbalanced
    scan row honours its new position instead of snapping back to the bottom zone --
    arranging a card counts as reviewing it. Rows already confirmed are unaffected."""
    for i, rids in enumerate(groups or [], 1):
        for rid in rids or []:
            try:
                rid = int(rid)
            except (TypeError, ValueError):
                continue
            cx.execute("UPDATE biofield_auth_chain SET layer=?, confirmed=1 "
                       "WHERE id=? AND test_id=?", (i, rid, _num(tid)))
    cx.commit()


def authored_report(cx, tid):
    init_auth_tables(cx)
    cx.row_factory = sqlite3.Row
    t = cx.execute("SELECT * FROM biofield_auth_tests WHERE id=?", (_num(tid),)).fetchone()
    layers = [{**l, "rid": l["id"]} for l in ordered_chain(cx, tid)]
    # Depth-of-penetration tags + reach match-check per layer (Increment 4b)
    for l in layers:
        sd = get_tag(cx, "auth_stress", l["rid"], DEPTH_KEY)
        rd = get_tag(cx, "auth_remedy", l["rid"], DEPTH_KEY)
        l["stress_depth"] = sd
        l["remedy_depth"] = rd
        l["depth_status"] = depth_match(sd, rd)
        l["depth_need"] = depth_label(cx, sd)
    schedule = build_schedule([
        {"name": l["remedy"], "dosage": l["dosage"],
         "frequency": l["frequency"], "timing": l["timing"]} for l in layers])
    return {"test_id": str(tid),
            "client": {"name": (t["name"] if t else "") or "",
                       "email": (t["email"] if t else "") or ""},
            "date": (t["date_test"] if t else "") or "",
            "layers": layers, "schedule": schedule}
