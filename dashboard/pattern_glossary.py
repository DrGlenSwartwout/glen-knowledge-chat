"""Read-only reader over e4l.db for the public stress-pattern glossary. Pure; never
writes; never raises into callers (open failure -> None / empty)."""
import sqlite3

from dashboard.ingredients import slugify as _slugify
from dashboard import biofield_e4l as _be

STYPE_LABELS = {
    "organ": "Organs", "system": "Body systems", "function": "Functions",
    "emotion": "Emotions", "substance": "Substances", "immune_cell": "Immune cells",
    "other": "Other",
}

# Fixed display order for category sections in the index; unknown categories append.
_CATEGORY_ORDER = ["ED", "EI", "ET", "ES", "ER", "MB", "MR", "BFA",
                   "Nutrition", "Environmental", "Sensitivity", "LifeJourney"]


def slug_for(code):
    return _slugify(code or "")


def open_ro(db_path=None):
    try:
        cx = _be._connect_ro(_be._db_path(db_path))
        cx.row_factory = sqlite3.Row
        return cx
    except Exception:
        return None


def _slug_map(cx):
    out = {}
    for r in cx.execute("SELECT code FROM e4l_items"):
        out.setdefault(slug_for(r["code"]), r["code"])
    return out


def _code_for(cx, slug):
    return _slug_map(cx).get((slug or "").strip().lower())


def _structures(cx, code):
    rows = cx.execute(
        "SELECT structure, stype, is_primary FROM e4l_pattern_structures WHERE code=? "
        "ORDER BY is_primary DESC, stype, structure", (code,)).fetchall()
    return [{"structure": r["structure"], "stype": (r["stype"] or "other"),
             "is_primary": r["is_primary"] or 0} for r in rows]


def get_pattern(cx, slug):
    code = _code_for(cx, slug)
    if not code:
        return None
    r = cx.execute(
        "SELECT code, category, subcategory, name, full_name, e4l_description "
        "FROM e4l_items WHERE code=?", (code,)).fetchone()
    if not r:
        return None
    structures = _structures(cx, code)
    desc = (r["e4l_description"] or "").strip()
    return {
        "code": r["code"], "name": (r["name"] or r["code"]).strip(),
        "full_name": (r["full_name"] or "").strip(),
        "category": r["category"] or "", "subcategory": (r["subcategory"] or "").strip(),
        "description": desc, "structures": structures,
        "has_page": bool(desc or structures),
    }


def page_exists(cx, slug):
    code = _code_for(cx, slug)
    if not code:
        return False
    r = cx.execute("SELECT TRIM(COALESCE(e4l_description,'')) d FROM e4l_items WHERE code=?",
                   (code,)).fetchone()
    if r and r["d"]:
        return True
    n = cx.execute("SELECT COUNT(*) n FROM e4l_pattern_structures WHERE code=?", (code,)).fetchone()
    return bool(n and n["n"])


def list_patterns(cx):
    rows = cx.execute(
        "SELECT i.code, i.category, i.name, i.full_name, i.sort_order, "
        "  TRIM(COALESCE(i.e4l_description,'')) AS d, "
        "  (SELECT COUNT(*) FROM e4l_pattern_structures s WHERE s.code=i.code) AS n "
        "FROM e4l_items i ORDER BY i.category, COALESCE(i.sort_order, 9999), i.name").fetchall()
    by_cat = {}
    for r in rows:
        if not (r["d"] or r["n"]):
            continue  # no page -> exclude
        by_cat.setdefault(r["category"] or "", []).append({
            "slug": slug_for(r["code"]), "name": (r["name"] or r["code"]).strip(),
            "full_name": (r["full_name"] or "").strip(),
            "has_desc": bool(r["d"]), "n_structures": r["n"] or 0,
        })
    ordered = [c for c in _CATEGORY_ORDER if c in by_cat]
    ordered += [c for c in by_cat if c not in _CATEGORY_ORDER]
    return [{"category": c, "patterns": by_cat[c]} for c in ordered]


def pattern_remedies(cx, code):
    """The formulations mapped to an E4L pattern code ("what may help"), ordered
    by priority then name, deduped by name (best/lowest priority wins). Each item
    is {name, priority}. Empty list on any failure or unknown code."""
    code = (code or "").strip()
    if not code:
        return []
    try:
        rows = cx.execute(
            "SELECT f.name AS name, m.priority AS priority "
            "FROM e4l_formulation_map m JOIN formulations f ON f.id = m.formulation_id "
            "WHERE m.item_code = ? ORDER BY COALESCE(m.priority, 5), f.name", (code,)).fetchall()
    except Exception:
        return []
    out, seen = [], set()
    for r in rows:
        nm = (r["name"] or "").strip()
        key = nm.lower()
        if not nm or key in seen:
            continue
        seen.add(key)
        out.append({"name": nm, "priority": r["priority"] if r["priority"] is not None else 5})
    return out
