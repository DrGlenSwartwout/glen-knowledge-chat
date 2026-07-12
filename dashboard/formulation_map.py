"""CRUD for the code->remedy formulation map (e4l.db: e4l_formulation_map JOIN
formulations), backing the curation tool. Curated mappings here drive BOTH the reveal
synthesis default (priority-1 per code) and the reveal alternatives (the rest, in
priority order). Pure sqlite; the caller passes the e4l.db connection. Semantic
proposals (Pinecone) live in a separate module so this stays unit-testable.
"""
import sqlite3


def init_tables(cx):
    """Ensure the two tables exist (real e4l.db already has them; this lets tests and a
    fresh db work). Mirrors the columns the ingest + synthesis rely on."""
    cx.execute("""CREATE TABLE IF NOT EXISTS formulations(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, sku TEXT, category TEXT,
        description TEXT, key_ingredients TEXT, url TEXT, notes TEXT,
        min_age INTEGER, max_age INTEGER)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS e4l_formulation_map(
        id INTEGER PRIMARY KEY AUTOINCREMENT, item_code TEXT, finding_pattern TEXT,
        score_min REAL, formulation_id INTEGER, priority INTEGER,
        other_therapies TEXT, notes TEXT, source TEXT)""")
    cx.commit()


def _formulation_id(cx, name):
    """The formulations.id for `name` (case-insensitive), creating the row if absent.
    Returns None for a blank name."""
    name = (name or "").strip()
    if not name:
        return None
    row = cx.execute("SELECT id FROM formulations WHERE lower(name)=lower(?)", (name,)).fetchone()
    if row:
        return row[0]
    cur = cx.execute("INSERT INTO formulations(name) VALUES(?)", (name,))
    cx.commit()
    return cur.lastrowid


def mappings_for(cx, code):
    """Current mappings for one code, priority order: [{formulation_id, name, priority}]."""
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT m.formulation_id, f.name, m.priority "
        "FROM e4l_formulation_map m JOIN formulations f ON f.id=m.formulation_id "
        "WHERE m.item_code=? ORDER BY m.priority ASC, m.id ASC", (code,)).fetchall()
    return [{"formulation_id": r["formulation_id"], "name": r["name"], "priority": r["priority"]}
            for r in rows]


def add_mapping(cx, code, remedy_name):
    """Append (code -> remedy) at the next priority (bottom). Idempotent: a code that
    already maps to that formulation is returned unchanged (no duplicate, no reorder).
    Returns the refreshed mappings for the code."""
    code = (code or "").strip()
    fid = _formulation_id(cx, remedy_name)
    if not code or fid is None:
        return mappings_for(cx, code)
    dup = cx.execute("SELECT 1 FROM e4l_formulation_map WHERE item_code=? AND formulation_id=?",
                     (code, fid)).fetchone()
    if dup:
        return mappings_for(cx, code)
    nxt = (cx.execute("SELECT COALESCE(MAX(priority),0) FROM e4l_formulation_map WHERE item_code=?",
                      (code,)).fetchone()[0] or 0) + 1
    cx.execute("INSERT INTO e4l_formulation_map(item_code,finding_pattern,formulation_id,priority,source) "
               "VALUES(?,?,?,?,?)", (code, code, fid, nxt, "curation"))
    cx.commit()
    return mappings_for(cx, code)


def remove_mapping(cx, code, formulation_id):
    """Remove one (code -> formulation) mapping; leaves the remaining priorities as-is
    (they stay strictly increasing, just possibly with a gap — reorder normalizes)."""
    cx.execute("DELETE FROM e4l_formulation_map WHERE item_code=? AND formulation_id=?",
               ((code or "").strip(), int(formulation_id)))
    cx.commit()
    return mappings_for(cx, code)


def reorder(cx, code, ordered_formulation_ids):
    """Set priority 1..N for the code from the given formulation_id order (drag/drop)."""
    code = (code or "").strip()
    for i, fid in enumerate(ordered_formulation_ids or [], start=1):
        cx.execute("UPDATE e4l_formulation_map SET priority=? WHERE item_code=? AND formulation_id=?",
                   (i, code, int(fid)))
    cx.commit()
    return mappings_for(cx, code)
