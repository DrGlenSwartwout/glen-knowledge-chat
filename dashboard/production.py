"""Production runs + consumption posting (Phase 3c-2)."""
import os
import sqlite3
from pathlib import Path
from typing import Optional


def _default_db_path() -> str:
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    cx = sqlite3.connect(db_path or _default_db_path())
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA foreign_keys=ON")
    return cx


def init_production_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS production_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT,
          formulation_id INTEGER REFERENCES formulations(id),
          batch_number TEXT,
          run_date TEXT,
          quantity_units REAL,
          status TEXT,
          source_kind TEXT,
          extras TEXT,
          notes TEXT,
          created_at TEXT DEFAULT (datetime('now')),
          updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prodrun_fmp ON production_runs(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_prodrun_form ON production_runs(formulation_id)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS production_run_items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT,
          production_run_id INTEGER REFERENCES production_runs(id),
          item_type TEXT,
          ingredient_id INTEGER REFERENCES ingredients(id),
          material_id INTEGER REFERENCES materials(id),
          item_label TEXT,
          qty_used REAL,
          unit TEXT,
          extras TEXT,
          notes TEXT,
          created_at TEXT DEFAULT (datetime('now')),
          updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prunitems_fmp ON production_run_items(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_prunitems_run ON production_run_items(production_run_id)")
    cx.commit()


def search_production_runs(q="", limit=50, offset=0, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT pr.*, f.name AS formulation_name
            FROM production_runs pr LEFT JOIN formulations f ON f.id = pr.formulation_id
            WHERE pr.batch_number LIKE ? OR f.name LIKE ?
            ORDER BY pr.run_date DESC, pr.id DESC LIMIT ? OFFSET ?
        """, (f"%{q}%", f"%{q}%", int(limit), int(offset))).fetchall()
    return [dict(r) for r in rows]


def get_production_run(run_id, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("""
            SELECT pr.*, f.name AS formulation_name
            FROM production_runs pr LEFT JOIN formulations f ON f.id = pr.formulation_id
            WHERE pr.id=?
        """, (run_id,)).fetchone()
    return dict(r) if r else None


def list_run_items(run_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT it.*, ing.name AS ingredient_canonical, mat.name AS material_name,
              CASE WHEN EXISTS (SELECT 1 FROM inventory_txns t
                                WHERE t.source_ref = 'prod_item:' || it.id) THEN 1 ELSE 0 END AS posted
            FROM production_run_items it
            LEFT JOIN ingredients ing ON ing.id = it.ingredient_id
            LEFT JOIN materials mat ON mat.id = it.material_id
            WHERE it.production_run_id = ? ORDER BY it.id
        """, (run_id,)).fetchall()
    return [dict(r) for r in rows]


_RUN_CURATED = {"notes"}
_ITEM_CURATED = {"notes"}


def _update_allowed(table, row_id, fields, allowed, db_path):
    cols = {k: v for k, v in (fields or {}).items() if k in allowed}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*cols.values(), row_id))
        cx.commit()


def update_run_curated(run_id, fields, db_path=None):
    _update_allowed("production_runs", run_id, fields, _RUN_CURATED, db_path)


def update_run_item_curated(item_id, fields, db_path=None):
    _update_allowed("production_run_items", item_id, fields, _ITEM_CURATED, db_path)


def post_consumption(cx, run_id=None, mode="all", cutoff_date=None) -> int:
    """Post negative consumption ledger rows for run items. mode: all|from_date|record_only.
    Takes an OPEN connection; caller commits. Idempotent via source_ref='prod_item:<item.id>'."""
    cx.row_factory = sqlite3.Row
    if mode == "record_only":
        return 0
    where = ["it.item_type = 'ingredient'", "it.ingredient_id IS NOT NULL",
             "it.qty_used IS NOT NULL", "it.qty_used <> 0"]
    params = []
    if run_id is not None:
        where.append("it.production_run_id = ?")
        params.append(run_id)
    if mode == "from_date" and cutoff_date:
        where.append("pr.run_date >= ?")
        params.append(cutoff_date)
    sql = """
        SELECT it.id AS item_id, it.ingredient_id, it.qty_used, it.unit, pr.run_date
        FROM production_run_items it
        JOIN production_runs pr ON pr.id = it.production_run_id
        WHERE """ + " AND ".join(where)
    n = 0
    for r in cx.execute(sql, params).fetchall():
        cur = cx.execute("""
            INSERT OR IGNORE INTO inventory_txns
              (ingredient_id, txn_type, qty, unit, txn_date, source_kind, source_ref)
            VALUES (?, 'consumption', ?, ?, ?, 'production_run', ?)
        """, (r["ingredient_id"], -abs(float(r["qty_used"])), r["unit"], r["run_date"],
              f"prod_item:{r['item_id']}"))
        n += cur.rowcount
    return n


def log_run(formulation_id, run_date, quantity_units, items, batch_number=None, db_path=None) -> int:
    if not items:
        raise ValueError("a production run needs at least one item")
    with _connect(db_path) as cx:
        if not cx.execute("SELECT 1 FROM formulations WHERE id=?", (formulation_id,)).fetchone():
            raise ValueError(f"no formulation {formulation_id}")
        cur = cx.execute("""
            INSERT INTO production_runs (formulation_id, batch_number, run_date, quantity_units, status, source_kind)
            VALUES (?, ?, ?, ?, 'completed', 'manual')
        """, (formulation_id, batch_number, run_date, quantity_units))
        rid = int(cur.lastrowid)
        for it in items:
            ing_id = it.get("ingredient_id")
            label = None
            if ing_id is not None:
                row = cx.execute("SELECT name FROM ingredients WHERE id=?", (ing_id,)).fetchone()
                label = row["name"] if row else None
            cx.execute("""
                INSERT INTO production_run_items
                  (production_run_id, item_type, ingredient_id, item_label, qty_used, unit)
                VALUES (?, 'ingredient', ?, ?, ?, ?)
            """, (rid, ing_id, label, it.get("qty_used"), it.get("unit")))
        post_consumption(cx, run_id=rid, mode="all")
        cx.commit()
        return rid


def recipe_prefill(formulation_id, db_path=None):
    from dashboard.formulations import list_items_for_formulation
    out = []
    for fi in list_items_for_formulation(formulation_id, db_path=db_path):
        out.append({
            "ingredient_id": fi.get("ingredient_id"),
            "item_label": fi.get("ingredient_canonical") or fi.get("ingredient_name"),
            "qty_used": fi.get("dose"),
            "unit": fi.get("dose_unit"),
        })
    return out
