"""Inventory ledger — persisted on-hand balance per ingredient (Phase 3c-1)."""
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


def init_inventory_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS inventory_txns (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ingredient_id INTEGER REFERENCES ingredients(id),
          txn_type TEXT NOT NULL,
          qty REAL NOT NULL,
          unit TEXT,
          txn_date TEXT,
          source_kind TEXT,
          source_ref TEXT,
          notes TEXT,
          created_at TEXT DEFAULT (datetime('now')),
          updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_invtxn_source ON inventory_txns(source_ref) WHERE source_ref IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_invtxn_ing ON inventory_txns(ingredient_id)")
    cx.commit()


def on_hand(ingredient_id, db_path=None) -> float:
    with _connect(db_path) as cx:
        r = cx.execute("SELECT COALESCE(SUM(qty),0) AS oh FROM inventory_txns WHERE ingredient_id=?",
                       (ingredient_id,)).fetchone()
    return float(r["oh"] or 0)


def inventory_levels(q="", limit=50, offset=0, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT i.id, i.name,
                   COALESCE((SELECT SUM(qty) FROM inventory_txns t WHERE t.ingredient_id=i.id),0) AS on_hand,
                   json_extract(i.extras,'$.par_level')      AS par_level,
                   json_extract(i.extras,'$.par_level_unit') AS par_level_unit
            FROM ingredients i
            WHERE i.name LIKE ?
            ORDER BY i.name
            LIMIT ? OFFSET ?
        """, (f"%{q}%", int(limit), int(offset))).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["on_hand"] = float(d["on_hand"] or 0)
        par = _to_num(d.get("par_level"))
        d["below_par"] = 1 if (par is not None and d["on_hand"] < par) else 0
        out.append(d)
    # below-par first, then name
    out.sort(key=lambda d: (-d["below_par"], (d["name"] or "").lower()))
    return out


def get_inventory(ingredient_id, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("SELECT * FROM ingredients WHERE id=?", (ingredient_id,)).fetchone()
        if not r:
            return None
        ing = dict(r)
        oh = cx.execute("SELECT COALESCE(SUM(qty),0) AS oh FROM inventory_txns WHERE ingredient_id=?",
                        (ingredient_id,)).fetchone()["oh"]
        txns = [dict(x) for x in cx.execute(
            "SELECT * FROM inventory_txns WHERE ingredient_id=? ORDER BY txn_date DESC, id DESC",
            (ingredient_id,)).fetchall()]
    par = _json_get(ing.get("extras"), "par_level")
    par_unit = _json_get(ing.get("extras"), "par_level_unit")
    return {"ingredient": ing, "on_hand": float(oh or 0),
            "par_level": par, "par_level_unit": par_unit, "txns": txns}


def list_txns(ingredient_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute(
            "SELECT * FROM inventory_txns WHERE ingredient_id=? ORDER BY txn_date DESC, id DESC",
            (ingredient_id,)).fetchall()
    return [dict(r) for r in rows]


def add_adjustment(ingredient_id, qty, unit=None, txn_date=None, notes=None, db_path=None) -> int:
    try:
        q = float(qty)
    except (TypeError, ValueError):
        raise ValueError("qty must be numeric")
    with _connect(db_path) as cx:
        if not cx.execute("SELECT 1 FROM ingredients WHERE id=?", (ingredient_id,)).fetchone():
            raise ValueError(f"no ingredient {ingredient_id}")
        cur = cx.execute("""
            INSERT INTO inventory_txns (ingredient_id, txn_type, qty, unit, txn_date, source_kind, notes)
            VALUES (?, 'adjustment', ?, ?, ?, 'manual', ?)
        """, (ingredient_id, q, unit, txn_date, notes))
        cx.commit()
        return int(cur.lastrowid)


_TXN_CURATED = {"notes"}


def update_txn_curated(txn_id, fields, db_path=None) -> None:
    cols = {k: v for k, v in (fields or {}).items() if k in _TXN_CURATED}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE inventory_txns SET {sets} WHERE id=?", (*cols.values(), txn_id))
        cx.commit()


def _to_num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _json_get(extras, key):
    import json
    if not extras:
        return None
    try:
        return json.loads(extras).get(key)
    except (ValueError, TypeError):
        return None


def seed_baselines(cx) -> int:
    cx.row_factory = sqlite3.Row
    rows = cx.execute("""
        SELECT id, json_extract(extras,'$.inventory_starting') AS start,
                   json_extract(extras,'$.par_level_unit')      AS unit
        FROM ingredients
        WHERE json_extract(extras,'$.inventory_starting') IS NOT NULL
    """).fetchall()
    n = 0
    for r in rows:
        qty = _to_num(r["start"])
        if qty is None:
            continue
        cur = cx.execute("""
            INSERT OR IGNORE INTO inventory_txns
                (ingredient_id, txn_type, qty, unit, txn_date, source_kind, source_ref)
            VALUES (?, 'baseline', ?, ?, NULL, 'fmp_baseline', ?)
        """, (r["id"], qty, r["unit"], f"baseline:{r['id']}"))
        n += cur.rowcount
    return n


def seed_receipts(cx) -> int:
    cx.row_factory = sqlite3.Row
    rows = cx.execute("""
        SELECT rec.id AS rec_id, pi.ingredient_id AS ingredient_id,
               rec.qty_received AS qty, rec.received_size AS unit,
               COALESCE(po.posted_date, po.po_date, date(rec.created_at)) AS txn_date
        FROM po_receiving rec
        JOIN po_items pi ON pi.id = rec.po_item_id
        LEFT JOIN purchase_orders po ON po.id = rec.po_id
        WHERE pi.ingredient_id IS NOT NULL
          AND rec.qty_received IS NOT NULL AND rec.qty_received <> 0
    """).fetchall()
    n = 0
    for r in rows:
        cur = cx.execute("""
            INSERT OR IGNORE INTO inventory_txns
                (ingredient_id, txn_type, qty, unit, txn_date, source_kind, source_ref)
            VALUES (?, 'receipt', ?, ?, ?, 'po_receiving', ?)
        """, (r["ingredient_id"], float(r["qty"]), r["unit"], r["txn_date"], f"po_receiving:{r['rec_id']}"))
        n += cur.rowcount
    return n
