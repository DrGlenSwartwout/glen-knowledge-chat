"""Supplier-quote review queue (email-sourcing collector)."""
import os
import sqlite3
from pathlib import Path
from typing import Optional

from dashboard.ingredient_catalog import create_source


def _default_db_path() -> str:
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    cx = sqlite3.connect(db_path or _default_db_path())
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA foreign_keys=ON")
    return cx


def init_sourcing_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS supplier_quotes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          gmail_msg_id TEXT, received_at TEXT, from_email TEXT, subject TEXT, raw_snippet TEXT,
          supplier_name TEXT, ingredient_name TEXT,
          price REAL, price_unit TEXT, currency TEXT,
          moq REAL, moq_unit TEXT, lead_time_days INTEGER, confidence REAL,
          supplier_id INTEGER REFERENCES suppliers(id),
          ingredient_id INTEGER REFERENCES ingredients(id),
          status TEXT DEFAULT 'pending', applied_source_id INTEGER REFERENCES ingredient_sources(id),
          has_attachments INTEGER DEFAULT 0, extras TEXT, notes TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_quotes_msg ON supplier_quotes(gmail_msg_id) WHERE gmail_msg_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_quotes_status ON supplier_quotes(status)")
    cx.commit()


_STAGE_COLS = ["gmail_msg_id", "received_at", "from_email", "subject", "raw_snippet",
               "supplier_name", "ingredient_name", "price", "price_unit", "currency",
               "moq", "moq_unit", "lead_time_days", "confidence", "has_attachments"]


def stage_quotes(cx, rows) -> int:
    cx.row_factory = sqlite3.Row
    n = 0
    for r in rows or []:
        vals = [r.get(c) for c in _STAGE_COLS]
        cur = cx.execute(
            f"INSERT OR IGNORE INTO supplier_quotes ({','.join(_STAGE_COLS)}) "
            f"VALUES ({','.join('?' for _ in _STAGE_COLS)})", vals)
        n += cur.rowcount
    return n


def list_quotes(status=None, limit=200, db_path=None):
    with _connect(db_path) as cx:
        if status:
            rows = cx.execute("SELECT * FROM supplier_quotes WHERE status=? ORDER BY id DESC LIMIT ?",
                              (status, int(limit))).fetchall()
        else:
            rows = cx.execute("SELECT * FROM supplier_quotes ORDER BY (status='pending') DESC, id DESC LIMIT ?",
                              (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def get_quote(qid, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("SELECT * FROM supplier_quotes WHERE id=?", (qid,)).fetchone()
    return dict(r) if r else None


_MATCH_EDITABLE = {"ingredient_id", "supplier_id", "supplier_name", "ingredient_name",
                   "price", "price_unit", "currency", "moq", "moq_unit", "lead_time_days", "notes"}


def update_quote_match(qid, fields, db_path=None) -> None:
    cols = {k: v for k, v in (fields or {}).items() if k in _MATCH_EDITABLE}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE supplier_quotes SET {sets} WHERE id=?", (*cols.values(), qid))
        cx.commit()


def match_quote(cx, qid) -> None:
    """Best-effort fuzzy match: set ingredient_id/supplier_id from the extracted names (exact-ish, single hit).

    NOTE: table and column names in the f-string are HARDCODED literals — never user-controlled.
    All values are passed via parameterised ? placeholders.
    """
    cx.row_factory = sqlite3.Row
    q = cx.execute("SELECT * FROM supplier_quotes WHERE id=?", (qid,)).fetchone()
    if not q:
        return

    def _one(table, col, name):
        if not name:
            return None
        hits = cx.execute(f"SELECT id FROM {table} WHERE lower({col}) = lower(?) LIMIT 2", (name.strip(),)).fetchall()
        if len(hits) == 1:
            return hits[0]["id"]
        like = cx.execute(f"SELECT id FROM {table} WHERE {col} LIKE ? LIMIT 2", (f"%{name.strip()}%",)).fetchall()
        return like[0]["id"] if len(like) == 1 else None

    iid = _one("ingredients", "name", q["ingredient_name"])
    sid = _one("suppliers", "company", q["supplier_name"])
    cx.execute(
        "UPDATE supplier_quotes SET ingredient_id=COALESCE(?,ingredient_id), supplier_id=COALESCE(?,supplier_id), updated_at=datetime('now') WHERE id=?",
        (iid, sid, qid))


def approve_quote(qid, db_path=None) -> int:
    q = get_quote(qid, db_path=db_path)
    if not q:
        raise ValueError(f"no quote {qid}")
    if q["status"] != "pending":
        raise ValueError(f"quote {qid} is already {q['status']}")
    if not q["ingredient_id"]:
        raise ValueError("match an ingredient before approving")
    src_fields = {
        "supplier_id": q["supplier_id"], "supplier_name": q["supplier_name"],
        "price_per_unit": q["price"], "unit_size": 1, "unit_type": q["price_unit"],
        "minimum_order": q["moq"], "minimum_order_unit": q["moq_unit"],
        "lead_time_days": q["lead_time_days"],
    }
    src_fields = {k: v for k, v in src_fields.items() if v is not None}
    sid = create_source(q["ingredient_id"], src_fields, db_path=db_path)
    with _connect(db_path) as cx:
        cx.execute("UPDATE supplier_quotes SET status='applied', applied_source_id=?, updated_at=datetime('now') WHERE id=?",
                   (sid, qid))
        cx.commit()
    return sid


def dismiss_quote(qid, db_path=None) -> None:
    with _connect(db_path) as cx:
        row = cx.execute("SELECT status FROM supplier_quotes WHERE id=?", (qid,)).fetchone()
        if not row:
            raise ValueError(f"no quote {qid}")
        if row["status"] == "applied":
            raise ValueError(f"quote {qid} is already applied (it created a source); cannot dismiss")
        cx.execute("UPDATE supplier_quotes SET status='dismissed', updated_at=datetime('now') WHERE id=?", (qid,))
        cx.commit()
