"""Business-OS Event/Audit log: one append-only timeline of business events and
operator/agent actions. Functions take a sqlite connection for testability."""
import json
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_event_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT,
            source TEXT,
            action_key TEXT,
            module TEXT,
            risk_tier TEXT,
            params_json TEXT,
            result_json TEXT,
            status TEXT NOT NULL,
            reversible INTEGER DEFAULT 0,
            ref_type TEXT,
            ref_id TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_events_module ON events(module)")
    cx.commit()


def _row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json") or "{}")
    rj = d.pop("result_json")
    d["result"] = json.loads(rj) if rj else None
    d["reversible"] = bool(d.get("reversible"))
    return d


def append_event(cx, *, actor, source, action_key, module, risk_tier,
                 params, result, status, reversible=False,
                 ref_type=None, ref_id=None):
    cur = cx.execute(
        """INSERT INTO events
           (ts, actor, source, action_key, module, risk_tier,
            params_json, result_json, status, reversible, ref_type, ref_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (_now(), actor, source, action_key, module, risk_tier,
         json.dumps(params or {}),
         json.dumps(result) if result is not None else None,
         status, 1 if reversible else 0, ref_type, ref_id))
    cx.commit()
    return cur.lastrowid


def get_event(cx, event_id):
    cur = cx.execute("SELECT * FROM events WHERE id=?", (event_id,))
    return _row_to_dict(cur.fetchone())


def list_events(cx, *, limit=50, status=None, module=None):
    q = "SELECT * FROM events"
    clauses, args = [], []
    if status:
        clauses.append("status=?"); args.append(status)
    if module:
        clauses.append("module=?"); args.append(module)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    cur = cx.execute(q, tuple(args))
    return [_row_to_dict(r) for r in cur.fetchall()]


def set_event_status(cx, event_id, status):
    cur = cx.execute("UPDATE events SET status=? WHERE id=?", (status, event_id))
    cx.commit()
    return cur.rowcount > 0
