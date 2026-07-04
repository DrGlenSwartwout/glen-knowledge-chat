import json, sqlite3
from datetime import datetime, timezone

def _now(): return datetime.now(timezone.utc).isoformat()

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS practitioner_recommendations ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, practitioner_id TEXT NOT NULL, "
               "patient_email TEXT NOT NULL, items_json TEXT NOT NULL DEFAULT '[]', "
               "note TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'sent', created_at TEXT NOT NULL)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_pr_patient ON practitioner_recommendations (lower(patient_email), status)")
    cx.commit()

def create(cx, *, practitioner_id, patient_email, items, note):
    init_table(cx)
    cur = cx.execute("INSERT INTO practitioner_recommendations "
        "(practitioner_id, patient_email, items_json, note, status, created_at) VALUES (?,?,?,?,'sent',?)",
        (str(practitioner_id), (patient_email or "").strip().lower(), json.dumps(items or []), note or "", _now()))
    cx.commit(); return cur.lastrowid

def active_for_patient(cx, patient_email):
    init_table(cx)
    r = cx.execute("SELECT * FROM practitioner_recommendations WHERE lower(patient_email)=lower(?) "
        "AND status!='dismissed' ORDER BY id DESC LIMIT 1", ((patient_email or "").strip(),)).fetchone()
    if not r: return None
    d = dict(r); d["items"] = json.loads(d.pop("items_json") or "[]"); return d

def set_status(cx, rec_id, status):
    init_table(cx)
    cx.execute("UPDATE practitioner_recommendations SET status=? WHERE id=?", (status, int(rec_id))); cx.commit()
