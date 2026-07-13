"""Per-patient practitioner-composed condition program. One active row per
patient. Pure sqlite, no Flask. Distinct from practitioner_recommendations
(the one-shot AI nudge) so it never collides with that card and keeps dose/alts."""
import json
from datetime import datetime, timezone

def _now(): return datetime.now(timezone.utc).isoformat()

def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS practitioner_programs (
        patient_email TEXT PRIMARY KEY,
        practitioner_id TEXT NOT NULL,
        condition_key TEXT,
        items_json TEXT NOT NULL DEFAULT '[]',
        note TEXT DEFAULT '',
        updated_at TEXT)""")
    cx.commit()

def upsert(cx, *, patient_email, practitioner_id, condition_key, items, note):
    init_table(cx)
    cx.execute("""INSERT INTO practitioner_programs
        (patient_email, practitioner_id, condition_key, items_json, note, updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(patient_email) DO UPDATE SET
          practitioner_id=excluded.practitioner_id, condition_key=excluded.condition_key,
          items_json=excluded.items_json, note=excluded.note, updated_at=excluded.updated_at""",
        ((patient_email or "").strip().lower(), str(practitioner_id), condition_key or "",
         json.dumps(items or []), note or "", _now()))
    cx.commit()

def get(cx, patient_email):
    init_table(cx)
    r = cx.execute("SELECT * FROM practitioner_programs WHERE patient_email=?",
                   ((patient_email or "").strip().lower(),)).fetchone()
    if not r: return None
    return {"patient_email": r["patient_email"], "practitioner_id": r["practitioner_id"],
            "condition_key": r["condition_key"], "items": json.loads(r["items_json"] or "[]"),
            "note": r["note"], "updated_at": r["updated_at"]}
