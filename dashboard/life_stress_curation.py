"""Per-client practitioner CURATION of Life Stress essences (an override/prescription).
When present, it replaces the auto-pool everywhere the client sees it. Pure sqlite
(LOG_DB), no Flask. Mirrors dashboard/practitioner_programs.py. Never raises on bad data."""
import json
import datetime


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS life_stress_curations (
        patient_email TEXT PRIMARY KEY,
        practitioner_id TEXT,
        slugs_json TEXT NOT NULL DEFAULT '[]',
        note TEXT,
        updated_at TEXT
    )""")


def get(cx, email):
    """{"slugs":[...],"note","updated_at"} or None (no row, empty slugs, or bad JSON)."""
    e = (email or "").strip().lower()
    if not e:
        return None
    try:
        init_table(cx)
        row = cx.execute("SELECT slugs_json, note, updated_at FROM life_stress_curations "
                         "WHERE patient_email=?", (e,)).fetchone()
        if not row:
            return None
        slugs = json.loads(row[0])
        slugs = [str(s) for s in slugs] if isinstance(slugs, list) else []
        if not slugs:
            return None
        return {"slugs": slugs, "note": row[1] or "", "updated_at": row[2]}
    except (ValueError, TypeError):
        return None


def set(cx, email, practitioner_id, slugs, note):
    e = (email or "").strip().lower()
    if not e:
        return
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = json.dumps([str(s) for s in (slugs or [])])
    init_table(cx)
    cx.execute("""INSERT INTO life_stress_curations(patient_email, practitioner_id, slugs_json, note, updated_at)
                  VALUES(?,?,?,?,?)
                  ON CONFLICT(patient_email) DO UPDATE SET
                    practitioner_id=excluded.practitioner_id, slugs_json=excluded.slugs_json,
                    note=excluded.note, updated_at=excluded.updated_at""",
               (e, str(practitioner_id or ""), payload, str(note or ""), now))
    cx.commit()


def clear(cx, email):
    e = (email or "").strip().lower()
    if not e:
        return
    init_table(cx)
    cx.execute("DELETE FROM life_stress_curations WHERE patient_email=?", (e,))
    cx.commit()
