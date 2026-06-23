"""Per-member longitudinal scan analysis, pushed by the local engine, stored for
the member-facing analysis page + chat (sub-project 2)."""
import json, sqlite3


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS scan_analyses (
        email TEXT PRIMARY KEY, analysis_json TEXT NOT NULL DEFAULT '{}',
        scan_count INT, date_range TEXT, generated_at TEXT,
        updated_at TEXT DEFAULT (datetime('now')))""")
    cx.commit()


def upsert(cx, email, artifact):
    e = (email or "").strip().lower()
    cx.execute("""INSERT INTO scan_analyses(email,analysis_json,scan_count,date_range,generated_at,updated_at)
        VALUES(?,?,?,?,?,datetime('now')) ON CONFLICT(email) DO UPDATE SET
        analysis_json=excluded.analysis_json, scan_count=excluded.scan_count,
        date_range=excluded.date_range, generated_at=excluded.generated_at, updated_at=datetime('now')""",
        (e, json.dumps(artifact), artifact.get("scan_count"),
         json.dumps(artifact.get("date_range")), artifact.get("generated_at")))
    cx.commit()


def get(cx, email):
    r = cx.execute("SELECT analysis_json, scan_count FROM scan_analyses WHERE email=lower(?)",
                   ((email or "").strip().lower(),)).fetchone()
    if not r:
        return None
    return {"analysis": json.loads(r[0] or "{}"), "scan_count": r[1]}
