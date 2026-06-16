# dashboard/cert_submissions.py
"""Certification work-product submission store (pure: cx + args, no Flask).

Mirrors dashboard/cert_bonus.py conventions. JSON list columns (formats, modules,
credited_modules) are stored as JSON text and parsed on read.
"""
import json
from datetime import datetime, timezone

_JSON_COLS = ("formats", "modules", "credited_modules")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_tables(cx):
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS cert_submissions (
          id TEXT PRIMARY KEY,
          email TEXT NOT NULL,
          practitioner_id TEXT,
          title TEXT,
          description TEXT,
          url TEXT,
          file_path TEXT,
          formats TEXT,            -- JSON list of format keys
          format_other TEXT,
          modules TEXT,            -- JSON list of module ids (student-claimed)
          module_other TEXT,
          topic_angle TEXT,
          permission INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'submitted',
          credited_modules TEXT,   -- JSON list of module ids credited on approve
          review_note TEXT,
          case_study_id TEXT,
          created_at TEXT,
          updated_at TEXT
        )
        """
    )
    cx.execute("CREATE INDEX IF NOT EXISTS idx_cert_sub_email ON cert_submissions(email)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_cert_sub_status ON cert_submissions(status)")
    cx.commit()


def _row(r):
    if r is None:
        return None
    d = dict(r)
    for c in _JSON_COLS:
        try:
            d[c] = json.loads(d.get(c) or "[]")
        except Exception:
            d[c] = []
    return d


def create(cx, *, sid, email, practitioner_id, title, description, url,
           file_path, formats, format_other, modules, module_other,
           topic_angle, permission):
    now = _now()
    cx.execute(
        """
        INSERT INTO cert_submissions
          (id, email, practitioner_id, title, description, url, file_path,
           formats, format_other, modules, module_other, topic_angle,
           permission, status, credited_modules, review_note, case_study_id,
           created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, 'submitted', '[]', '', '', ?, ?)
        """,
        (sid, email, practitioner_id, title, description, url, file_path,
         json.dumps(list(formats or [])), format_other,
         json.dumps([int(m) for m in (modules or [])]), module_other,
         topic_angle, int(bool(permission)), now, now),
    )
    cx.commit()
    return sid


def get(cx, sid):
    r = cx.execute("SELECT * FROM cert_submissions WHERE id = ?", (sid,)).fetchone()
    return _row(r)


def list_for_email(cx, email):
    rows = cx.execute(
        "SELECT * FROM cert_submissions WHERE lower(email) = lower(?) "
        "ORDER BY created_at DESC", (email,)
    ).fetchall()
    return [_row(r) for r in rows]


def list_by_status(cx, status=None):
    if status:
        rows = cx.execute(
            "SELECT * FROM cert_submissions WHERE status = ? ORDER BY created_at DESC",
            (status,)
        ).fetchall()
    else:
        rows = cx.execute(
            "SELECT * FROM cert_submissions ORDER BY created_at DESC"
        ).fetchall()
    return [_row(r) for r in rows]


def set_status(cx, sid, status, *, credited_modules=None, review_note=None,
               case_study_id=None):
    sets = ["status = ?", "updated_at = ?"]
    vals = [status, _now()]
    if credited_modules is not None:
        sets.append("credited_modules = ?")
        vals.append(json.dumps([int(m) for m in credited_modules]))
    if review_note is not None:
        sets.append("review_note = ?")
        vals.append(review_note)
    if case_study_id is not None:
        sets.append("case_study_id = ?")
        vals.append(case_study_id)
    vals.append(sid)
    cx.execute(f"UPDATE cert_submissions SET {', '.join(sets)} WHERE id = ?", vals)
    cx.commit()
