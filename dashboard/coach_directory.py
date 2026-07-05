"""Coach volunteer directory (coaching arc, slice 1). Pure sqlite; no app-layer
imports. Members see only {name, focus, intro_video_url} — a coach's email is
never exposed. Only active AND cert_ok volunteers are listed."""

_DDL = """
CREATE TABLE IF NOT EXISTS coach_volunteers (
    email TEXT PRIMARY KEY,
    name TEXT,
    focus TEXT,
    intro_video_url TEXT,
    capacity INTEGER DEFAULT 3,
    active INTEGER DEFAULT 1,
    cert_ok INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_coach_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def upsert_volunteer(cx, *, email, name, focus, intro_video_url, capacity, cert_ok):
    email = _lc(email)
    now = _now()
    cx.execute(
        "INSERT INTO coach_volunteers (email,name,focus,intro_video_url,capacity,"
        "active,cert_ok,created_at,updated_at) VALUES (?,?,?,?,?,1,?,?,?) "
        "ON CONFLICT(email) DO UPDATE SET name=excluded.name, focus=excluded.focus, "
        "intro_video_url=excluded.intro_video_url, capacity=excluded.capacity, "
        "cert_ok=excluded.cert_ok, updated_at=excluded.updated_at",
        (email, name, focus, intro_video_url, int(capacity), int(cert_ok), now, now))
    cx.commit()


def set_active(cx, email, active):
    cx.execute("UPDATE coach_volunteers SET active=?, updated_at=? WHERE email=?",
               (1 if active else 0, _now(), _lc(email)))
    cx.commit()


def get_volunteer(cx, email):
    row = cx.execute("SELECT * FROM coach_volunteers WHERE email=?", (_lc(email),)).fetchone()
    return dict(row) if row else None


def list_active(cx):
    rows = cx.execute("SELECT name, focus, intro_video_url FROM coach_volunteers "
                      "WHERE active=1 AND cert_ok=1 ORDER BY updated_at DESC").fetchall()
    return [{"name": r["name"], "focus": r["focus"], "intro_video_url": r["intro_video_url"]}
            for r in rows]


def list_active_full(cx):
    """Server-side: active+cert_ok volunteers WITH email + capacity (for ref +
    capacity composition in the route). Never send this shape to a member."""
    rows = cx.execute("SELECT email, name, focus, intro_video_url, capacity "
                      "FROM coach_volunteers WHERE active=1 AND cert_ok=1 "
                      "ORDER BY updated_at DESC").fetchall()
    return [{"email": r["email"], "name": r["name"], "focus": r["focus"],
             "intro_video_url": r["intro_video_url"], "capacity": r["capacity"]}
            for r in rows]
