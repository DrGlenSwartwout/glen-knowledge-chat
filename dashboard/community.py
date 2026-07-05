"""Community content library store (slice 1, Layer A).

Pure sqlite. Full items (coaching_replay / course_session) are tier='paid' and
carry a Rumble embed video_ref; out-takes are tier='free', type='outtake', and
point at a full parent via parent_id. The membership gate (_is_paid_member) lives
in the route layer, so this module has no app-layer imports."""

import json

_DDL = """
CREATE TABLE IF NOT EXISTS community_content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    title TEXT,
    description TEXT,
    video_ref TEXT,
    tier TEXT NOT NULL,
    interest_tags TEXT,
    parent_id INTEGER,
    transcript TEXT,
    published INTEGER DEFAULT 0,
    published_at TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_community_parent ON community_content(parent_id);
CREATE INDEX IF NOT EXISTS ix_community_videoref ON community_content(video_ref);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def init_community_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def create_content(cx, *, type, title, description, video_ref, tier,
                   interest_tags, parent_id=None, transcript=None):
    cur = cx.execute(
        "INSERT INTO community_content (type,title,description,video_ref,tier,"
        "interest_tags,parent_id,transcript,published,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,0,?)",
        (type, title, description, video_ref, tier,
         json.dumps(list(interest_tags or [])), parent_id, transcript, _now()))
    cx.commit()
    return cur.lastrowid


def get_content(cx, content_id):
    row = cx.execute("SELECT * FROM community_content WHERE id=?", (content_id,)).fetchone()
    return dict(row) if row else None


def publish(cx, content_id):
    cx.execute("UPDATE community_content SET published=1, published_at=? WHERE id=?",
               (_now(), content_id))
    cx.commit()


def upsert_full(cx, *, type, title, description, video_ref, interest_tags, transcript):
    """Create-or-update a full (paid) item keyed on video_ref. On update, clear the
    item's existing out-takes so a re-publish replaces rather than duplicates."""
    tags = json.dumps(list(interest_tags or []))
    existing = cx.execute("SELECT id FROM community_content WHERE video_ref=? AND type!='outtake'",
                          (video_ref,)).fetchone()
    if existing:
        cid = existing[0]
        cx.execute("UPDATE community_content SET type=?, title=?, description=?, "
                   "tier='paid', interest_tags=?, transcript=? WHERE id=?",
                   (type, title, description, tags, transcript, cid))
        cx.execute("DELETE FROM community_content WHERE parent_id=?", (cid,))
        cx.commit()
        return cid
    cur = cx.execute(
        "INSERT INTO community_content (type,title,description,video_ref,tier,"
        "interest_tags,transcript,published,created_at) "
        "VALUES (?,?,?,?, 'paid', ?,?,0,?)",
        (type, title, description, video_ref, tags, transcript, _now()))
    cx.commit()
    return cur.lastrowid


def add_outtake(cx, *, parent_id, title, video_ref, interest_tags):
    cur = cx.execute(
        "INSERT INTO community_content (type,title,description,video_ref,tier,"
        "interest_tags,parent_id,published,created_at) "
        "VALUES ('outtake', ?, '', ?, 'free', ?, ?, 0, ?)",
        (title, video_ref, json.dumps(list(interest_tags or [])), parent_id, _now()))
    cx.commit()
    return cur.lastrowid


def _row_tags(row):
    try:
        return json.loads(row["interest_tags"] or "[]")
    except Exception:
        return []


def list_outtakes(cx, parent_id=None):
    if parent_id is None:
        rows = cx.execute("SELECT * FROM community_content WHERE type='outtake' "
                          "AND published=1 ORDER BY published_at DESC").fetchall()
    else:
        rows = cx.execute("SELECT * FROM community_content WHERE type='outtake' "
                          "AND published=1 AND parent_id=? ORDER BY published_at DESC",
                          (parent_id,)).fetchall()
    return [{"id": r["id"], "title": r["title"], "video_ref": r["video_ref"],
             "parent_id": r["parent_id"], "interest_tags": _row_tags(r)} for r in rows]


def list_full(cx):
    rows = cx.execute("SELECT * FROM community_content WHERE type!='outtake' "
                      "AND tier='paid' AND published=1 "
                      "ORDER BY published_at DESC").fetchall()
    out = []
    for r in rows:
        out.append({"id": r["id"], "type": r["type"], "title": r["title"],
                    "description": r["description"], "video_ref": r["video_ref"],
                    "interest_tags": _row_tags(r), "published_at": r["published_at"],
                    "outtakes": list_outtakes(cx, parent_id=r["id"])})
    return out


_FEED_DDL = """
CREATE TABLE IF NOT EXISTS community_embeddings (
    content_id INTEGER PRIMARY KEY,
    vec TEXT,
    model TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS member_interest (
    email TEXT PRIMARY KEY,
    vec TEXT,
    model TEXT,
    built_at TEXT
);
"""


def init_feed_tables(cx):
    cx.executescript(_FEED_DDL)
    cx.commit()


def set_embedding(cx, content_id, vec, model):
    cx.execute(
        "INSERT INTO community_embeddings (content_id,vec,model,updated_at) VALUES (?,?,?,?) "
        "ON CONFLICT(content_id) DO UPDATE SET vec=excluded.vec, model=excluded.model, "
        "updated_at=excluded.updated_at",
        (content_id, json.dumps(list(vec)), model, _now()))
    cx.commit()


def get_embeddings(cx, content_ids, model):
    if not content_ids:
        return {}
    qs = ",".join("?" * len(content_ids))
    rows = cx.execute(
        f"SELECT content_id, vec FROM community_embeddings "
        f"WHERE model=? AND content_id IN ({qs})",
        [model, *content_ids]).fetchall()
    return {r["content_id"]: json.loads(r["vec"]) for r in rows}


def set_member_interest(cx, email, vec, model):
    cx.execute(
        "INSERT INTO member_interest (email,vec,model,built_at) VALUES (?,?,?,?) "
        "ON CONFLICT(email) DO UPDATE SET vec=excluded.vec, model=excluded.model, "
        "built_at=excluded.built_at",
        ((email or "").strip().lower(), json.dumps(list(vec)), model, _now()))
    cx.commit()


def get_member_interest(cx, email, model):
    row = cx.execute("SELECT vec, built_at FROM member_interest WHERE email=? AND model=?",
                     ((email or "").strip().lower(), model)).fetchone()
    if not row:
        return None
    return {"vec": json.loads(row["vec"]), "built_at": row["built_at"]}
