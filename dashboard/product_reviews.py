import datetime
import sqlite3


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS product_reviews ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, product_slug TEXT, email TEXT, name TEXT, "
        "rating INTEGER NOT NULL, body TEXT DEFAULT '', video_kind TEXT DEFAULT '', "
        "video_ref TEXT DEFAULT '', ai_score INTEGER DEFAULT 0, ai_verdict TEXT DEFAULT '', "
        "ai_recommend_publish INTEGER DEFAULT 0, points_awarded INTEGER DEFAULT 0, "
        "status TEXT DEFAULT 'pending', featured INTEGER DEFAULT 0, created_at TEXT, "
        "reviewed_at TEXT, reviewed_by TEXT, UNIQUE(product_slug, email))")
    for _col in ("video_points INTEGER DEFAULT 0", "transcript TEXT DEFAULT ''",
                 "video_status TEXT DEFAULT ''", "publish_risk INTEGER DEFAULT 0",
                 "video_verdict TEXT DEFAULT ''", "video_orig_ref TEXT DEFAULT ''",
                 "kind TEXT DEFAULT 'product'", "practitioner_id INTEGER DEFAULT 0",
                 "consent_public INTEGER DEFAULT 0", "source_tag TEXT DEFAULT ''",
                 "compliance_score INTEGER DEFAULT 0", "publication_score INTEGER DEFAULT 0",
                 "authenticity_score INTEGER DEFAULT 0", "specificity_score INTEGER DEFAULT 0"):
        try:
            cx.execute(f"ALTER TABLE product_reviews ADD COLUMN {_col}")
        except sqlite3.OperationalError:
            pass
    cx.commit()


def _row(cx, where, args):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    r = cur.execute(f"SELECT * FROM product_reviews WHERE {where}", args).fetchone()
    return dict(r) if r else None


def has_reviewed(cx, slug, email):
    init_table(cx)
    e = (email or "").strip().lower()
    return cx.execute("SELECT 1 FROM product_reviews WHERE product_slug=? AND email=?",
                      (slug, e)).fetchone() is not None


def upsert_review(cx, slug, email, name, rating, body="", video_kind="", video_ref="",
                  *, kind="product", practitioner_id=0, consent_public=0, source_tag=""):
    init_table(cx)
    e = (email or "").strip().lower()
    now = _now()
    cx.execute(
        "INSERT INTO product_reviews (product_slug, email, name, rating, body, video_kind, "
        "video_ref, kind, practitioner_id, consent_public, source_tag, status, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending',?) "
        "ON CONFLICT(product_slug, email) DO UPDATE SET name=excluded.name, rating=excluded.rating, "
        "body=excluded.body, video_kind=excluded.video_kind, video_ref=excluded.video_ref, "
        "kind=excluded.kind, practitioner_id=excluded.practitioner_id, "
        "consent_public=excluded.consent_public, source_tag=excluded.source_tag, "
        "status='pending', ai_score=0, ai_verdict='', ai_recommend_publish=0, points_awarded=0, "
        "featured=0, reviewed_at='', reviewed_by=''",
        (slug, e, name or "", int(rating), body or "", video_kind or "", video_ref or "",
         kind or "product", int(practitioner_id or 0), 1 if consent_public else 0,
         (source_tag or "")[:64], now))
    cx.commit()
    return cx.execute("SELECT id FROM product_reviews WHERE product_slug=? AND email=?",
                      (slug, e)).fetchone()[0]


def set_scores(cx, review_id, *, compliance=0, publication=0, authenticity=0, specificity=0):
    """Store the four 1-10 AI rating dimensions (10 = best; 0 = unscored) on a review."""
    init_table(cx)
    _clamp = lambda v: max(0, min(10, int(v or 0)))
    cx.execute(
        "UPDATE product_reviews SET compliance_score=?, publication_score=?, "
        "authenticity_score=?, specificity_score=? WHERE id=?",
        (_clamp(compliance), _clamp(publication), _clamp(authenticity), _clamp(specificity), review_id))
    cx.commit()


def set_ai_result(cx, review_id, ai_score, ai_verdict, recommend_publish):
    init_table(cx)
    cx.execute("UPDATE product_reviews SET ai_score=?, ai_verdict=?, ai_recommend_publish=? WHERE id=?",
               (int(ai_score), ai_verdict or "", 1 if recommend_publish else 0, review_id))
    cx.commit()


def set_points(cx, review_id, points):
    init_table(cx)
    cx.execute("UPDATE product_reviews SET points_awarded=? WHERE id=?", (int(points), review_id))
    cx.commit()


def set_status(cx, review_id, status, by=""):
    init_table(cx)
    cx.execute("UPDATE product_reviews SET status=?, reviewed_at=?, reviewed_by=? WHERE id=?",
               (status, _now(), by or "", review_id))
    cx.commit()


def set_featured(cx, review_id, on):
    init_table(cx)
    cx.execute("UPDATE product_reviews SET featured=? WHERE id=?", (1 if on else 0, review_id))
    cx.commit()


def get_review(cx, review_id):
    init_table(cx)
    return _row(cx, "id=?", (review_id,))


def approved_for_slug(cx, slug):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT * FROM product_reviews WHERE product_slug=? AND status='approved' "
        "ORDER BY featured DESC, created_at DESC, id DESC", (slug,)).fetchall()
    return [dict(r) for r in rows]


def aggregate(cx, slug):
    init_table(cx)
    row = cx.execute(
        "SELECT COUNT(*), AVG(rating) FROM product_reviews WHERE product_slug=? AND status='approved'",
        (slug,)).fetchone()
    n = row[0] or 0
    return {"count": n, "avg": round(row[1], 1) if n else 0.0}


def pending_queue(cx, limit=100):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT * FROM product_reviews WHERE status='pending' ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,)).fetchall()
    return [dict(r) for r in rows]


def set_video_result(cx, review_id, video_points, transcript, status, publish_risk=0, video_verdict=""):
    init_table(cx)
    cx.execute(
        "UPDATE product_reviews SET video_points=?, transcript=?, video_status=?, "
        "publish_risk=?, video_verdict=? WHERE id=?",
        (int(video_points), transcript or "", status, 1 if publish_risk else 0,
         video_verdict or "", review_id))
    cx.commit()


def has_successful_video(cx, email):
    init_table(cx)
    e = (email or "").strip().lower()
    return cx.execute(
        "SELECT 1 FROM product_reviews WHERE email=? AND video_points>0 AND status='approved' LIMIT 1",
        (e,)).fetchone() is not None


def set_trimmed(cx, review_id, trimmed_ref):
    init_table(cx)
    cur = cx.cursor(); cur.row_factory = sqlite3.Row
    row = cur.execute("SELECT video_ref, video_orig_ref FROM product_reviews WHERE id=?",
                      (review_id,)).fetchone()
    if not row:
        return
    orig = row["video_orig_ref"] or row["video_ref"] or ""
    cx.execute("UPDATE product_reviews SET video_ref=?, video_orig_ref=? WHERE id=?",
               (trimmed_ref, orig, review_id))
    cx.commit()
