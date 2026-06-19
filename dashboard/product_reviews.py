import datetime


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
    cx.commit()


def _row(cx, where, args):
    cx.row_factory = __import__("sqlite3").Row
    r = cx.execute(f"SELECT * FROM product_reviews WHERE {where}", args).fetchone()
    return dict(r) if r else None


def has_reviewed(cx, slug, email):
    init_table(cx)
    e = (email or "").strip().lower()
    return cx.execute("SELECT 1 FROM product_reviews WHERE product_slug=? AND email=?",
                      (slug, e)).fetchone() is not None


def upsert_review(cx, slug, email, name, rating, body="", video_kind="", video_ref=""):
    init_table(cx)
    e = (email or "").strip().lower()
    now = _now()
    cx.execute(
        "INSERT INTO product_reviews (product_slug, email, name, rating, body, video_kind, "
        "video_ref, status, created_at) VALUES (?,?,?,?,?,?,?,'pending',?) "
        "ON CONFLICT(product_slug, email) DO UPDATE SET name=excluded.name, rating=excluded.rating, "
        "body=excluded.body, video_kind=excluded.video_kind, video_ref=excluded.video_ref, "
        "status='pending', ai_score=0, ai_verdict='', ai_recommend_publish=0, points_awarded=0, "
        "featured=0, reviewed_at='', reviewed_by=''",
        (slug, e, name or "", int(rating), body or "", video_kind or "", video_ref or "", now))
    cx.commit()
    return cx.execute("SELECT id FROM product_reviews WHERE product_slug=? AND email=?",
                      (slug, e)).fetchone()[0]


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
    cx.row_factory = __import__("sqlite3").Row
    rows = cx.execute(
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
    cx.row_factory = __import__("sqlite3").Row
    rows = cx.execute(
        "SELECT * FROM product_reviews WHERE status='pending' ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,)).fetchall()
    return [dict(r) for r in rows]
