import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS review_video_jobs ("
        "review_id INTEGER PRIMARY KEY, status TEXT DEFAULT 'pending', "
        "enqueued_at TEXT, done_at TEXT)")
    cx.commit()


def enqueue(cx, review_id):
    init_table(cx)
    cx.execute(
        "INSERT INTO review_video_jobs (review_id, status, enqueued_at) VALUES (?,'pending',?) "
        "ON CONFLICT(review_id) DO UPDATE SET status='pending', enqueued_at=excluded.enqueued_at, done_at=NULL",
        (review_id, _now()))
    cx.commit()


def claim_pending(cx, limit=3):
    init_table(cx)
    rows = cx.execute(
        "SELECT review_id FROM review_video_jobs WHERE status='pending' "
        "ORDER BY enqueued_at LIMIT ?", (limit,)).fetchall()
    return [r[0] for r in rows]


def mark(cx, review_id, status):
    init_table(cx)
    cx.execute("UPDATE review_video_jobs SET status=?, done_at=? WHERE review_id=?",
               (status, _now(), review_id))
    cx.commit()
