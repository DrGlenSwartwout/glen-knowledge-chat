import json
import os
import datetime

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "review-gifts.json")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_catalog(path=None):
    try:
        with open(path or _DEFAULT_PATH) as fh:
            data = json.load(fh)
        return [g for g in data if g.get("sku")]
    except Exception:
        return []


def catalog_by_sku(path=None):
    return {g["sku"]: g for g in load_catalog(path)}


def valid_sku(sku, path=None):
    return sku in catalog_by_sku(path)


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS review_gifts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, review_id INTEGER, email TEXT, "
        "gift_sku TEXT, gift_label TEXT, reason TEXT, status TEXT DEFAULT 'suggested', "
        "created_at TEXT, approved_by TEXT DEFAULT '', approved_at TEXT DEFAULT '', "
        "fulfilled_order_id INTEGER, fulfilled_at TEXT DEFAULT '')")
    cx.commit()


def _row(cx, where, args):
    cur = cx.cursor(); cur.row_factory = __import__("sqlite3").Row
    r = cur.execute(f"SELECT * FROM review_gifts WHERE {where}", args).fetchone()
    return dict(r) if r else None


def add_suggestion(cx, review_id, email, sku, label, reason):
    init_table(cx)
    cx.execute(
        "INSERT INTO review_gifts (review_id, email, gift_sku, gift_label, reason, status, created_at) "
        "VALUES (?,?,?,?,?, 'suggested', ?)",
        (review_id, (email or "").strip().lower(), sku, label, reason or "", _now()))
    cx.commit()
    return cx.execute("SELECT id FROM review_gifts WHERE review_id=? ORDER BY id DESC LIMIT 1",
                      (review_id,)).fetchone()[0]


def recent_active_gift(cx, email, days=30):
    init_table(cx)
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()
    return cx.execute(
        "SELECT 1 FROM review_gifts WHERE email=? AND status!='rejected' AND created_at>=? LIMIT 1",
        ((email or "").strip().lower(), cutoff)).fetchone() is not None


def get_for_review(cx, review_id):
    init_table(cx)
    return _row(cx, "review_id=? ORDER BY id DESC LIMIT 1", (review_id,))


def set_status(cx, gift_id, status, by=""):
    init_table(cx)
    if status == "approved":
        cx.execute("UPDATE review_gifts SET status=?, approved_by=?, approved_at=? WHERE id=?",
                   (status, by or "", _now(), gift_id))
    else:
        cx.execute("UPDATE review_gifts SET status=? WHERE id=?", (status, gift_id))
    cx.commit()


def swap_sku(cx, gift_id, sku, label):
    init_table(cx)
    cx.execute("UPDATE review_gifts SET gift_sku=?, gift_label=? WHERE id=?", (sku, label, gift_id))
    cx.commit()


def pending_for(cx, email):
    init_table(cx)
    cur = cx.cursor(); cur.row_factory = __import__("sqlite3").Row
    rows = cur.execute(
        "SELECT * FROM review_gifts WHERE email=? AND status='approved' AND fulfilled_order_id IS NULL "
        "ORDER BY id", ((email or "").strip().lower(),)).fetchall()
    return [dict(r) for r in rows]


def mark_fulfilled(cx, gift_id, order_id):
    init_table(cx)
    cx.execute("UPDATE review_gifts SET status='fulfilled', fulfilled_order_id=?, fulfilled_at=? WHERE id=?",
               (order_id, _now(), gift_id))
    cx.commit()


def suggested_queue(cx):
    init_table(cx)
    cur = cx.cursor(); cur.row_factory = __import__("sqlite3").Row
    rows = cur.execute(
        "SELECT * FROM review_gifts WHERE status='suggested' ORDER BY created_at DESC, id DESC").fetchall()
    return [dict(r) for r in rows]
