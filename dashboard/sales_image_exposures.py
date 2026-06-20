import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_exposures ("
               "product_slug TEXT, session_id TEXT, created_at TEXT DEFAULT '', "
               "UNIQUE(product_slug, session_id))")
    cx.commit()

def record(cx, slug, session_id):
    session_id = (session_id or "").strip()
    if not session_id:
        return
    init_table(cx)
    cx.execute("INSERT INTO sales_image_exposures (product_slug, session_id, created_at) "
               "VALUES (?,?,?) ON CONFLICT(product_slug, session_id) DO NOTHING",
               (slug, session_id, _now()))
    cx.commit()

def per_product_counts(cx):
    init_table(cx)
    rows = cx.execute("SELECT product_slug, COUNT(*) FROM sales_image_exposures "
                      "GROUP BY product_slug").fetchall()
    return {r[0]: r[1] for r in rows}
