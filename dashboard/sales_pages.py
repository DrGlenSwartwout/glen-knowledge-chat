import json
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS sales_pages ("
        "product_slug TEXT PRIMARY KEY, state TEXT DEFAULT 'draft', "
        "content_json TEXT DEFAULT '{}', model TEXT DEFAULT '', "
        "generated_at TEXT DEFAULT '', created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '')")
    cx.commit()


def get_page(cx, slug):
    init_table(cx)
    row = cx.execute(
        "SELECT product_slug, state, content_json, model, generated_at "
        "FROM sales_pages WHERE product_slug=?", (slug,)).fetchone()
    if not row:
        return None
    return {"product_slug": row[0], "state": row[1],
            "content": json.loads(row[2] or "{}"), "model": row[3], "generated_at": row[4]}


def get_section(cx, slug, section):
    page = get_page(cx, slug)
    if not page:
        return None
    return page["content"].get(section) or None


def upsert_section(cx, slug, section, text, model=""):
    init_table(cx)
    now = _now()
    row = cx.execute("SELECT content_json FROM sales_pages WHERE product_slug=?", (slug,)).fetchone()
    content = json.loads(row[0]) if row and row[0] else {}
    content[section] = text
    cx.execute(
        "INSERT INTO sales_pages (product_slug, state, content_json, model, generated_at, created_at, updated_at) "
        "VALUES (?, 'draft', ?, ?, ?, ?, ?) "
        "ON CONFLICT(product_slug) DO UPDATE SET content_json=excluded.content_json, "
        "model=excluded.model, generated_at=excluded.generated_at, updated_at=excluded.updated_at",
        (slug, json.dumps(content), model, now, now, now))
    cx.commit()
