import json
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _ensure_columns(cx):
    cols = {r[1] for r in cx.execute("PRAGMA table_info(sales_pages)").fetchall()}
    if "approved_at" not in cols:
        cx.execute("ALTER TABLE sales_pages ADD COLUMN approved_at TEXT DEFAULT ''")
    if "approved_by" not in cols:
        cx.execute("ALTER TABLE sales_pages ADD COLUMN approved_by TEXT DEFAULT ''")


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS sales_pages ("
        "product_slug TEXT PRIMARY KEY, state TEXT DEFAULT 'draft', "
        "content_json TEXT DEFAULT '{}', model TEXT DEFAULT '', "
        "generated_at TEXT DEFAULT '', created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '')")
    _ensure_columns(cx)
    cx.commit()


def set_state(cx, slug, state, by=""):
    init_table(cx)
    now = _now()
    if state == "approved":
        cx.execute(
            "UPDATE sales_pages SET state=?, approved_at=?, approved_by=?, updated_at=? "
            "WHERE product_slug=?", (state, now, by, now, slug))
    else:
        cx.execute("UPDATE sales_pages SET state=?, updated_at=? WHERE product_slug=?",
                   (state, now, slug))
    cx.commit()


def list_draft_pages(cx):
    init_table(cx)
    rows = cx.execute(
        "SELECT product_slug, state, content_json FROM sales_pages "
        "ORDER BY updated_at DESC").fetchall()
    out = []
    for slug, state, cj in rows:
        content = json.loads(cj or "{}")
        if not content:
            continue
        out.append({"slug": slug, "state": state or "draft",
                    "sections": sorted(content.keys())})
    return out


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
