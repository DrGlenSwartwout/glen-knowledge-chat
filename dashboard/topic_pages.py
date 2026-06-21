"""Store and request/notify for public topic pages (symptom/condition/function).

Mirrors dashboard/ingredient_pages.py. One table, kind-discriminated. Public path
serves approved-only. Adds links_json, compliance_json, seo_json columns.
"""
import datetime
import json
import sqlite3


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


VALID_KINDS = ("symptom", "condition", "function")


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS topic_pages ("
        "slug TEXT PRIMARY KEY, "
        "kind TEXT DEFAULT '', "
        "name TEXT DEFAULT '', "
        "state TEXT DEFAULT 'draft', "
        "content_json TEXT DEFAULT '{}', "
        "links_json TEXT DEFAULT '{}', "
        "compliance_json TEXT DEFAULT '{}', "
        "seo_json TEXT DEFAULT '{}', "
        "model TEXT DEFAULT '', "
        "generated_at TEXT DEFAULT '', "
        "approved_at TEXT DEFAULT '', "
        "approved_by TEXT DEFAULT '', "
        "created_at TEXT DEFAULT '', "
        "updated_at TEXT DEFAULT '')"
    )
    cx.execute(
        "CREATE TABLE IF NOT EXISTS topic_page_requests ("
        "slug TEXT, "
        "email TEXT, "
        "requested_at TEXT, "
        "emailed_at TEXT DEFAULT '', "
        "PRIMARY KEY(slug, email))"
    )
    cx.commit()


def get_page(cx, slug):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    row = cur.execute(
        "SELECT slug, kind, name, state, content_json, links_json, compliance_json, "
        "seo_json, model, generated_at, approved_at, approved_by, created_at, updated_at "
        "FROM topic_pages WHERE slug=?", (slug,)
    ).fetchone()
    if not row:
        return None
    return {
        "slug": row["slug"],
        "kind": row["kind"] or "",
        "name": row["name"] or "",
        "state": row["state"] or "draft",
        "content": json.loads(row["content_json"] or "{}"),
        "links": json.loads(row["links_json"] or "{}"),
        "compliance": json.loads(row["compliance_json"] or "{}"),
        "seo": json.loads(row["seo_json"] or "{}"),
        "model": row["model"] or "",
        "generated_at": row["generated_at"] or "",
        "approved_at": row["approved_at"] or "",
        "approved_by": row["approved_by"] or "",
        "created_at": row["created_at"] or "",
        "updated_at": row["updated_at"] or "",
    }


def get_section(cx, slug, section):
    page = get_page(cx, slug)
    if not page:
        return None
    return page["content"].get(section) or None


def _upsert_col(cx, slug, col, value):
    """Generic single-column upsert that also touches updated_at/created_at."""
    init_table(cx)
    now = _now()
    cx.execute(
        f"INSERT INTO topic_pages (slug, {col}, created_at, updated_at) "
        f"VALUES (?, ?, ?, ?) "
        f"ON CONFLICT(slug) DO UPDATE SET {col}=excluded.{col}, updated_at=excluded.updated_at",
        (slug, value, now, now),
    )
    cx.commit()


def upsert_section(cx, slug, section, text, model=""):
    init_table(cx)
    now = _now()
    row = cx.execute("SELECT content_json FROM topic_pages WHERE slug=?", (slug,)).fetchone()
    content = json.loads(row[0]) if row and row[0] else {}
    content[section] = text
    cx.execute(
        "INSERT INTO topic_pages (slug, state, content_json, model, generated_at, created_at, updated_at) "
        "VALUES (?, 'draft', ?, ?, ?, ?, ?) "
        "ON CONFLICT(slug) DO UPDATE SET "
        "content_json=excluded.content_json, model=excluded.model, "
        "generated_at=excluded.generated_at, updated_at=excluded.updated_at",
        (slug, json.dumps(content), model, now, now, now),
    )
    cx.commit()


def set_state(cx, slug, state, by=""):
    init_table(cx)
    now = _now()
    if state == "approved":
        cx.execute(
            "UPDATE topic_pages SET state=?, approved_at=?, approved_by=?, updated_at=? WHERE slug=?",
            (state, now, by, now, slug),
        )
    else:
        cx.execute(
            "UPDATE topic_pages SET state=?, updated_at=? WHERE slug=?", (state, now, slug)
        )
    cx.commit()


def set_name(cx, slug, name):
    _upsert_col(cx, slug, "name", name or "")


def set_kind(cx, slug, kind):
    _upsert_col(cx, slug, "kind", kind or "")


def set_links(cx, slug, links):
    _upsert_col(cx, slug, "links_json", json.dumps(links or {}))


def set_compliance(cx, slug, result):
    _upsert_col(cx, slug, "compliance_json", json.dumps(result or {}))


def set_seo(cx, slug, seo):
    _upsert_col(cx, slug, "seo_json", json.dumps(seo or {}))


def list_pages(cx):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT slug, kind, name, state, content_json, compliance_json "
        "FROM topic_pages ORDER BY updated_at DESC"
    ).fetchall()
    out = []
    for r in rows:
        content = json.loads(r["content_json"] or "{}")
        comp = json.loads(r["compliance_json"] or "{}")
        out.append({
            "slug": r["slug"], "kind": r["kind"] or "", "name": r["name"] or r["slug"],
            "state": r["state"] or "draft", "sections": sorted(content.keys()),
            "compliance_passed": comp.get("passed"),
        })
    return out


# ---------------------------------------------------------------------------
# Request / notify (mirrors ingredient_pages)
# ---------------------------------------------------------------------------

def record_request(cx, slug, email):
    init_table(cx)
    e = _norm(email)
    if not e:
        return
    cx.execute(
        "INSERT OR IGNORE INTO topic_page_requests (slug, email, requested_at, emailed_at) "
        "VALUES (?, ?, ?, '')",
        (slug, e, _now()),
    )
    cx.commit()


def requesters_to_email(cx, slug):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT email, requested_at FROM topic_page_requests "
        "WHERE slug=? AND COALESCE(emailed_at,'')='' ORDER BY requested_at",
        (slug,),
    ).fetchall()
    return [{"email": r["email"], "requested_at": r["requested_at"]} for r in rows]


def mark_emailed(cx, slug, email):
    init_table(cx)
    cx.execute(
        "UPDATE topic_page_requests SET emailed_at=? WHERE slug=? AND email=?",
        (_now(), slug, _norm(email)),
    )
    cx.commit()


def notify_on_approve(cx, slug, name, base_url, *, send, strip=None):
    """Email each un-emailed requester once; mark each after send; never raises."""
    if strip is None:
        strip = lambda s: s  # noqa: E731
    requesters = requesters_to_email(cx, slug)
    link = f"{base_url}/learn/{slug}"
    subject = f"Your {name} guide is ready"
    for r in requesters:
        email = r["email"]
        body = strip(
            f"Aloha,\n\nThe guide you asked about, {name}, is ready:\n\n{link}\n\n"
            f"In wellness,\nDr. Glen & Rae"
        )
        try:
            send(email, subject, body)
            mark_emailed(cx, slug, email)
        except Exception as exc:  # noqa: BLE001 - one bad send must not stop the rest
            print(f"[topic-pages] send failed for {email}: {exc}", flush=True)
