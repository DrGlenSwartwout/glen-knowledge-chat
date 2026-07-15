"""Store and request/notify for mentor pages.

Mirrors dashboard/ingredient_pages.py (single-table page store, per-cursor Row
factory, request/notify helpers). A mentor page is a public, server-rendered
biography/lineage page served at /mentors/<slug>. Approved-only is public.

Mentor-specific columns beyond the ingredient shape: field (short descriptor),
lifespan (e.g. "1889-1973"), vital_status ("deceased" | "living"), lineage_json
(the intellectual chain), sources_json (dossier/reference labels), seo_json.
"""
import datetime
import json
import sqlite3


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS mentor_pages ("
        "mentor_slug TEXT PRIMARY KEY, "
        "name TEXT DEFAULT '', "
        "state TEXT DEFAULT 'draft', "
        "content_json TEXT DEFAULT '{}', "
        "field TEXT DEFAULT '', "
        "lifespan TEXT DEFAULT '', "
        "vital_status TEXT DEFAULT '', "
        "lineage_json TEXT DEFAULT '[]', "
        "sources_json TEXT DEFAULT '[]', "
        "seo_json TEXT DEFAULT '{}', "
        "model TEXT DEFAULT '', "
        "generated_at TEXT DEFAULT '', "
        "approved_at TEXT DEFAULT '', "
        "approved_by TEXT DEFAULT '', "
        "created_at TEXT DEFAULT '', "
        "updated_at TEXT DEFAULT '')"
    )
    cx.execute(
        "CREATE TABLE IF NOT EXISTS mentor_page_requests ("
        "mentor_slug TEXT, "
        "email TEXT, "
        "requested_at TEXT, "
        "emailed_at TEXT DEFAULT '', "
        "PRIMARY KEY(mentor_slug, email))"
    )
    cx.commit()


def get_page(cx, slug):
    """Return a dict with parsed content/lineage/sources/seo + state, or None."""
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    row = cur.execute(
        "SELECT mentor_slug, name, state, content_json, field, lifespan, vital_status, "
        "lineage_json, sources_json, seo_json, model, generated_at, approved_at, "
        "approved_by, created_at, updated_at "
        "FROM mentor_pages WHERE mentor_slug=?", (slug,)
    ).fetchone()
    if not row:
        return None
    return {
        "mentor_slug": row["mentor_slug"],
        "slug": row["mentor_slug"],
        "name": row["name"] or "",
        "state": row["state"] or "draft",
        "content": json.loads(row["content_json"] or "{}"),
        "field": row["field"] or "",
        "lifespan": row["lifespan"] or "",
        "vital_status": row["vital_status"] or "",
        "lineage": json.loads(row["lineage_json"] or "[]"),
        "sources": json.loads(row["sources_json"] or "[]"),
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
    """Generic single-column upsert that also stamps timestamps."""
    init_table(cx)
    now = _now()
    cx.execute(
        f"INSERT INTO mentor_pages (mentor_slug, {col}, created_at, updated_at) "
        "VALUES (?, ?, ?, ?) "
        f"ON CONFLICT(mentor_slug) DO UPDATE SET {col}=excluded.{col}, "
        "updated_at=excluded.updated_at",
        (slug, value, now, now),
    )
    cx.commit()


def upsert_section(cx, slug, section, text, model=""):
    init_table(cx)
    now = _now()
    row = cx.execute(
        "SELECT content_json FROM mentor_pages WHERE mentor_slug=?", (slug,)
    ).fetchone()
    content = json.loads(row[0]) if row and row[0] else {}
    content[section] = text
    cx.execute(
        "INSERT INTO mentor_pages "
        "(mentor_slug, state, content_json, model, generated_at, created_at, updated_at) "
        "VALUES (?, 'draft', ?, ?, ?, ?, ?) "
        "ON CONFLICT(mentor_slug) DO UPDATE SET "
        "content_json=excluded.content_json, "
        "model=excluded.model, "
        "generated_at=excluded.generated_at, "
        "updated_at=excluded.updated_at",
        (slug, json.dumps(content), model, now, now, now),
    )
    cx.commit()


def set_state(cx, slug, state, by=""):
    init_table(cx)
    now = _now()
    if state == "approved":
        cx.execute(
            "UPDATE mentor_pages SET state=?, approved_at=?, approved_by=?, updated_at=? "
            "WHERE mentor_slug=?", (state, now, by, now, slug)
        )
    else:
        cx.execute(
            "UPDATE mentor_pages SET state=?, updated_at=? WHERE mentor_slug=?",
            (state, now, slug),
        )
    cx.commit()


def set_name(cx, slug, name):
    _upsert_col(cx, slug, "name", name or "")


def set_field(cx, slug, field):
    _upsert_col(cx, slug, "field", field or "")


def set_lifespan(cx, slug, lifespan):
    _upsert_col(cx, slug, "lifespan", lifespan or "")


def set_vital_status(cx, slug, status):
    _upsert_col(cx, slug, "vital_status", status or "")


def set_lineage(cx, slug, lineage):
    """Store the intellectual-lineage chain as a JSON list of names."""
    _upsert_col(cx, slug, "lineage_json", json.dumps(lineage or []))


def set_sources(cx, slug, sources):
    """Store reference/source labels as a JSON list."""
    _upsert_col(cx, slug, "sources_json", json.dumps(sources or []))


def set_seo(cx, slug, seo):
    _upsert_col(cx, slug, "seo_json", json.dumps(seo or {}))


def list_pages(cx):
    """All mentor pages (any state), newest-updated first, for the console."""
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT mentor_slug, name, state, field, lifespan, vital_status, content_json, "
        "approved_at, updated_at FROM mentor_pages ORDER BY updated_at DESC"
    ).fetchall()
    out = []
    for r in rows:
        content = json.loads(r["content_json"] or "{}")
        out.append({
            "slug": r["mentor_slug"], "name": r["name"] or r["mentor_slug"],
            "state": r["state"] or "draft", "field": r["field"] or "",
            "lifespan": r["lifespan"] or "", "vital_status": r["vital_status"] or "",
            "sections": sorted(content.keys()),
            "approved_at": r["approved_at"] or "", "updated_at": r["updated_at"] or "",
        })
    return out


def list_public(cx):
    """Approved pages only, for the public index + sitemap. Alphabetical by name."""
    return sorted(
        [p for p in list_pages(cx) if p["state"] == "approved"],
        key=lambda p: (p.get("name") or "").lower(),
    )


# ---------------------------------------------------------------------------
# Request / notify (mirrors dashboard/ingredient_pages.py)
# ---------------------------------------------------------------------------

def record_request(cx, slug, email):
    """INSERT OR IGNORE - one row per (slug, email)."""
    init_table(cx)
    e = _norm(email)
    if not e:
        return
    cx.execute(
        "INSERT OR IGNORE INTO mentor_page_requests "
        "(mentor_slug, email, requested_at, emailed_at) VALUES (?, ?, ?, '')",
        (slug, e, _now()),
    )
    cx.commit()


def requesters_to_email(cx, slug):
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT email, requested_at FROM mentor_page_requests "
        "WHERE mentor_slug=? AND COALESCE(emailed_at,'')='' "
        "ORDER BY requested_at",
        (slug,),
    ).fetchall()
    return [{"email": r["email"], "requested_at": r["requested_at"]} for r in rows]


def mark_emailed(cx, slug, email):
    init_table(cx)
    cx.execute(
        "UPDATE mentor_page_requests SET emailed_at=? "
        "WHERE mentor_slug=? AND email=?",
        (_now(), slug, _norm(email)),
    )
    cx.commit()


def notify_on_approve(cx, slug, name, base_url, *, send, strip=None):
    """Email each un-emailed requester once; mark each after send; never raises."""
    if strip is None:
        strip = lambda s: s  # noqa: E731
    requesters = requesters_to_email(cx, slug)
    link = f"{base_url}/mentors/{slug}"
    subject = f"The {name} page is ready"
    for r in requesters:
        email = r["email"]
        body = strip(
            f"Aloha,\n\nThe page on {name} is ready:\n\n{link}\n\nIn wellness,\nDr. Glen & Rae"
        )
        try:
            send(email, subject, body)
            mark_emailed(cx, slug, email)
        except Exception as exc:  # noqa: BLE001 - one bad send must not stop the rest
            print(f"[mentor-pages] send failed for {email}: {exc}", flush=True)
