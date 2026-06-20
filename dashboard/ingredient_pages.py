"""Store and request/notify for ingredient pages.

Mirrors dashboard/sales_pages.py (page table + getters, per-cursor Row factory)
and dashboard/sales_page_viewers.py (record_request/requesters_to_email/mark_emailed/notify_on_approve).
"""
import datetime
import json
import sqlite3


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def _clamp_score(v):
    """Clamp a score to 1-10 as int, or return None if v is None."""
    if v is None:
        return None
    try:
        return max(1, min(10, int(v)))
    except (TypeError, ValueError):
        return None


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS ingredient_pages ("
        "ingredient_slug TEXT PRIMARY KEY, "
        "name TEXT DEFAULT '', "
        "state TEXT DEFAULT 'draft', "
        "content_json TEXT DEFAULT '{}', "
        "research_score INTEGER, "
        "traditional_score INTEGER, "
        "traditional_use_json TEXT DEFAULT '[]', "
        "related_forms_json TEXT DEFAULT '[]', "
        "model TEXT DEFAULT '', "
        "generated_at TEXT DEFAULT '', "
        "approved_at TEXT DEFAULT '', "
        "approved_by TEXT DEFAULT '', "
        "created_at TEXT DEFAULT '', "
        "updated_at TEXT DEFAULT '')"
    )
    cx.execute(
        "CREATE TABLE IF NOT EXISTS ingredient_page_requests ("
        "ingredient_slug TEXT, "
        "email TEXT, "
        "requested_at TEXT, "
        "emailed_at TEXT DEFAULT '', "
        "PRIMARY KEY(ingredient_slug, email))"
    )
    cx.commit()


def get_page(cx, slug):
    """Return a dict with parsed content/traditional_use/related_forms + scores + state, or None."""
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    row = cur.execute(
        "SELECT ingredient_slug, name, state, content_json, research_score, "
        "traditional_score, traditional_use_json, related_forms_json, "
        "model, generated_at, approved_at, approved_by, created_at, updated_at "
        "FROM ingredient_pages WHERE ingredient_slug=?", (slug,)
    ).fetchone()
    if not row:
        return None
    return {
        "ingredient_slug": row["ingredient_slug"],
        "name": row["name"] or "",
        "state": row["state"] or "draft",
        "content": json.loads(row["content_json"] or "{}"),
        "research_score": row["research_score"],
        "traditional_score": row["traditional_score"],
        "traditional_use": json.loads(row["traditional_use_json"] or "[]"),
        "related_forms": json.loads(row["related_forms_json"] or "[]"),
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


def upsert_section(cx, slug, section, text, model=""):
    init_table(cx)
    now = _now()
    row = cx.execute(
        "SELECT content_json FROM ingredient_pages WHERE ingredient_slug=?", (slug,)
    ).fetchone()
    content = json.loads(row[0]) if row and row[0] else {}
    content[section] = text
    cx.execute(
        "INSERT INTO ingredient_pages "
        "(ingredient_slug, state, content_json, model, generated_at, created_at, updated_at) "
        "VALUES (?, 'draft', ?, ?, ?, ?, ?) "
        "ON CONFLICT(ingredient_slug) DO UPDATE SET "
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
            "UPDATE ingredient_pages SET state=?, approved_at=?, approved_by=?, updated_at=? "
            "WHERE ingredient_slug=?", (state, now, by, now, slug)
        )
    else:
        cx.execute(
            "UPDATE ingredient_pages SET state=?, updated_at=? WHERE ingredient_slug=?",
            (state, now, slug),
        )
    cx.commit()


def set_scores(cx, slug, research, traditional):
    """Set research_score and traditional_score, each clamped to 1-10 or None."""
    init_table(cx)
    now = _now()
    r = _clamp_score(research)
    t = _clamp_score(traditional)
    cx.execute(
        "INSERT INTO ingredient_pages (ingredient_slug, research_score, traditional_score, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(ingredient_slug) DO UPDATE SET "
        "research_score=excluded.research_score, "
        "traditional_score=excluded.traditional_score, "
        "updated_at=excluded.updated_at",
        (slug, r, t, now, now),
    )
    cx.commit()


def set_related_forms(cx, slug, forms):
    """Store a list of related-form dicts as JSON."""
    init_table(cx)
    now = _now()
    cx.execute(
        "INSERT INTO ingredient_pages (ingredient_slug, related_forms_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(ingredient_slug) DO UPDATE SET "
        "related_forms_json=excluded.related_forms_json, "
        "updated_at=excluded.updated_at",
        (slug, json.dumps(forms or []), now, now),
    )
    cx.commit()


def set_traditional_use(cx, slug, entries):
    """Store a list of traditional-use dicts as JSON."""
    init_table(cx)
    now = _now()
    cx.execute(
        "INSERT INTO ingredient_pages (ingredient_slug, traditional_use_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(ingredient_slug) DO UPDATE SET "
        "traditional_use_json=excluded.traditional_use_json, "
        "updated_at=excluded.updated_at",
        (slug, json.dumps(entries or []), now, now),
    )
    cx.commit()


def set_name(cx, slug, name):
    """Set or update the human-readable name for this slug."""
    init_table(cx)
    now = _now()
    cx.execute(
        "INSERT INTO ingredient_pages (ingredient_slug, name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(ingredient_slug) DO UPDATE SET "
        "name=excluded.name, "
        "updated_at=excluded.updated_at",
        (slug, name or "", now, now),
    )
    cx.commit()


# ---------------------------------------------------------------------------
# Request / notify (mirrors dashboard/sales_page_viewers.py Phase-5b pattern)
# ---------------------------------------------------------------------------

def record_request(cx, slug, email):
    """INSERT OR IGNORE - one row per (slug, email)."""
    init_table(cx)
    e = _norm(email)
    if not e:
        return
    cx.execute(
        "INSERT OR IGNORE INTO ingredient_page_requests "
        "(ingredient_slug, email, requested_at, emailed_at) VALUES (?, ?, ?, '')",
        (slug, e, _now()),
    )
    cx.commit()


def requesters_to_email(cx, slug):
    """Return rows with emailed_at null/empty as list of dicts."""
    init_table(cx)
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        "SELECT email, requested_at FROM ingredient_page_requests "
        "WHERE ingredient_slug=? AND COALESCE(emailed_at,'')='' "
        "ORDER BY requested_at",
        (slug,),
    ).fetchall()
    return [{"email": r["email"], "requested_at": r["requested_at"]} for r in rows]


def mark_emailed(cx, slug, email):
    """Mark a single requester as emailed."""
    init_table(cx)
    cx.execute(
        "UPDATE ingredient_page_requests SET emailed_at=? "
        "WHERE ingredient_slug=? AND email=?",
        (_now(), slug, _norm(email)),
    )
    cx.commit()


def notify_on_approve(cx, slug, name, base_url, *, send, strip=None):
    """Email each un-emailed requester once; mark each after send; never raises.

    At-most-once: mark_emailed is called per requester immediately after a
    successful send so a subsequent call sees no un-emailed requesters.
    One failed send does not stop the others.
    """
    if strip is None:
        strip = lambda s: s  # noqa: E731

    requesters = requesters_to_email(cx, slug)
    link = f"{base_url}/begin/ingredient/{slug}"
    subject = f"Your {name} deep-dive is ready"

    for r in requesters:
        email = r["email"]
        body = strip(
            f"Aloha,\n\nYour {name} deep-dive is ready:\n\n{link}\n\nIn wellness,\nDr. Glen & Rae"
        )
        try:
            send(email, subject, body)
            mark_emailed(cx, slug, email)
        except Exception as exc:  # noqa: BLE001 - one bad send must not stop the rest
            print(f"[ingredient-pages] send failed for {email}: {exc}", flush=True)
