"""Email-keyed ebook entitlement store. A grant records that an email is
entitled to an ebook slug (from a site opt-in). Keyed by EMAIL, not the portal
token, because tokens rotate while the email identity is durable."""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx) -> None:
    cx.execute(
        "CREATE TABLE IF NOT EXISTS portal_library ("
        "  id INTEGER PRIMARY KEY,"
        "  email TEXT,"
        "  ebook_slug TEXT,"
        "  granted_at TEXT,"
        "  source_site TEXT,"
        "  UNIQUE(email, ebook_slug))")


def grant(cx, email, slug, source_site="") -> bool:
    email = (email or "").strip().lower()
    slug = (slug or "").strip()
    cur = cx.execute(
        "INSERT OR IGNORE INTO portal_library (email, ebook_slug, granted_at, source_site) "
        "VALUES (?,?,?,?)", (email, slug, _now(), (source_site or "").strip()))
    return cur.rowcount == 1


def list_for_email(cx, email) -> list:
    email = (email or "").strip().lower()
    rows = cx.execute(
        "SELECT ebook_slug, granted_at, source_site FROM portal_library "
        "WHERE email=? ORDER BY granted_at DESC, id DESC", (email,)).fetchall()
    return [{"slug": r[0], "granted_at": r[1], "source_site": r[2]} for r in rows]


def has(cx, email, slug) -> bool:
    email = (email or "").strip().lower()
    return cx.execute(
        "SELECT 1 FROM portal_library WHERE email=? AND ebook_slug=?",
        (email, (slug or "").strip())).fetchone() is not None
