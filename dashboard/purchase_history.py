"""Slug-keyed client purchase history backfilled from external sources
(FMP, GrooveKart). Separate from the live `orders` board. Feeds repertoire
seeding only. Pure: caller passes cx."""
from datetime import datetime, timedelta, timezone

def _norm(v): return (v or "").strip().lower()

def init_purchase_history_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS purchase_history (
        email TEXT NOT NULL, slug TEXT NOT NULL,
        purchased_at TEXT NOT NULL, source TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        PRIMARY KEY (source, source_ref, slug))""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_ph_email ON purchase_history(email)")
    cx.commit()

def _default_resolve(slug):
    """Redirect a retired slug onto its live twin. Imported lazily: this module is
    otherwise pure (caller passes cx) and used in tests without the catalog."""
    from dashboard.products import superseded_slug
    return superseded_slug(slug)


def replace_source(cx, source, rows, *, resolve=None):
    """`resolve` defaults to the REAL catalog redirect, never a no-op — the 'groovekart'
    slice scrapes slugs out of storefront URLs and happily yields retired ones, and a
    dead slug stored here silently costs a member their repertoire reorder discount."""
    resolve = resolve or _default_resolve
    cx.execute("DELETE FROM purchase_history WHERE source=?", (source,))
    n = 0
    for email, slug, purchased_at, source_ref in rows:
        e, s = _norm(email), _norm(slug)
        if not (e and s):
            continue
        s = _norm(resolve(s))
        if not s:
            continue
        if cx.execute("INSERT OR IGNORE INTO purchase_history"
                      "(email, slug, purchased_at, source, source_ref) VALUES (?,?,?,?,?)",
                      (e, s, purchased_at, source, str(source_ref))).rowcount == 1:
            n += 1
    cx.commit()
    return n

def slugs_since(cx, email, window_days):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(window_days))).isoformat()
    return {r[0] for r in cx.execute(
        "SELECT DISTINCT slug FROM purchase_history WHERE email=? AND purchased_at>=?",
        (_norm(email), cutoff))}
