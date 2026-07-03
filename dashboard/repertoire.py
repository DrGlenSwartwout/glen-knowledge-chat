"""Per-member SKU repertoire (email-keyed). Members get a flat reorder price on
these SKUs; first buy of a SKU is regular. No per-SKU decay — pricing eligibility
is gated on active membership at read time, not stored here. Pure: caller passes cx."""
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_repertoire_table(cx):
    cx.execute(
        """CREATE TABLE IF NOT EXISTS repertoire (
             email TEXT NOT NULL,
             slug  TEXT NOT NULL,
             added_at TEXT NOT NULL,
             PRIMARY KEY (email, slug)
           )"""
    )
    cx.execute("CREATE INDEX IF NOT EXISTS ix_repertoire_email ON repertoire(email)")
    cx.commit()


def add_skus(cx, email, slugs, *, at=None):
    email = _norm(email)
    at = at or _now_iso()
    seen, added = set(), 0
    for s in slugs:
        s = (s or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        if cx.execute(
            "INSERT OR IGNORE INTO repertoire(email, slug, added_at) VALUES (?,?,?)",
            (email, s, at),
        ).rowcount == 1:
            added += 1
    cx.commit()
    return added


def repertoire_slugs(cx, email):
    email = _norm(email)
    return {
        r[0]
        for r in cx.execute("SELECT slug FROM repertoire WHERE email=?", (email,))
    }


def seed_from_history(cx, email, window_days, *, order_slugs_fn):
    slugs = order_slugs_fn(cx, _norm(email), window_days) or []
    return add_skus(cx, email, slugs)
