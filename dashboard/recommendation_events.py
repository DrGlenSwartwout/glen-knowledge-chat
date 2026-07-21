"""Append-only per-client recommendation-provenance log + aggregates.
One row per counted action: (client_email, product_key=slug, source_key, occurred_at,
origin_ref). Idempotent on (client_email, product_key, source_key, origin_ref).
Pure: functions take an open sqlite connection."""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_recommendation_events(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_email TEXT NOT NULL,
            product_key  TEXT NOT NULL,
            source_key   TEXT NOT NULL,
            occurred_at  TEXT,
            origin_ref   TEXT NOT NULL DEFAULT '',
            created_at   TEXT NOT NULL
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_rec_events "
               "ON recommendation_events(client_email, product_key, source_key, origin_ref)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_rec_events_email "
               "ON recommendation_events(client_email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_hidden (
            client_email TEXT NOT NULL,
            product_key  TEXT NOT NULL,
            hidden_at    TEXT,
            PRIMARY KEY (client_email, product_key)
        )""")
    cx.commit()


def record_event(cx, email, product_key, source_key, *, occurred_at, origin_ref, commit=True):
    e = (email or "").strip().lower()
    pk = (product_key or "").strip()
    sk = (source_key or "").strip()
    if not e or not pk or not sk:
        return False
    cur = cx.execute(
        "INSERT OR IGNORE INTO recommendation_events "
        "(client_email, product_key, source_key, occurred_at, origin_ref, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (e, pk, sk, occurred_at, str(origin_ref or ""), _now()))
    if commit:
        cx.commit()
    return cur.rowcount == 1


def list_events(cx, email):
    e = (email or "").strip().lower()
    rows = cx.execute(
        "SELECT product_key, source_key, occurred_at, origin_ref FROM recommendation_events "
        "WHERE client_email=? ORDER BY id", (e,)).fetchall()
    return [{"product_key": r[0], "source_key": r[1], "occurred_at": r[2], "origin_ref": r[3]}
            for r in rows]


# NOTE: biofield ingest is intentionally NOT in Phase 1. Per the refined counting rule,
# a biofield event counts only when the client ACTS on a reveal (clicks to learn about a
# product, or orders it) - both are engagement signals needing the reveal-click tracking
# and order-line source capture that arrive in Phase 2. Until then the only source with a
# real recorded action is `purchased` (a paid order line). See the design spec, Phase 2.
def ingest_purchased(cx, email):
    """One purchased event per (line slug, PAID order). occurred_at = paid_at; origin_ref = order id."""
    from dashboard import orders
    try:
        rows = orders.list_orders_by_email(cx, email)
    except Exception:
        return 0
    n = 0
    for o in rows:
        if (o.get("pay_status") or "").strip().lower() != "paid":
            continue
        oid = o.get("id")
        occ = o.get("paid_at") or o.get("created_at") or ""
        for line in (o.get("items") or []):
            slug = (line.get("slug") or "").strip()
            if not slug:
                continue
            # dedup grain: one event per (line slug, order)
            if record_event(cx, email, slug, "purchased", occurred_at=occ, origin_ref=str(oid), commit=False):
                n += 1
    if n:
        cx.commit()
    return n


def set_hidden(cx, email, product_key, hidden=True):
    """Toggle the recommendation_hidden flag for one (client_email, product_key)."""
    e = (email or "").strip().lower()
    pk = (product_key or "").strip()
    if not e or not pk:
        return
    if hidden:
        cx.execute("INSERT OR REPLACE INTO recommendation_hidden "
                   "(client_email, product_key, hidden_at) VALUES (?,?,?)", (e, pk, _now()))
    else:
        cx.execute("DELETE FROM recommendation_hidden WHERE client_email=? AND product_key=?", (e, pk))
    cx.commit()


def product_sources(cx, email):
    """Per product: its sources (each with count, first_touch, last_touch), ordered by
    first_touch (icon order), plus a hidden flag. Callers sort/limit products for display."""
    e = (email or "").strip().lower()
    rows = cx.execute(
        "SELECT product_key, source_key, COUNT(*) n, MIN(occurred_at) ft, MAX(occurred_at) lt "
        "FROM recommendation_events WHERE client_email=? GROUP BY product_key, source_key",
        (e,)).fetchall()
    hidden = {r[0] for r in cx.execute(
        "SELECT product_key FROM recommendation_hidden WHERE client_email=?", (e,)).fetchall()}
    prods = {}
    for pk, sk, n, ft, lt in rows:
        p = prods.setdefault(pk, {"product_key": pk, "hidden": pk in hidden, "sources": []})
        p["sources"].append({"source": sk, "count": int(n),
                             "first_touch": ft or "", "last_touch": lt or ""})
    out = []
    for p in prods.values():
        p["sources"].sort(key=lambda s: s["first_touch"])
        out.append(p)
    return out
