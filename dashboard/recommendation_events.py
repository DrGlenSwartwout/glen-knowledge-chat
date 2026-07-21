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


def record_event(cx, email, product_key, source_key, *, occurred_at, origin_ref):
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
    cx.commit()
    return cur.rowcount == 1


def list_events(cx, email):
    e = (email or "").strip().lower()
    rows = cx.execute(
        "SELECT product_key, source_key, occurred_at, origin_ref FROM recommendation_events "
        "WHERE client_email=? ORDER BY id", (e,)).fetchall()
    return [{"product_key": r[0], "source_key": r[1], "occurred_at": r[2], "origin_ref": r[3]}
            for r in rows]


def ingest_biofield(cx, email):
    """One biofield event per (remedy slug, reveal). occurred_at/origin_ref = scan_date."""
    from dashboard import biofield_reveals
    try:
        rows = biofield_reveals.list_for_email(cx, email)
    except Exception:
        return 0
    n = 0
    for r in rows:
        sd = (r.get("scan_date") or "")
        for rem in (r.get("remedies") or []):
            slug = (rem.get("slug") or "").strip()
            if not slug:
                continue
            if record_event(cx, email, slug, "biofield", occurred_at=sd, origin_ref=sd):
                n += 1
    return n


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
            if record_event(cx, email, slug, "purchased", occurred_at=occ, origin_ref=str(oid)):
                n += 1
    return n
